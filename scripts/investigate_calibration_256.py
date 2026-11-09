#!/usr/bin/env python3
"""キャリブレーション 256 サンプル前提の再調査用計測スクリプト（docs/calibration-256-investigation.md の再現用）。

量子化スクリプト（quantize_w4a16.py）には依存せず、その挙動も変えない。GPU 不要。
自己完結（investigate_calibration.py を import しない）。評価データは使わない。重複・オーバーラップでの水増しはしない。

    python3 -I scripts/investigate_calibration_256.py /path/to/c4_en_ja_imatrix.txt \\
        --tokenizer /path/to/qwen-tok --json-out /tmp/calib256.json

--tokenizer は Qwen/Qwen3.5-9B のトークナイザファイル一式を置いたディレクトリ（または repo id）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics as st
import sys

MIN_CHARS = 64
SEED = 0
TARGET = 256
MAX_LEN = 512
SHORT_TAIL = 32


def split_docs(text: str) -> list[str]:
    return [d.strip() for d in re.split(r"\n\s*\n", text) if d.strip()]


def char_chunks(doc: str, size: int, min_chars: int = MIN_CHARS) -> list[str]:
    out = [doc[i:i + size] for i in range(0, len(doc), size)]
    return [c for c in out if len(c) >= min_chars]


def dist(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]  # noqa: E731
    return {"n": len(xs), "mean": round(st.fmean(xs), 1), "median": q(0.5), "p10": q(0.10),
            "p90": q(0.90), "min": xs[0], "max": xs[-1], "total": int(sum(xs))}


def digest_ids(seqs: list[list[int]]) -> str:
    h = hashlib.sha256()
    for s in seqs:
        h.update(json.dumps(s).encode())
        h.update(b"\n")
    return h.hexdigest()


def segments_f(ids: list[int], max_len: int = MAX_LEN, tail_policy: str = "keep", short_tail: int = SHORT_TAIL):
    """方式 F: 文書内を max_len で区切り、端数も 1 サンプルにする（文書はまたがない）。
    tail_policy: keep=(a) そのまま / merge=(b) short_tail 未満の端数は直前に結合 / drop=(c) short_tail 未満の端数は捨てる。"""
    segs = [ids[i:i + max_len] for i in range(0, len(ids), max_len)]
    if len(segs) >= 1 and len(segs[-1]) < short_tail and tail_policy != "keep":
        tail = segs.pop()
        if tail_policy == "merge":
            if segs:
                segs[-1] = segs[-1] + tail
            else:
                segs.append(tail)  # 文書全体が短い場合は結合先が無いのでそのまま残す
        # drop: 捨てる（文書全体が短い場合も捨てる）
    return segs


def pick(pool: list, n: int, seed: int):
    return random.Random(seed).sample(pool, n) if len(pool) >= n else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calib_text")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids_of = lambda texts: tok(texts, add_special_tokens=False)["input_ids"]  # noqa: E731

    with open(args.calib_text, "rb") as f:
        raw = f.read()
    docs = split_docs(raw.decode("utf-8"))
    doc_ids = ids_of(docs)
    R: dict = {"file_sha256": hashlib.sha256(raw).hexdigest(), "docs": len(docs),
               "tokens_total_per_doc_sum": sum(map(len, doc_ids)),
               "tokenizer": {"class": type(tok).__name__, "vocab": len(tok)}}
    R["current_scheme"] = {"samples": len(docs), "tokens_trunc512": sum(min(len(x), 512) for x in doc_ids),
                           "tokens_dropped_by_trunc512": sum(max(0, len(x) - 512) for x in doc_ids)}

    # ---- 方式 F ----
    F: dict = {}
    for name in ("keep", "merge", "drop"):
        pool = [s for d in doc_ids for s in segments_f(d, MAX_LEN, name)]
        lens = [len(s) for s in pool]
        sel = pick(pool, TARGET, args.seed)
        F[name] = {"pool": dist(lens), "ge_256": len(pool) >= TARGET,
                   "n_short_lt32": sum(1 for n in lens if n < SHORT_TAIL),
                   "n_full_512": sum(1 for n in lens if n == MAX_LEN),
                   "n_over_512": sum(1 for n in lens if n > MAX_LEN),
                   "sample256": dist([len(s) for s in sel]) if sel else None,
                   "sample256_sha256": digest_ids(sel) if sel else None}
        if name == "keep":
            again = pick(pool, TARGET, args.seed)
            F[name]["deterministic"] = (again == sel)
            # 文書をまたいでいないことの確認: 全セグメントを連結すると元の文書 ID 列に戻る
            F[name]["roundtrip_ok"] = all(sum(segments_f(d, MAX_LEN, "keep"), []) == d for d in doc_ids)
            # 現行方式で捨てていた部分（512 超過分）が F では使われる量
            F[name]["tokens_beyond_512_of_docs"] = R["current_scheme"]["tokens_dropped_by_trunc512"]
            F[name]["docs_longer_than_512"] = sum(1 for d in doc_ids if len(d) > MAX_LEN)
            F[name]["docs_le_512"] = sum(1 for d in doc_ids if len(d) <= MAX_LEN)
    R["F_variable_segments"] = F

    # ---- C'-256: 文書ごと 256 固定 ----
    def fixed(L):
        return [d[i:i + L] for d in doc_ids for i in range(0, len(d) - L + 1, L)]
    for L, key in ((256, "Cprime_256"), (128, "Cprime_128_ref")):
        pool = fixed(L)
        n = TARGET if L == 256 else 512
        sel = pick(pool, n, args.seed)
        R[key] = {"pool": len(pool), "n": n, "ok": bool(sel), "total_tokens": sum(map(len, sel)), "sha256": digest_ids(sel) if sel else None}
    # 参考: 512 サンプル × 系列長 256 / 512
    R["ref_512_samples"] = {"len256_per_doc": len(fixed(256)), "len512_per_doc": len(fixed(512)),
                            "concat_len256": len(ids_of(["\n".join(docs)])[0]) // 256,
                            "concat_len512": len(ids_of(["\n".join(docs)])[0]) // 512,
                            "concat_total_tokens": len(ids_of(["\n".join(docs)])[0])}

    # ---- B-256: 文字チャンク ----
    for size in (512, 1000, 1500):
        pool = sorted({c for d in docs for c in char_chunks(d, size)})
        sel = pick(pool, TARGET, args.seed)
        ln = [len(x) for x in ids_of(sel)] if sel else []
        pool_ln = [len(x) for x in ids_of(pool)]
        R[f"B_char{size}"] = {"pool": len(pool), "ok": bool(sel), "pool_tokens": dist(pool_ln),
                              "sample256_tokens": dist(ln) if ln else None}

    out = json.dumps(R, ensure_ascii=False, indent=2)
    print(out)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    sys.exit(main())
