#!/usr/bin/env python3
"""キャリブレーション 512 サンプル維持の調査用計測スクリプト（docs/calibration-512-investigation.md の再現用）。

量子化スクリプト（quantize_w4a16.py）には依存せず、その挙動も変えない。GPU 不要。
入力ファイルは引数で受ける（データ自体はコミットしない）:

    curl -sL -o /path/to/c4_en_ja_imatrix.txt \\
      https://huggingface.co/datasets/TFMC/imatrix-dataset-for-japanese-llm/resolve/main/c4_en_ja_imatrix.txt
    python3 scripts/investigate_calibration.py /path/to/c4_en_ja_imatrix.txt \\
      --tokenizer Qwen/Qwen3.5-9B --json-out /tmp/calib_stats.json

トークナイザ計測には `transformers`（または `tokenizers`）が要る。--no-tokenizer で文字ベースの集計だけ行える。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics as st
import sys

MIN_CHARS = 64          # quantize_w4a16.py の MIN_CHARS と同じ値（比較用。import はしない）
SEED = 0
TARGET = 512

JA_RE = re.compile(r"[぀-ゟ゠-ヿ一-鿿]")  # ひらがな・カタカナ・漢字
# 粗い文分割: 日本語の句点・感嘆/疑問（直後の空白は任意）、英語の . ! ? の後に空白
SENT_RE = re.compile(r"(?<=[。！？])|(?<=[.!?])\s+")


def split_docs(text: str) -> list[str]:
    """空行区切りで文書に分割する（c4_en_ja_imatrix.txt は文書間が空行）。"""
    return [d.strip() for d in re.split(r"\n\s*\n", text) if d.strip()]


def split_sentences(doc: str) -> list[str]:
    parts = [p.strip() for p in SENT_RE.split(doc.replace("\n", " "))]
    return [p for p in parts if p]


def char_chunks(doc: str, size: int, min_chars: int = MIN_CHARS) -> list[str]:
    """重なりなしの固定長文字チャンク。端数が min_chars 未満なら捨てる。"""
    out = [doc[i:i + size] for i in range(0, len(doc), size)]
    return [c for c in out if len(c) >= min_chars]


def dist(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]  # noqa: E731
    return {"n": len(xs), "mean": round(st.fmean(xs), 1), "stdev": round(st.pstdev(xs), 1),
            "min": xs[0], "p10": q(0.10), "median": q(0.5), "p90": q(0.90), "max": xs[-1]}


def digest(samples: list[str]) -> str:
    h = hashlib.sha256()
    for s in samples:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def tok_ids(tok, texts: list[str]) -> list[list[int]]:
    # quantize_w4a16.py と同じく add_special_tokens=False
    return tok(texts, add_special_tokens=False)["input_ids"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calib_text", help="c4_en_ja_imatrix.txt のパス")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    with open(args.calib_text, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")
    R: dict = {"file_bytes": len(raw), "file_sha256": hashlib.sha256(raw).hexdigest()}

    # ---- 1. 基礎統計 ----
    pieces = text.split("\n")
    docs = split_docs(text)
    lens = [len(d) for d in docs]
    ja = sum(len(JA_RE.findall(d)) for d in docs)
    R["split_by_newline"] = {"pieces": len(pieces), "blank": sum(1 for p in pieces if not p.strip())}
    R["docs"] = {"count": len(docs), "unique": len(set(docs)), "ge_min_chars": sum(1 for n in lens if n >= MIN_CHARS),
                 "total_chars": sum(lens), "chars": dist(lens), "ja_char_ratio": round(ja / sum(lens), 4)}
    sents = [s for d in docs for s in split_sentences(d)]
    R["sentences"] = {"total": len(sents), "ge_min_chars": sum(1 for s in sents if len(s) >= MIN_CHARS),
                      "unique_ge_min_chars": len({s for s in sents if len(s) >= MIN_CHARS}), "chars": dist([len(s) for s in sents])}
    R["char_chunks"] = {str(sz): sum(len(char_chunks(d, sz)) for d in docs) for sz in (400, 512, 700, 1024)}

    rng = random.Random(args.seed)
    sent_pool = sorted({s for s in sents if len(s) >= MIN_CHARS})  # 重複除去 + 順序非依存にしてから抽出
    A = rng.sample(sent_pool, TARGET) if len(sent_pool) >= TARGET else []
    chunk_pool = sorted({c for d in docs for c in char_chunks(d, 512)})
    B = random.Random(args.seed).sample(chunk_pool, TARGET) if len(chunk_pool) >= TARGET else []
    R["A_sentence_sample"] = {"ok": len(A) == TARGET, "chars": dist([len(s) for s in A]), "sha256": digest(A) if A else None}
    R["B_char512_sample"] = {"ok": len(B) == TARGET, "chars": dist([len(s) for s in B]), "sha256": digest(B) if B else None}
    # 決定性: 同じ入力・seed で 2 回作って一致するか（B2/B3 同一サンプルの前提）
    again = random.Random(args.seed).sample(sent_pool, TARGET) if A else []
    R["A_deterministic"] = (again == A)

    # ---- 2. トークナイザ計測 ----
    if not args.no_tokenizer:
        tok = load_tokenizer(args.tokenizer)
        R["tokenizer"] = {"name": args.tokenizer, "class": type(tok).__name__, "vocab": len(tok)}
        doc_ids = tok_ids(tok, docs)
        doc_tok = [len(x) for x in doc_ids]
        R["tokens"] = {"total_per_doc_sum": sum(doc_tok), "per_doc": dist(doc_tok),
                       "chars_per_token": round(sum(lens) / sum(doc_tok), 3),
                       "whole_text_single_pass": len(tok_ids(tok, [text])[0])}
        # 現行方式（1 文書 1 サンプル、512 で truncation）で実効的に使えるトークン量
        R["current_scheme_tokens_trunc512"] = sum(min(n, 512) for n in doc_tok)
        # C. 連結チャンク（文書間は改行 1 つで連結 → 全体トークナイズ → 固定長に切る。端数は捨てる）
        concat_ids = tok_ids(tok, ["\n".join(docs)])[0]
        R["C_concat_total_tokens"] = len(concat_ids)
        R["C_concat_chunks"] = {str(L): len(concat_ids) // L for L in (1024, 512, 256, 128, 64)}
        # C'. 文書ごとにチャンク化（文書境界をまたがない。端数は捨てる）
        R["C_perdoc_chunks"] = {str(L): sum(n // L for n in doc_tok) for L in (1024, 512, 256, 128, 64)}
        # 512 サンプルを得るのに必要な最大系列長（連結チャンク）
        R["C_concat_max_len_for_512"] = len(concat_ids) // TARGET
        # 512 サンプル * 系列長の総トークン量（参考）
        R["tokens_needed"] = {str(L): TARGET * L for L in (512, 256, 128)}
        # A/B のトークン長分布（512 truncation 前後）
        for key, S in (("A_sentence_sample", A), ("B_char512_sample", B)):
            if S:
                ids = tok_ids(tok, S)
                ln = [len(x) for x in ids]
                R[key]["tokens"] = dist(ln)
                R[key]["tokens_total"] = sum(ln)
                R[key]["tokens_total_trunc512"] = sum(min(n, 512) for n in ln)
        # 文プール全体の短文比率（トークン）
        sid = [len(x) for x in tok_ids(tok, sent_pool)]
        R["sentence_pool_tokens"] = dist(sid)
        R["sentence_pool_tokens_sum"] = sum(sid)
        # C の 1 サンプル目の例
        if concat_ids:
            R["C_example_512_first_chars"] = tok.decode(concat_ids[:512])[:120]

    out = json.dumps(R, ensure_ascii=False, indent=2)
    print(out)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    sys.exit(main())
