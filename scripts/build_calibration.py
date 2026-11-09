#!/usr/bin/env python3
"""ハイブリッド・キャリブレーション（既存コーパス全量＋日本語公開テキスト補充）を 256 x 1024 にパックする。

出力: --out（パック済みトークン ID 列 JSON）と --out の拡張子を .meta.json にした構築メタ情報。
  python scripts/build_calibration.py --corpus data_main/imatrix_calib.txt --tokenizer models/Qwen3.5-9B \
      --out calib/hybrid_256x1024.json --decontam data_main/jcommonsenseqa.jsonl ... --hf-fineweb2
  python scripts/build_calibration.py --verify-only --out calib/hybrid_256x1024.json \
      --corpus data_main/imatrix_calib.txt --tokenizer models/Qwen3.5-9B

補充ソースは --hf-fineweb2（HuggingFaceFW/fineweb-2 jpn_Jpan を streaming）または --supplement-jsonl（{"id","text"} の
JSONL。テスト・オフライン用）。tokenizer は --tokenizer（HF ローカルディレクトリ）か --dummy-tokenizer（文字コード＝ID。テスト用）。
パッキング本体は calib_pack.py（torch 不要）。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_pack as cp  # noqa: E402

FINEWEB_REPO = "HuggingFaceFW/fineweb-2"
FINEWEB_SHARD = "data/jpn_Jpan/train/000_00000.parquet"   # 1 シャードに固定（約 4.8GB。streaming で先頭のみ読む）
SHUFFLE_BUFFER = 1000


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def meta_path(packed: str) -> str:
    return os.path.splitext(packed)[0] + ".meta.json"


def read_corpus_docs(text: str) -> list[str]:
    """既存コーパス: 空行を除く各行を 1 文書とする（フィルタ・重複除去・切り詰めなし＝全量）。"""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def make_tokenize(tokenizer_dir: str | None, dummy: bool):
    if dummy:
        return (lambda s: [ord(c) for c in s]), {"name": "dummy-char-code", "fingerprint": None}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    tj = os.path.join(tokenizer_dir, "tokenizer.json")
    info = {"name": type(tok).__name__, "dir": os.path.basename(os.path.normpath(tokenizer_dir)),
            "vocab_size": len(tok), "fingerprint": sha256_file(tj) if os.path.exists(tj) else None}
    try:
        from importlib import metadata
        info["transformers"] = metadata.version("transformers")
    except Exception:  # noqa: BLE001
        pass
    return (lambda s: tok(s, add_special_tokens=False)["input_ids"]), info


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)


def iter_fineweb2(seed: int, revision: str | None):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    rev = revision or HfApi().dataset_info(FINEWEB_REPO).sha
    ds = load_dataset(FINEWEB_REPO, data_files=FINEWEB_SHARD, split="train", streaming=True, revision=rev)
    ds = ds.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)
    return ({"id": r["id"], "text": r["text"]} for r in ds), {
        "dataset": FINEWEB_REPO, "config": "jpn_Jpan", "split": "train", "data_file": FINEWEB_SHARD,
        "revision": rev, "streaming": True, "shuffle_seed": seed, "shuffle_buffer": SHUFFLE_BUFFER, "license": "ODC-By 1.0"}


def decontam_texts(paths: list[str]):
    """評価データ（jsonl は全文字列値、それ以外はテキスト全体）を文字列として列挙する。"""
    for p in paths:
        if p.endswith(".jsonl"):
            for row in iter_jsonl(p):
                for v in row.values():
                    if isinstance(v, str):
                        yield v
                    elif isinstance(v, list):
                        yield from (x for x in v if isinstance(x, str))
        else:
            with open(p, encoding="utf-8") as f:
                yield f.read()


def write_packed(path: str, samples: list[list[int]], seq_len: int) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"format": cp.FORMAT, "seq_len": seq_len, "num_samples": len(samples), "samples": samples},
                  f, separators=(",", ":"))
    return sha256_file(path)


def build(args) -> dict:
    tokenize, tok_info = make_tokenize(args.tokenizer, args.dummy_tokenizer)
    with open(args.corpus, encoding="utf-8") as f:
        raw = f.read()
    existing = [tokenize(d) for d in read_corpus_docs(raw)]
    e_tokens = sum(len(d) for d in existing)
    print(f"existing corpus: {len(existing)} docs, {e_tokens} tokens")

    shingles = cp.shingle_set(decontam_texts(args.decontam)) if args.decontam else None
    if args.supplement_jsonl:
        stream, src = iter_jsonl(args.supplement_jsonl), {"dataset": "local-jsonl",
                                                          "file_sha256": sha256_file(args.supplement_jsonl)}
    elif args.hf_fineweb2:
        stream, src = iter_fineweb2(args.seed, args.hf_revision)
    else:
        raise SystemExit("補充ソースを指定してください（--hf-fineweb2 または --supplement-jsonl）")

    sup = cp.collect_supplement(stream, tokenize, e_tokens, args.seq_len, args.num_samples + args.pool_margin,
                                shingles=shingles)
    res = cp.build_hybrid(existing, sup["docs"], args.seq_len, args.num_samples, args.seed)
    packed_sha = write_packed(args.out, res["samples"], args.seq_len)
    meta = {
        "method": "hybrid-pack", "format": cp.FORMAT, "num_samples": args.num_samples, "seq_len": args.seq_len,
        "total_tokens": args.num_samples * args.seq_len, "seed": args.seed, "pool_margin": args.pool_margin,
        "pool_chunks": res["pool_chunks"], "dropped_tail_tokens": res["dropped_tail_tokens"],
        "chunk_indices": res["chunk_indices"],
        "existing": {"file": os.path.basename(args.corpus), "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                     "docs": len(existing), "tokens_total": res["existing_tokens"],
                     "tokens_used": res["existing_tokens_used"]},
        "supplement": {**src, "tokens_fetched": res["supplement_tokens_fetched"],
                       "tokens_used": res["supplement_tokens_used"], "docs_consumed": len(sup["used"]),
                       "skipped": sup["skipped"], "doc_ids": sup["used"],
                       "per_doc_max_chars": 3000, "min_chars": 200,
                       "decontam": {"shingle_chars": cp.SHINGLE,
                                    "files": [os.path.basename(p) for p in (args.decontam or [])]}},
        "tokenizer": tok_info, "add_special_tokens": False, "doc_separator": None,
        "packed_sha256": packed_sha, "samples_sha256": cp.samples_digest_ids(res["samples"]),
    }
    with open(meta_path(args.out), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("num_samples", "seq_len", "total_tokens", "pool_chunks",
                                           "packed_sha256", "samples_sha256")}, indent=2))
    print(f"existing used {meta['existing']['tokens_used']} / supplement used {meta['supplement']['tokens_used']}")
    return meta


def verify(args) -> list[str]:
    """既存のパック済みファイルが現在の入力と整合するか。問題のリストを返す（空なら OK）。"""
    problems: list[str] = []
    if not os.path.exists(args.out) or not os.path.exists(meta_path(args.out)):
        return ["パック済みファイルまたはメタが無い"]
    meta = json.load(open(meta_path(args.out), encoding="utf-8"))
    if sha256_file(args.out) != meta["packed_sha256"]:
        problems.append("packed_sha256 不一致（ファイル破損または改変）")
    d = json.load(open(args.out))
    s = d["samples"]
    if d.get("format") != cp.FORMAT or len(s) != meta["num_samples"] or any(len(x) != meta["seq_len"] for x in s):
        problems.append("形状がメタと不一致")
    elif cp.samples_digest_ids(s) != meta["samples_sha256"]:
        problems.append("samples_sha256 不一致")
    if args.corpus and os.path.exists(args.corpus):
        got = hashlib.sha256(open(args.corpus, encoding="utf-8").read().encode("utf-8")).hexdigest()
        if got != meta["existing"]["sha256"]:
            problems.append("既存コーパスの sha256 が構築時と違う")
    if args.tokenizer:
        tj = os.path.join(args.tokenizer, "tokenizer.json")
        fp = meta["tokenizer"].get("fingerprint")
        if fp and os.path.exists(tj) and sha256_file(tj) != fp:
            problems.append("tokenizer.json の sha256 が構築時と違う（ID 列が別トークナイザ由来）")
    return problems


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data_main/imatrix_calib.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=None, help="HF ローカルディレクトリ")
    ap.add_argument("--dummy-tokenizer", action="store_true", help="文字コード＝ID（テスト用）")
    ap.add_argument("--hf-fineweb2", action="store_true")
    ap.add_argument("--hf-revision", default=None, help="fineweb-2 の revision（省略時は最新 sha を解決して記録）")
    ap.add_argument("--supplement-jsonl", default=None)
    ap.add_argument("--decontam", nargs="*", default=None, help="評価データ（jsonl / txt）。重複する補充文書を除く")
    ap.add_argument("--seq-len", type=int, default=cp.SEQ_LEN)
    ap.add_argument("--num-samples", type=int, default=cp.NUM_SAMPLES)
    ap.add_argument("--seed", type=int, default=cp.SEED)
    ap.add_argument("--pool-margin", type=int, default=cp.POOL_MARGIN)
    ap.add_argument("--verify-only", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.decontam:
        args.decontam = [p for a in args.decontam for p in (glob.glob(a) or [a])]
    if args.verify_only:
        probs = verify(args)
        for p in probs:
            print("VERIFY NG:", p)
        print("VERIFY OK" if not probs else "VERIFY FAILED")
        sys.exit(1 if probs else 0)
    if not args.tokenizer and not args.dummy_tokenizer:
        raise SystemExit("--tokenizer か --dummy-tokenizer が必要")
    build(args)


if __name__ == "__main__":
    main()
