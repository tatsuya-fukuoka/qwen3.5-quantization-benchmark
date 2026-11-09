#!/usr/bin/env python3
"""B2/B3: llm-compressor oneshot による W4A16 量子化（--method gptq|awq）。llm-compressor の venv で実行する。

quantize_fp8.py と同じ流儀: 対象は言語モデルの Linear のみ。lm_head と Vision 系は ignore（BF16 のまま）。
キャリブレーションは 2 経路。B2/B3 の本測定は --calib-packed（build_calibration.py が作る 256 x 1024 のハイブリッド・パック済み
トークン ID 列。B2/B3 で完全同一）。--calib-text は後方互換の行ベース経路（imatrix_calib.txt から seed 固定で 512 行）。
出力ディレクトリに quant_meta.json（量子化パラメータ・キャリブレーション条件・バージョン）を残す。

純 Python の部分（IGNORE / build_calibration_samples / recipe_spec）は torch 無しで import・テストできる。
llm-compressor / torch / transformers の import は main() 内に閉じ込めている。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import sys
from importlib import metadata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_pack  # noqa: E402

IGNORE = ["lm_head", "re:.*visual.*", "re:.*vision.*"]  # quantize_fp8.py と同一
SCHEME = "W4A16"
TARGETS = "Linear"
CONDITIONS = {"gptq": "B2", "awq": "B3"}
NUM_SAMPLES = 512
SEED = 0
MAX_SEQ_LEN = 512      # トークン数の上限（oneshot の max_seq_length / トークナイズ時の truncation）
MIN_CHARS = 64         # これより短い行はキャリブレーションに使わない
MAX_CHARS = 4000       # 極端に長い行は先頭だけ使う（トークナイズ前の安全弁。実際の上限は MAX_SEQ_LEN）
GPTQ_PARAMS = {"dampening_frac": 0.01, "block_size": 128}  # GPTQModifier の既定値を明示して記録


def build_calibration_samples(text: str, n: int = NUM_SAMPLES, seed: int = SEED,
                              min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS) -> list[str]:
    """キャリブレーションテキストを改行で分割し、短すぎる行を除き、seed 固定で n 件を抽出する。

    - 分割: 空行を除く各行を 1 サンプルとする（c4_en_ja_imatrix.txt は 1 行 1 文書）。
    - 重複行は除く（初出順を保持）。
    - 抽出: random.Random(seed).sample（元の並び順に依存しない固定 seed）。件数不足なら例外。
    同じ入力・seed からは常に同一のリストが返る（B2/B3 で同一サンプルを使うための保証）。
    """
    seen: set[str] = set()
    lines: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if len(s) < min_chars or s in seen:
            continue
        seen.add(s)
        lines.append(s[:max_chars])
    if len(lines) < n:
        raise SystemExit(f"キャリブレーション候補が不足: {len(lines)} 行 < {n}（min_chars={min_chars}）")
    return random.Random(seed).sample(lines, n)


def samples_digest(samples: list[str]) -> str:
    h = hashlib.sha256()
    for s in samples:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_packed_calibration(path: str) -> tuple[list[list[int]], int, str, str]:
    """パック済みファイルを読み (samples, seq_len, file_sha256, samples_sha256) を返す。形状を検証する。"""
    with open(path, "rb") as f:
        raw = f.read()
    d = json.loads(raw)
    if d.get("format") != calib_pack.FORMAT:
        raise SystemExit(f"パック済みキャリブレーションの format が不正: {d.get('format')}")
    samples, seq_len = d["samples"], d["seq_len"]
    if len(samples) != d["num_samples"] or any(len(s) != seq_len for s in samples):
        raise SystemExit("パック済みキャリブレーションの形状が不正（全サンプルが seq_len ちょうどであること）")
    return samples, seq_len, hashlib.sha256(raw).hexdigest(), calib_pack.samples_digest_ids(samples)


def packed_calibration_meta(path: str, samples: list[list[int]], seq_len: int, file_sha: str, digest: str,
                            build_meta: dict | None) -> dict:
    """quant_meta.json の calibration 欄（hybrid-pack）。build_meta はパック時の .meta.json（無ければ None）。"""
    cal = {"method": "hybrid-pack", "source_file": os.path.basename(path), "num_samples": len(samples),
           "seq_len": seq_len, "max_seq_len": seq_len, "total_tokens": len(samples) * seq_len,
           "packed_sha256": file_sha, "samples_sha256": digest}
    if build_meta:
        cal.update({"seed": build_meta.get("seed"),
                    "existing_tokens": build_meta["existing"]["tokens_used"],
                    "supplement_tokens": build_meta["supplement"]["tokens_used"],
                    "existing_source": build_meta["existing"]["file"],
                    "existing_sha256": build_meta["existing"]["sha256"],
                    "supplement_source": {k: build_meta["supplement"].get(k) for k in ("dataset", "config", "revision", "data_file")},
                    "supplement_docs": build_meta["supplement"]["docs_consumed"],
                    "tokenizer": build_meta.get("tokenizer")})
    return cal


def recipe_spec(method: str, mappings: list | None = None) -> dict:
    """モディファイアの種類と引数を dict で返す（テスト・meta 記録用。llm-compressor 不要）。"""
    if method not in CONDITIONS:
        raise ValueError(f"unknown method: {method}")
    spec = {"modifier": "GPTQModifier" if method == "gptq" else "AWQModifier",
            "targets": TARGETS, "scheme": SCHEME, "ignore": list(IGNORE)}
    if method == "gptq":
        spec.update(GPTQ_PARAMS)
    elif mappings is not None:
        spec["mappings"] = mappings
    return spec


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--model", required=True, help="HF のローカルディレクトリ（BF16）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-text", default="data_main/imatrix_calib.txt", help="行ベース経路（後方互換）")
    ap.add_argument("--calib-packed", default=None,
                    help="build_calibration.py が作ったパック済み JSON（指定すると --calib-text 経路は使わない）")
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help=f"行ベース経路の上限（既定 {MAX_SEQ_LEN}）。パック経路では seq_len に固定され、小さい値はエラー")
    ap.add_argument("--awq-mappings-json", default=None,
                    help="AWQ の smoothing mappings を JSON で上書き（auto 推定が Qwen3.5 で失敗した場合の逃げ道）")
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cond = CONDITIONS[args.method]

    packed = None
    if args.calib_packed:
        ids, seq_len, file_sha, digest = load_packed_calibration(args.calib_packed)
        if args.max_seq_len is not None and args.max_seq_len < seq_len:
            raise SystemExit(f"--max-seq-len {args.max_seq_len} < パック済みの seq_len {seq_len}")
        mp = os.path.splitext(args.calib_packed)[0] + ".meta.json"
        build_meta = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else None
        if build_meta and build_meta.get("samples_sha256") != digest:
            raise SystemExit("パック済みファイルと .meta.json の samples_sha256 が不一致")
        packed = packed_calibration_meta(args.calib_packed, ids, seq_len, file_sha, digest, build_meta)
        max_seq_len, n_samples = seq_len, len(ids)
        print(f"[{cond}] calibration (hybrid-pack): {n_samples} x {seq_len}, samples_sha256={digest[:16]}…")
    else:
        max_seq_len = args.max_seq_len or MAX_SEQ_LEN
        with open(args.calib_text, encoding="utf-8") as f:
            raw_text = f.read()
        samples = build_calibration_samples(raw_text, args.num_samples, args.seed)
        digest = samples_digest(samples)
        n_samples = len(samples)
        print(f"[{cond}] calibration: {n_samples} samples, sha256={digest[:16]}…")

    mappings = None
    if args.awq_mappings_json:
        with open(args.awq_mappings_json, encoding="utf-8") as f:
            mappings = json.load(f)

    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from transformers import AutoProcessor, AutoTokenizer

    try:  # マルチモーダル構成のまま読む（quantize_fp8.py と同じ）
        from transformers import AutoModelForImageTextToText as Auto
        model = Auto.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    except Exception as e:  # noqa: BLE001
        print(f"AutoModelForImageTextToText で読めなかったので CausalLM にフォールバック: {e}")
        from transformers import AutoModelForCausalLM as Auto
        model = Auto.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")

    tok = AutoTokenizer.from_pretrained(args.model)
    if packed is not None:  # トークナイズ済み。0.15.0 は input_ids 列があれば再処理せずそのまま使う
        ds = Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * seq_len for _ in ids]})
    else:
        ds = Dataset.from_dict({"text": samples}).map(
            lambda ex: tok(ex["text"], max_length=max_seq_len, truncation=True, add_special_tokens=False),
            remove_columns=["text"])

    spec = recipe_spec(args.method, mappings)
    if args.method == "gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier
        recipe = GPTQModifier(targets=spec["targets"], scheme=spec["scheme"], ignore=spec["ignore"],
                              dampening_frac=spec["dampening_frac"], block_size=spec["block_size"])
    else:
        from llmcompressor.modifiers.awq import AWQModifier
        kw = {"mappings": mappings} if mappings is not None else {}
        recipe = AWQModifier(targets=[spec["targets"]], scheme=spec["scheme"], ignore=spec["ignore"], **kw)

    t0 = time.time()
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=max_seq_len,
            num_calibration_samples=n_samples)
    elapsed = time.time() - t0

    quantized = [n for n, m in model.named_modules() if getattr(m, "quantization_scheme", None) is not None]
    bad = [n for n in quantized if "visual" in n or "vision" in n or n.endswith("lm_head")]
    print(f"量子化された Linear: {len(quantized)} 層 / 対象外のはずなのに量子化: {len(bad)}")
    if not quantized or bad:
        raise SystemExit(f"量子化対象の検証に失敗: quantized={len(quantized)} bad={bad[:5]}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, save_compressed=True)
    tok.save_pretrained(args.out)
    try:
        AutoProcessor.from_pretrained(args.model).save_pretrained(args.out)
    except Exception as e:  # noqa: BLE001
        print(f"processor は保存しない: {e}")

    def ver(p: str):
        try:
            return metadata.version(p)
        except metadata.PackageNotFoundError:
            return None

    meta = {
        "condition": cond, "method": f"llm-compressor oneshot ({args.method.upper()})", "scheme": SCHEME,
        "targets": TARGETS, "ignore": IGNORE, "recipe": spec,
        "calibration": packed if packed is not None else {
            "source_text": os.path.basename(args.calib_text), "num_samples": n_samples,
            "seed": args.seed, "split": "non-empty lines, len>=%d chars, dedup, random.sample" % MIN_CHARS,
            "max_seq_len": max_seq_len, "samples_sha256": digest},
        "source_model": args.model_id, "quantized_linear_layers": len(quantized), "seconds": round(elapsed, 1),
        "versions": {p: ver(p) for p in ("llmcompressor", "compressed-tensors", "transformers", "torch")},
    }
    with open(os.path.join(args.out, "quant_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
