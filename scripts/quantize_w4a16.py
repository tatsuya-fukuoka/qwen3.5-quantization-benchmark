#!/usr/bin/env python3
"""B2/B3: llm-compressor oneshot による W4A16 量子化（--method gptq|awq）。llm-compressor の venv で実行する。

quantize_fp8.py と同じ流儀: 対象は言語モデルの Linear のみ。lm_head と Vision 系は ignore（BF16 のまま）。
キャリブレーションは data_main/imatrix_calib.txt から固定 seed で 512 サンプルを抽出し、B2/B3 で同一にする。
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
from importlib import metadata

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
    ap.add_argument("--calib-text", default="data_main/imatrix_calib.txt")
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--awq-mappings-json", default=None,
                    help="AWQ の smoothing mappings を JSON で上書き（auto 推定が Qwen3.5 で失敗した場合の逃げ道）")
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cond = CONDITIONS[args.method]

    with open(args.calib_text, encoding="utf-8") as f:
        raw_text = f.read()
    samples = build_calibration_samples(raw_text, args.num_samples, args.seed)
    digest = samples_digest(samples)
    print(f"[{cond}] calibration: {len(samples)} samples, sha256={digest[:16]}…")

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
    ds = Dataset.from_dict({"text": samples}).map(
        lambda ex: tok(ex["text"], max_length=args.max_seq_len, truncation=True, add_special_tokens=False),
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
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=args.max_seq_len,
            num_calibration_samples=len(samples))
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
        "calibration": {"source_text": os.path.basename(args.calib_text), "num_samples": len(samples),
                        "seed": args.seed, "split": "non-empty lines, len>=%d chars, dedup, random.sample" % MIN_CHARS,
                        "max_seq_len": args.max_seq_len, "samples_sha256": digest},
        "source_model": args.model_id, "quantized_linear_layers": len(quantized), "seconds": round(elapsed, 1),
        "versions": {p: ver(p) for p in ("llmcompressor", "compressed-tensors", "transformers", "torch")},
    }
    with open(os.path.join(args.out, "quant_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
