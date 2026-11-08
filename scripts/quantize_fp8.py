#!/usr/bin/env python3
"""B1: llm-compressor oneshot による FP8 dynamic (W8A8) 量子化。llm-compressor の venv で実行する。

対象は言語モデルの Linear のみ。lm_head と Vision (visual) 系は ignore（BF16 のまま）。
FP8_DYNAMIC はキャリブレーションデータ不要。出力ディレクトリに quant_meta.json（全パラメータとバージョン）を残す。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from importlib import metadata

IGNORE = ["lm_head", "re:.*visual.*", "re:.*vision.*"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF のローカルディレクトリ（BF16）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    args = ap.parse_args()

    import torch
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoProcessor, AutoTokenizer

    try:  # マルチモーダル構成のまま読む（Vision を BF16 で残し、vLLM 側は --language-model-only で無視）
        from transformers import AutoModelForImageTextToText as Auto
        model = Auto.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    except Exception as e:  # noqa: BLE001
        print(f"AutoModelForImageTextToText で読めなかったので CausalLM にフォールバック: {e}")
        from transformers import AutoModelForCausalLM as Auto
        model = Auto.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")

    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
    t0 = time.time()
    oneshot(model=model, recipe=recipe)
    elapsed = time.time() - t0

    # 検証: Vision 系が量子化されていないこと / 言語側が量子化されていること
    quantized = [n for n, m in model.named_modules() if getattr(m, "quantization_scheme", None) is not None]
    bad = [n for n in quantized if "visual" in n or "vision" in n or n.endswith("lm_head")]
    print(f"量子化された Linear: {len(quantized)} 層 / 対象外のはずなのに量子化: {len(bad)}")
    if not quantized or bad:
        raise SystemExit(f"量子化対象の検証に失敗: quantized={len(quantized)} bad={bad[:5]}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, save_compressed=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
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
        "condition": "B1", "method": "llm-compressor oneshot", "scheme": "FP8_DYNAMIC", "targets": "Linear",
        "ignore": IGNORE, "calibration": None, "source_model": args.model_id,
        "quantized_linear_layers": len(quantized), "seconds": round(elapsed, 1),
        "versions": {p: ver(p) for p in ("llmcompressor", "compressed-tensors", "transformers", "torch")},
    }
    with open(os.path.join(args.out, "quant_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
