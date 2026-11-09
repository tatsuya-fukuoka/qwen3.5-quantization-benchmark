#!/usr/bin/env python3
"""vLLM 条件の評価（オフライン LLM クラス）。vLLM nightly の venv で実行すること。

言語モデルのみ: LLM(language_model_only=True) = CLI の --language-model-only 相当。
thinking OFF: chat_template_kwargs={"enable_thinking": False}。greedy: temperature=0, seed=0。
プロンプト整形は GGUF 側と同じ common.build_messages を通す。
PPL: 固定テキストを PPL_CTX トークンで分割し、各ウィンドウの後半のみ採点（llama-perplexity と同方式。
ただしトークナイズ・窓の切り方の細部は異なるため、Track をまたいだ PPL の絶対比較はしない）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def compute_ppl(llm, text: str) -> float:
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    ids = tok.encode(text)
    win = common.PPL_CTX
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0, seed=common.SEED)
    nll, cnt = 0.0, 0
    for s in range(0, len(ids) - win + 1, win):
        chunk = ids[s:s + win]
        out = llm.generate([{"prompt_token_ids": chunk}], sp, use_tqdm=False)[0]
        for pos in range(win // 2, win):
            lp = out.prompt_logprobs[pos][chunk[pos]].logprob
            nll -= lp
            cnt += 1
    return math.exp(nll / cnt) if cnt else float("nan")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, help="B1 など")
    ap.add_argument("--model", required=True, help="量子化済みモデルのディレクトリ")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n", type=int, default=100,
                    help="各タスクの先頭件数（既定 100 = Phase 0）。0 以下ならデータファイルの全行を評価")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="vLLM の max_num_seqs。未指定なら LLM() に渡さず vLLM 既定値のまま"
                         "（Mamba 層を含むモデルで既定 256 がキャッシュブロック数を超える場合に指定する）")
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--qualitative", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "prompts_qualitative.txt"))
    return ap


def build_llm_kwargs(args) -> dict:
    """LLM() に渡す引数。max_num_seqs は指定時のみ含める（未指定は vLLM 既定値）。"""
    kw = {"model": args.model, "max_model_len": common.CTX, "seed": common.SEED,
          "gpu_memory_utilization": args.gpu_mem_util, "language_model_only": True}
    if args.max_num_seqs is not None:
        kw["max_num_seqs"] = args.max_num_seqs
    return kw


def main() -> None:
    args = build_parser().parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    gen_dir = os.path.join(args.results_dir, args.condition)
    os.makedirs(gen_dir, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    with common.GpuMonitor() as mon:
        llm = LLM(**build_llm_kwargs(args))
        sp_cache: dict[int, SamplingParams] = {}

        def sp(mt: int) -> SamplingParams:
            if mt not in sp_cache:
                sp_cache[mt] = SamplingParams(temperature=0, max_tokens=mt, seed=common.SEED)
            return sp_cache[mt]

        def to_out(o):
            return {"text": o.outputs[0].text, "completion_tokens": len(o.outputs[0].token_ids)}

        def gen_batch(msgs_list, mt):
            outs = llm.chat(msgs_list, sp(mt), chat_template_kwargs=common.THINKING_KWARGS, use_tqdm=False)
            return [to_out(o) for o in outs]

        def gen_one(msgs, mt):
            return gen_batch([msgs], mt)[0]

        ev = common.run_eval(gen_one, gen_batch, args.data_dir, gen_dir, args.n, args.qualitative)
        ppl = compute_ppl(llm, open(os.path.join(args.data_dir, "ppl_ja.txt"), encoding="utf-8").read())
    vram = mon.peak_gb

    meta_path = os.path.join(args.model, "quant_meta.json")
    quant_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    metrics = {**ev["metrics"], "ppl": round(ppl, 4), "vram_gb": vram,
               "file_size_gb": common.path_size_gb(args.model)}
    record = {
        "condition": args.condition, "kind": "vLLM", "started": started,
        "model": {"id": args.model_id, "path": args.model, "text_only": True,
                  "quantization": quant_meta},
        "versions": {"vllm": vllm.__version__, "python": sys.version.split()[0],
                     "llmcompressor": (quant_meta or {}).get("versions", {}).get("llmcompressor")},
        "settings": {"n_per_task": args.n if args.n > 0 else "all", "ctx": common.CTX, "temperature": 0, "seed": common.SEED,
                     "thinking": "off (chat_template_kwargs enable_thinking=false)",
                     "max_tokens": common.MAX_TOKENS, "ppl_ctx": common.PPL_CTX,
                     "gpu_memory_utilization": args.gpu_mem_util,
                     "max_num_seqs": args.max_num_seqs,  # 未指定は null（vLLM 既定値）
                     "vram_note": "vLLM は gpu_memory_utilization 分を先取りするため、VRAM は GGUF と直接比較しない",
                     "file_size_note": "*.safetensors 合計（Vision Encoder の BF16 重みを含む）"},
        "gpu": common.gpu_name(),
        "n": ev["counts"], "metrics": metrics, "generations": ev["generations"],
    }
    common.write_result(args.results_dir, args.condition, record)


if __name__ == "__main__":
    main()
