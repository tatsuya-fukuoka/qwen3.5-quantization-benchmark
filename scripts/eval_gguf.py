#!/usr/bin/env python3
"""GGUF 条件の評価。llama-server を OpenAI 互換 API (/v1/chat/completions) で叩く。

thinking OFF: サーバ起動時 --chat-template-kwargs と、各リクエストの chat_template_kwargs の両方で指定。
greedy: temperature=0, seed=0。コンテキスト長 8192。リクエストは 1 件ずつ逐次。
PPL は llama-perplexity を別プロセスで実行（サーバ停止後）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def llama_version(llama_dir: str) -> str:
    try:
        out = subprocess.run([os.path.join(llama_dir, "build/bin/llama-server"), "--version"],
                             capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip().splitlines()[-1] if (out.stdout + out.stderr).strip() else "unknown"
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def start_server(llama_dir: str, gguf: str, port: int, log_path: str) -> subprocess.Popen:
    cmd = [
        os.path.join(llama_dir, "build/bin/llama-server"),
        "-m", gguf, "-c", str(common.CTX), "-ngl", "99", "-np", "1",
        "--host", "127.0.0.1", "--port", str(port),
        "--jinja", "--chat-template-kwargs", json.dumps(common.THINKING_KWARGS),
        "--seed", str(common.SEED),
    ]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server が終了した (code {proc.returncode})。ログ: {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        time.sleep(3)
    proc.kill()
    raise TimeoutError("llama-server の起動待ちがタイムアウト")


def make_gen_one(port: int):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    def gen_one(messages, max_tokens):
        body = {
            "messages": messages, "max_tokens": max_tokens, "temperature": 0, "top_k": 1,
            "seed": common.SEED, "chat_template_kwargs": common.THINKING_KWARGS, "stream": False,
        }
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
        return {"text": d["choices"][0]["message"].get("content") or "",
                "completion_tokens": d["usage"]["completion_tokens"]}

    return gen_one


def run_perplexity(llama_dir: str, gguf: str, text_path: str, log_path: str) -> float | None:
    cmd = [os.path.join(llama_dir, "build/bin/llama-perplexity"), "-m", gguf, "-f", text_path,
           "-c", str(common.PPL_CTX), "-ngl", "99", "--seed", str(common.SEED)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    with open(log_path, "w") as f:
        f.write(out.stdout + "\n" + out.stderr)
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", out.stdout + out.stderr)
    return float(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, help="A0 / A3 / A3i など")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama-dir", default="llama.cpp")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--quant", default="", help="記録用の量子化名 (F16 / Q4_K_M ...)")
    ap.add_argument("--imatrix", default="", help="使用した imatrix のパス（記録用）")
    ap.add_argument("--imatrix-chunks", type=int, default=0)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--qualitative", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "prompts_qualitative.txt"))
    args = ap.parse_args()

    gen_dir = os.path.join(args.results_dir, args.condition)
    os.makedirs(gen_dir, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    proc = start_server(args.llama_dir, args.gguf, args.port, os.path.join(gen_dir, "server.log"))
    try:
        gen_one = make_gen_one(args.port)
        with common.GpuMonitor() as mon:
            ev = common.run_eval(
                gen_one, lambda msgs, mt: [gen_one(m, mt) for m in msgs],
                args.data_dir, gen_dir, args.n, args.qualitative)
        vram = mon.peak_gb
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    ppl = run_perplexity(args.llama_dir, args.gguf, os.path.join(args.data_dir, "ppl_ja.txt"),
                         os.path.join(gen_dir, "perplexity.log"))

    metrics = {**ev["metrics"], "ppl": ppl, "vram_gb": vram, "file_size_gb": common.path_size_gb(args.gguf)}
    record = {
        "condition": args.condition, "kind": "GGUF", "started": started,
        "model": {"id": args.model_id, "file": args.gguf, "quant": args.quant,
                  "imatrix": args.imatrix or None, "imatrix_chunks": args.imatrix_chunks or None,
                  "text_only": True},
        "versions": {"llama.cpp": llama_version(args.llama_dir), "python": sys.version.split()[0]},
        "settings": {"n_per_task": args.n, "ctx": common.CTX, "temperature": 0, "seed": common.SEED,
                     "thinking": "off (chat_template_kwargs enable_thinking=false)",
                     "max_tokens": common.MAX_TOKENS, "ppl_ctx": common.PPL_CTX,
                     "vram_note": "評価中の GPU 使用メモリ総量のピーク (nvidia-smi)"},
        "gpu": common.gpu_name(),
        "n": ev["counts"], "metrics": metrics, "generations": ev["generations"],
    }
    common.write_result(args.results_dir, args.condition, record)


if __name__ == "__main__":
    main()
