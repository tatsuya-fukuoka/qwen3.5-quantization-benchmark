"""eval_gguf.py / eval_vllm.py 共通部。

プロンプトの整形・生成設定・評価ループをここに一本化し、全条件で同一の整形コードを通す。
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Callable

import score as scoring

SEED = 0
CTX = 8192
THINKING_KWARGS = {"enable_thinking": False}  # thinking OFF 固定 (llama-server / vLLM 共通)
TASKS = ("jcommonsenseqa", "jsquad", "mgsm")
MAX_TOKENS = {"jcommonsenseqa": 16, "jsquad": 64, "mgsm": 768, "qualitative": 512}
PPL_CTX = 2048  # perplexity のウィンドウ長（後半のみ採点。llama-perplexity と同じ）


# ---------------------------------------------------------------- I/O
def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_qualitative(path: str) -> list[str]:
    prompts, buf = [], []
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines + ["====="]:
        if line.strip() == "=====":
            text = "\n".join(buf).strip()
            if text:
                prompts.append(text)
            buf = []
        elif not line.startswith("#"):
            buf.append(line)
    return prompts


# ---------------------------------------------------------------- プロンプト整形（全条件共通）
def build_messages(task: str, item: dict) -> list[dict]:
    if task == "jcommonsenseqa":
        choices = "\n".join(f"{i}: {c}" for i, c in enumerate(item["choices"]))
        user = (
            "以下の質問に対して、最も適切な選択肢の番号を1つだけ答えてください。\n\n"
            f"質問: {item['question']}\n\n選択肢:\n{choices}\n\n番号のみを出力してください。"
        )
    elif task == "jsquad":
        user = (
            "以下の文章を読んで、質問に対する答えを文章中の語句そのままで簡潔に答えてください。"
            "説明は不要で、答えのみを出力してください。\n\n"
            f"文章: {item['context']}\n\n質問: {item['question']}\n\n答え:"
        )
    elif task == "mgsm":
        user = (
            "次の算数の問題を解いてください。途中の計算を簡潔に示し、"
            "最後の行に「答え: 」に続けて数値のみ（単位・カンマなし）を書いてください。\n\n"
            f"問題: {item['question']}"
        )
    elif task == "qualitative":
        user = item["prompt"]
    else:
        raise ValueError(task)
    return [{"role": "user", "content": user}]


# ---------------------------------------------------------------- 計測補助
class GpuMonitor:
    """nvidia-smi を 1 秒間隔で読み、GPU 使用メモリのピーク(GB)を記録する。"""

    def __init__(self, interval: float = 1.0):
        self.peak_mib = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, args=(interval,), daemon=True)

    def _run(self, interval: float) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.split()
                if out:
                    self.peak_mib = max(self.peak_mib, int(out[0]))
            except Exception:
                pass
            self._stop.wait(interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)

    @property
    def peak_gb(self) -> float:
        return round(self.peak_mib / 1024, 2)


def gpu_name() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def path_size_gb(path: str) -> float:
    """ファイルなら実サイズ、ディレクトリなら配下の *.safetensors の合計（GB, 10^9）。"""
    if os.path.isfile(path):
        return round(os.path.getsize(path) / 1e9, 3)
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files if f.endswith(".safetensors"))
    return round(total / 1e9, 3)


# ---------------------------------------------------------------- 評価ループ（共通）
def select_items(rows: list[dict], n: int) -> list[dict]:
    return rows[:n] if n > 0 else rows


GenOne = Callable[[list[dict], int], dict]  # (messages, max_tokens) -> {"text", "completion_tokens"}
GenBatch = Callable[[list[list[dict]], int], list[dict]]


def run_eval(gen_one: GenOne, gen_batch: GenBatch, data_dir: str, gen_dir: str, n: int,
             qualitative_path: str) -> dict:
    """3 タスクの採点と定性評価・速度計測を行い、metrics と生成物パスを返す。

    n > 0: 各データファイルの先頭 n 件（Phase 0 の挙動）。n <= 0: データファイルの全行。
    """
    metrics: dict = {}
    counts: dict = {}
    gen_paths: dict = {}
    for task in TASKS:
        items = select_items(read_jsonl(os.path.join(data_dir, f"{task}.jsonl")), n)
        t0 = time.time()
        outs = gen_batch([build_messages(task, it) for it in items], MAX_TOKENS[task])
        elapsed = time.time() - t0
        texts = [o["text"] for o in outs]
        res = scoring.score_task(task, texts, items)
        counts[task] = res.pop("n")
        for k, v in res.items():
            metrics[{"jcommonsenseqa": "jcsqa_", "jsquad": "jsquad_", "mgsm": "mgsm_"}[task] + k] = round(v, 4)
        path = os.path.join(gen_dir, f"gen_{task}.jsonl")
        write_jsonl(path, [{"id": it["id"], "output": t, "completion_tokens": o["completion_tokens"]}
                           for it, t, o in zip(items, texts, outs)])
        gen_paths[task] = path
        print(f"[{task}] {res} n={counts[task]} ({elapsed:.0f}s)", flush=True)

    # 定性評価 + 速度（1 リクエストずつ逐次。速度は Track 内でのみ比較する）
    prompts = load_qualitative(qualitative_path)
    gen_one(build_messages("qualitative", {"prompt": prompts[0]}), 16)  # ウォームアップ
    rows, tok, sec = [], 0, 0.0
    for i, p in enumerate(prompts):
        t0 = time.time()
        o = gen_one(build_messages("qualitative", {"prompt": p}), MAX_TOKENS["qualitative"])
        dt = time.time() - t0
        tok, sec = tok + o["completion_tokens"], sec + dt
        rows.append({"id": f"q{i + 1:02d}", "prompt": p, "output": o["text"],
                     "completion_tokens": o["completion_tokens"], "seconds": round(dt, 2)})
    path = os.path.join(gen_dir, "qualitative.jsonl")
    write_jsonl(path, rows)
    gen_paths["qualitative"] = path
    counts["qualitative"] = len(prompts)
    metrics["tok_s"] = round(tok / sec, 2) if sec else None
    return {"metrics": metrics, "counts": counts, "generations": gen_paths}


def write_result(results_dir: str, condition: str, record: dict) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{condition}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print("wrote", path)
    return path
