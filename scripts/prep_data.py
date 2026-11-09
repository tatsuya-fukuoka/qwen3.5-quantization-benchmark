#!/usr/bin/env python3
"""評価データの取得と整形（スモーク用に先頭 N 件へ絞る）。

出力（--outdir、既定 data/。.gitignore 対象）:
  jcommonsenseqa.jsonl  {id, question, choices[5], label}      JGLUE 公式 valid-v1.3
  jsquad.jsonl          {id, context, question, answers[]}      JGLUE 公式 valid-v1.3（SQuAD 形式）
  mgsm.jsonl            {id, question, answer}                  MGSM 公式 mgsm_ja.tsv
  ppl_ja.txt            日本語 perplexity 用の固定テキスト
  imatrix_calib.txt     imatrix 用（--with-imatrix 指定時のみ。HF から取得）

各タスクは「ファイル中の出現順の先頭 N 件」（乱択しない＝再現性）。
ppl_ja.txt は JSQuAD valid の Wikipedia 由来 context のうち、評価に使う先頭 N 件の質問が
参照しない context を出現順に連結し、--ppl-chars 文字で切ったもの。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request

JGLUE = "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets"
URLS = {
    "jcommonsenseqa": f"{JGLUE}/jcommonsenseqa-v1.3/valid-v1.3.json",
    "jsquad": f"{JGLUE}/jsquad-v1.3/valid-v1.3.json",
    "mgsm": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_ja.tsv",
}
IMATRIX_REPO = "TFMC/imatrix-dataset-for-japanese-llm"
IMATRIX_FILE = "c4_en_ja_imatrix.txt"


def fetch(url: str, dest: str) -> str:
    if not os.path.exists(dest):
        print(f"download {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prep_jcommonsenseqa(raw: str, n: int) -> list[dict]:
    rows = []
    with open(raw, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)  # JSON Lines
            rows.append(
                {
                    "id": str(d["q_id"]),
                    "question": d["question"],
                    "choices": [d[f"choice{i}"] for i in range(5)],
                    "label": int(d["label"]),
                }
            )
            if len(rows) >= n:
                break
    return rows


def _jsquad_flat(raw: str):
    """(context_index, context, qa) を出現順に返す。"""
    with open(raw, encoding="utf-8") as f:
        data = json.load(f)
    ci = 0
    for article in data["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                yield ci, para["context"], qa
            ci += 1


def prep_jsquad(raw: str, n: int) -> tuple[list[dict], list[str]]:
    """評価用 N 件と、PPL 用に使える（評価で未使用の）context のリスト。"""
    rows, used_ctx, all_ctx = [], set(), {}
    for ci, ctx, qa in _jsquad_flat(raw):
        all_ctx.setdefault(ci, ctx)
        if len(rows) < n:
            used_ctx.add(ci)
            answers = []
            for a in qa["answers"]:
                if a["text"] not in answers:
                    answers.append(a["text"])
            rows.append({"id": qa["id"], "context": ctx, "question": qa["question"], "answers": answers})
    ppl_ctx = [c for i, c in all_ctx.items() if i not in used_ctx]
    return rows, ppl_ctx


def prep_mgsm(raw: str, n: int) -> list[dict]:
    rows = []
    with open(raw, encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.reader(f, delimiter="\t")):
            if len(r) < 2:
                continue
            rows.append({"id": f"mgsm_ja_{i:03d}", "question": r[0].strip(), "answer": r[1].strip()})
            if len(rows) >= n:
                break
    return rows


def build_ppl_text(contexts: list[str], max_chars: int) -> str:
    text = "\n\n".join(contexts)
    return text[:max_chars]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--n", type=int, default=100, help="各タスクの先頭件数（スモーク=100）")
    ap.add_argument("--ppl-chars", type=int, default=50000)
    ap.add_argument("--with-imatrix", action="store_true", help="imatrix 用テキストも HF から取得する")
    args = ap.parse_args()

    raw_dir = os.path.join(args.outdir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raws = {
        "jcommonsenseqa": fetch(URLS["jcommonsenseqa"], os.path.join(raw_dir, "jcommonsenseqa_valid-v1.3.json")),
        "jsquad": fetch(URLS["jsquad"], os.path.join(raw_dir, "jsquad_valid-v1.3.json")),
        "mgsm": fetch(URLS["mgsm"], os.path.join(raw_dir, "mgsm_ja.tsv")),
    }

    jc = prep_jcommonsenseqa(raws["jcommonsenseqa"], args.n)
    js, ppl_ctx = prep_jsquad(raws["jsquad"], args.n)
    mg = prep_mgsm(raws["mgsm"], args.n)
    write_jsonl(os.path.join(args.outdir, "jcommonsenseqa.jsonl"), jc)
    write_jsonl(os.path.join(args.outdir, "jsquad.jsonl"), js)
    write_jsonl(os.path.join(args.outdir, "mgsm.jsonl"), mg)
    ppl = build_ppl_text(ppl_ctx, args.ppl_chars)
    with open(os.path.join(args.outdir, "ppl_ja.txt"), "w", encoding="utf-8") as f:
        f.write(ppl)
    print(f"jcommonsenseqa={len(jc)} jsquad={len(js)} mgsm={len(mg)} ppl_chars={len(ppl)}")
    for name, rows in (("jcommonsenseqa", jc), ("jsquad", js), ("mgsm", mg)):
        if len(rows) < args.n:
            raise SystemExit(f"{name}: {len(rows)} 件しか取れていない（要求 {args.n}）")

    if args.with_imatrix:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(IMATRIX_REPO, IMATRIX_FILE, repo_type="dataset", local_dir=args.outdir)
        os.replace(p, os.path.join(args.outdir, "imatrix_calib.txt"))
        print("imatrix calibration text ->", os.path.join(args.outdir, "imatrix_calib.txt"))


if __name__ == "__main__":
    main()
