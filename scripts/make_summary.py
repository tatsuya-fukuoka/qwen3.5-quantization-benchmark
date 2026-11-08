#!/usr/bin/env python3
"""results/<condition>.json を集めて results/summary.tsv を作り、表示する。"""
import glob
import json
import os
import sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
ORDER = ["A0", "A1", "A2", "A3", "A3i", "A4", "A4i", "A5", "A5i", "B0", "B1", "B2", "B3"]
COLS = ["条件", "種別", "ファイルサイズGB", "JCSQA acc", "JSQuAD F1", "MGSM acc", "PPL", "tok/s", "VRAM GB"]


def fmt(v, nd=3):
    if v is None:
        return ""
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


recs = []
for p in glob.glob(os.path.join(RESULTS, "*.json")):
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    if "condition" in r and "metrics" in r:
        recs.append(r)
recs.sort(key=lambda r: ORDER.index(r["condition"]) if r["condition"] in ORDER else 99)

rows = []
for r in recs:
    m = r["metrics"]
    rows.append([r["condition"], r["kind"], fmt(m.get("file_size_gb"), 2), fmt(m.get("jcsqa_acc")),
                 fmt(m.get("jsquad_f1")), fmt(m.get("mgsm_acc")), fmt(m.get("ppl"), 3),
                 fmt(m.get("tok_s"), 1), fmt(m.get("vram_gb"), 1)])

out = os.path.join(RESULTS, "summary.tsv")
with open(out, "w", encoding="utf-8") as f:
    f.write("\t".join(COLS) + "\n")
    for row in rows:
        f.write("\t".join(row) + "\n")
print("\t".join(COLS))
for row in rows:
    print("\t".join(row))
print("->", out)
