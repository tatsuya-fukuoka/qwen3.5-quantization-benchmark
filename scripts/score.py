#!/usr/bin/env python3
"""採点のみを分離したモジュール。

- JCommonsenseQA : accuracy（出力から選択肢番号 0-4 を抽出）
- JSQuAD         : F1 / EM（文字単位。NFKC 正規化・空白と句読記号を除去、正解複数時は最大値）
- MGSM           : 数値の exact match（「答え: N」を優先、無ければ最後の数値）

どの条件・どのランタイムでも同一の採点コードを通す。
CLI:  python score.py --task jsquad --data data/jsquad.jsonl --gens results/A0/gen_jsquad.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from typing import Iterable, Optional


# ---------------------------------------------------------------- JCommonsenseQA
def extract_choice(text: str) -> Optional[int]:
    """出力の最初に現れる 0-4 の数字を選択肢番号として返す。無ければ None。"""
    t = unicodedata.normalize("NFKC", text or "")
    m = re.search(r"[0-4]", t)
    return int(m.group(0)) if m else None


def score_jcommonsenseqa(preds: list[str], items: list[dict]) -> dict:
    assert len(preds) == len(items), "preds と items の件数が違う"
    correct = sum(extract_choice(p) == int(it["label"]) for p, it in zip(preds, items))
    n = len(items)
    return {"acc": correct / n if n else 0.0, "n": n}


# ---------------------------------------------------------------- JSQuAD
_PUNCT_CATEGORIES = ("P", "S")  # 句読点・記号 (Unicode カテゴリ P*, S*)


def normalize_ja(s: str) -> str:
    """NFKC → 小文字化 → 空白と句読点・記号を除去。"""
    s = unicodedata.normalize("NFKC", s or "").lower()
    return "".join(
        c for c in s if not c.isspace() and unicodedata.category(c)[0] not in _PUNCT_CATEGORIES
    )


def _f1_em(pred: str, gold: str) -> tuple[float, float]:
    p, g = normalize_ja(pred), normalize_ja(gold)
    em = 1.0 if p == g else 0.0
    if not p or not g:
        return (1.0 if p == g else 0.0), em
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, em
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall), em


def score_jsquad(preds: list[str], items: list[dict]) -> dict:
    assert len(preds) == len(items), "preds と items の件数が違う"
    f1s, ems = [], []
    for p, it in zip(preds, items):
        # 出力が複数行ならまず先頭行（「答え:」等の接頭辞は除く）を答えとみなす
        first = (p or "").strip().splitlines()[0] if (p or "").strip() else ""
        first = re.sub(r"^\s*(答え|回答)\s*[:：]\s*", "", first)
        best_f1, best_em = 0.0, 0.0
        for gold in it["answers"]:
            f1, em = _f1_em(first, gold)
            best_f1, best_em = max(best_f1, f1), max(best_em, em)
        f1s.append(best_f1)
        ems.append(best_em)
    n = len(items)
    return {"f1": sum(f1s) / n if n else 0.0, "em": sum(ems) / n if n else 0.0, "n": n}


# ---------------------------------------------------------------- MGSM
_NUM = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"


def extract_number(text: str) -> Optional[float]:
    """「答え: N」の最後の出現を優先し、無ければ文中最後の数値を返す。"""
    t = unicodedata.normalize("NFKC", text or "")
    ans = re.findall(r"答え\s*[:：]?\s*[^\d\-.\n]{0,3}(" + _NUM + ")", t)
    cands = ans if ans else re.findall(_NUM, t)
    if not cands:
        return None
    try:
        return float(cands[-1].replace(",", ""))
    except ValueError:
        return None


def score_mgsm(preds: list[str], items: list[dict]) -> dict:
    assert len(preds) == len(items), "preds と items の件数が違う"
    correct = 0
    for p, it in zip(preds, items):
        v = extract_number(p)
        gold = float(str(it["answer"]).replace(",", ""))
        if v is not None and abs(v - gold) < 1e-6:
            correct += 1
    n = len(items)
    return {"acc": correct / n if n else 0.0, "n": n}


SCORERS = {
    "jcommonsenseqa": score_jcommonsenseqa,
    "jsquad": score_jsquad,
    "mgsm": score_mgsm,
}


def score_task(task: str, preds: Iterable[str], items: list[dict]) -> dict:
    return SCORERS[task](list(preds), items)


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(SCORERS))
    ap.add_argument("--data", required=True, help="prep_data.py が作った jsonl")
    ap.add_argument("--gens", required=True, help='{"id":..., "output":...} の jsonl')
    args = ap.parse_args()
    items = _read_jsonl(args.data)
    gens = {g["id"]: g["output"] for g in _read_jsonl(args.gens)}
    items = [it for it in items if it["id"] in gens]
    print(json.dumps(score_task(args.task, [gens[it["id"]] for it in items], items), ensure_ascii=False))


if __name__ == "__main__":
    main()
