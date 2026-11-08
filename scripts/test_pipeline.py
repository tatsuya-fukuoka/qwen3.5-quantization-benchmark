#!/usr/bin/env python3
"""GPU なしで通る配管テスト: prep_data の整形（上流と同形式のダミー入力）と common.run_eval をダミー生成器で確認。
    python scripts/test_pipeline.py
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import prep_data  # noqa: E402


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.d = self.tmp.name
        with open(f"{d}/jc.json", "w", encoding="utf-8") as f:  # JGLUE: JSON Lines
            for i in range(5):
                f.write(json.dumps({"q_id": i, "question": f"q{i}", "choice0": "a", "choice1": "b",
                                    "choice2": "c", "choice3": "d", "choice4": "e", "label": i % 5},
                                   ensure_ascii=False) + "\n")
        sq = {"data": [{"title": "t", "paragraphs": [
            {"context": f"文脈{c}", "qas": [
                {"id": f"c{c}q{k}", "question": "?", "answers": [{"text": "文脈", "answer_start": 0}]}
                for k in range(2)]} for c in range(6)]}]}
        with open(f"{d}/sq.json", "w", encoding="utf-8") as f:
            json.dump(sq, f, ensure_ascii=False)
        with open(f"{d}/m.tsv", "w", encoding="utf-8") as f:
            f.write("問題1\t18\n問題2\t1,234\n問題3\t7\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_prep_and_eval(self):
        d = self.d
        jc = prep_data.prep_jcommonsenseqa(f"{d}/jc.json", 3)
        js, ppl_ctx = prep_data.prep_jsquad(f"{d}/sq.json", 4)
        mg = prep_data.prep_mgsm(f"{d}/m.tsv", 3)
        self.assertEqual((len(jc), len(js), len(mg)), (3, 4, 3))
        self.assertEqual(ppl_ctx, ["文脈2", "文脈3", "文脈4", "文脈5"])  # 評価で使った context は PPL 用から除外
        for name, rows in (("jcommonsenseqa", jc), ("jsquad", js), ("mgsm", mg)):
            common.write_jsonl(f"{d}/{name}.jsonl", rows)

        def one(msgs, mt):
            u = msgs[0]["content"]
            text = "答え: 18" if "問題" in u else ("0" if "番号" in u else "文脈")
            return {"text": text, "completion_tokens": 5}

        res = common.run_eval(one, lambda ms, mt: [one(m, mt) for m in ms], d, f"{d}/gen", 3,
                              os.path.join(HERE, "prompts_qualitative.txt"))
        self.assertEqual(res["counts"]["qualitative"], 10)
        self.assertAlmostEqual(res["metrics"]["jcsqa_acc"], 1 / 3, places=3)  # label 0,1,2 に対し常に "0"
        self.assertAlmostEqual(res["metrics"]["jsquad_em"], 1.0)
        self.assertAlmostEqual(res["metrics"]["mgsm_acc"], 1 / 3, places=3)  # 18 のみ正解
        self.assertIsNotNone(res["metrics"]["tok_s"])
        for p in res["generations"].values():
            self.assertTrue(os.path.exists(p))

    def test_messages_are_shared_format(self):
        m = common.build_messages("jcommonsenseqa", {"question": "Q", "choices": list("abcde")})
        self.assertEqual(m[0]["role"], "user")
        self.assertIn("4: e", m[0]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
