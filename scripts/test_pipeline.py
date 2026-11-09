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

    def test_mgsm_count_is_independent(self):
        rows = {"jcommonsenseqa": [1] * 3, "jsquad": [1] * 3, "mgsm": [1] * 5}
        prep_data.check_counts(rows, {"jcommonsenseqa": 3, "jsquad": 3, "mgsm": 5})  # 通る
        with self.assertRaises(SystemExit):
            prep_data.check_counts(rows, {"jcommonsenseqa": 3, "jsquad": 3, "mgsm": 6})
        with self.assertRaises(SystemExit):
            prep_data.check_counts({**rows, "jsquad": [1] * 2}, {"jcommonsenseqa": 3, "jsquad": 3, "mgsm": 5})

    def test_prep_mgsm_count_differs_from_others(self):
        d = self.d
        with open(f"{d}/m.tsv", "a", encoding="utf-8") as f:
            f.write("問題4\t4\n問題5\t5\n")
        self.assertEqual(len(prep_data.prep_mgsm(f"{d}/m.tsv", 5)), 5)
        self.assertEqual(len(prep_data.prep_jcommonsenseqa(f"{d}/jc.json", 3)), 3)

    def test_ppl_text_independent_of_eval_n(self):
        d = self.d
        # 評価 4 件でも、除外を 2 問分（= n=2 のとき）に固定すれば PPL 用 context は n=2 のときと同一
        rows2, ppl2 = prep_data.prep_jsquad(f"{d}/sq.json", 2)
        rows4, ppl4 = prep_data.prep_jsquad(f"{d}/sq.json", 4, ppl_exclude_n=2)
        self.assertEqual((len(rows2), len(rows4)), (2, 4))
        self.assertEqual(ppl2, ppl4)
        self.assertEqual(rows4[:2], rows2)  # 先頭 N 件の関係も保たれる
        # 既定（指定なし）は従来どおり n と連動
        self.assertEqual(prep_data.prep_jsquad(f"{d}/sq.json", 4)[1], ["文脈2", "文脈3", "文脈4", "文脈5"])

    def test_eval_n_zero_uses_all_rows(self):
        d = self.d
        common.write_jsonl(f"{d}/jcommonsenseqa.jsonl", prep_data.prep_jcommonsenseqa(f"{d}/jc.json", 5))
        common.write_jsonl(f"{d}/jsquad.jsonl", prep_data.prep_jsquad(f"{d}/sq.json", 4)[0])
        common.write_jsonl(f"{d}/mgsm.jsonl", prep_data.prep_mgsm(f"{d}/m.tsv", 3))

        def one(msgs, mt):
            return {"text": "0", "completion_tokens": 1}

        def run(n):
            return common.run_eval(one, lambda ms, mt: [one(m, mt) for m in ms], d, f"{d}/gen{n}", n,
                                   os.path.join(HERE, "prompts_qualitative.txt"))["counts"]

        c = run(0)
        self.assertEqual((c["jcommonsenseqa"], c["jsquad"], c["mgsm"]), (5, 4, 3))
        self.assertEqual(run(-1)["jcommonsenseqa"], 5)
        # Phase 0 の「先頭 N 件」セマンティクスは不変
        c2 = run(2)
        self.assertEqual((c2["jcommonsenseqa"], c2["jsquad"], c2["mgsm"]), (2, 2, 2))
        self.assertEqual(run(100)["mgsm"], 3)

    def test_eval_scripts_defaults_and_n_zero_record(self):
        for name in ("eval_gguf.py", "eval_vllm.py"):
            with open(os.path.join(HERE, name), encoding="utf-8") as f:
                src = f.read()
            self.assertIn('"--n", type=int, default=100', src)  # 既定は Phase 0 のまま
            self.assertIn('"n_per_task": args.n if args.n > 0 else "all"', src)

    def test_eval_vllm_max_num_seqs(self):
        import eval_vllm  # vllm は main() 内で import されるのでここでは不要
        base = ["--condition", "B0", "--model", "m"]
        a = eval_vllm.build_parser().parse_args(base)
        self.assertIsNone(a.max_num_seqs)
        self.assertNotIn("max_num_seqs", eval_vllm.build_llm_kwargs(a))  # 未指定は vLLM 既定のまま
        b = eval_vllm.build_parser().parse_args(base + ["--max-num-seqs", "128"])
        self.assertEqual(b.max_num_seqs, 128)
        kw = eval_vllm.build_llm_kwargs(b)
        self.assertEqual(kw["max_num_seqs"], 128)
        self.assertEqual((kw["max_model_len"], kw["seed"], kw["language_model_only"]),
                         (common.CTX, common.SEED, True))
        with open(os.path.join(HERE, "eval_vllm.py"), encoding="utf-8") as f:
            self.assertIn('"max_num_seqs": args.max_num_seqs', f.read())  # 結果 JSON の settings に記録

    def test_messages_are_shared_format(self):
        m = common.build_messages("jcommonsenseqa", {"question": "Q", "choices": list("abcde")})
        self.assertEqual(m[0]["role"], "user")
        self.assertIn("4: e", m[0]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
