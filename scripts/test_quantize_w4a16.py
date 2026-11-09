#!/usr/bin/env python3
"""GPU・llm-compressor なしで通る Phase 2 のテスト: 量子化スクリプトの引数・対象レイヤー・キャリブレーション抽出、
Track B 用の集計/タイミング出力先の分離。
    python scripts/test_quantize_w4a16.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import quantize_fp8  # noqa: E402
import quantize_w4a16 as q  # noqa: E402


def _text(n_lines: int) -> str:
    return "\n".join(f"サンプル文書 {i} " + "あ" * 80 for i in range(n_lines))


class TestCalibration(unittest.TestCase):
    def test_fixed_seed_is_deterministic(self):
        t = _text(2000)
        a = q.build_calibration_samples(t)
        b = q.build_calibration_samples(t)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 512)
        self.assertEqual(len(set(a)), 512)
        self.assertEqual(q.samples_digest(a), q.samples_digest(b))

    def test_seed_changes_selection(self):
        t = _text(2000)
        self.assertNotEqual(q.build_calibration_samples(t, seed=0), q.build_calibration_samples(t, seed=1))

    def test_short_and_duplicate_lines_are_dropped(self):
        t = "短い\n\n" + ("x" * 100 + "\n") * 5 + _text(10)
        out = q.build_calibration_samples(t, n=11)
        self.assertEqual(len(out), 11)
        self.assertNotIn("短い", out)
        self.assertEqual(out.count("x" * 100), 1)

    def test_too_few_lines_fails(self):
        with self.assertRaises(SystemExit):
            q.build_calibration_samples(_text(100), n=512)

    def test_long_line_is_capped(self):
        out = q.build_calibration_samples("y" * 100000 + "\n" + _text(3), n=4)
        self.assertTrue(all(len(s) <= q.MAX_CHARS for s in out))


class TestRecipeAndCli(unittest.TestCase):
    def test_ignore_matches_fp8(self):
        self.assertEqual(q.IGNORE, quantize_fp8.IGNORE)
        for k in ("lm_head", "re:.*visual.*", "re:.*vision.*"):
            self.assertIn(k, q.IGNORE)

    def test_gptq_spec(self):
        s = q.recipe_spec("gptq")
        self.assertEqual((s["modifier"], s["scheme"], s["targets"]), ("GPTQModifier", "W4A16", "Linear"))
        self.assertEqual(s["ignore"], q.IGNORE)

    def test_awq_spec(self):
        s = q.recipe_spec("awq")
        self.assertEqual((s["modifier"], s["scheme"], s["targets"]), ("AWQModifier", "W4A16", "Linear"))
        self.assertNotIn("mappings", s)
        self.assertEqual(q.recipe_spec("awq", [["a", "b"]])["mappings"], [["a", "b"]])
        with self.assertRaises(ValueError):
            q.recipe_spec("bogus")

    def test_conditions_and_parser(self):
        self.assertEqual(q.CONDITIONS, {"gptq": "B2", "awq": "B3"})
        a = q.build_parser().parse_args(["--method", "awq", "--model", "m", "--out", "o"])
        self.assertEqual((a.num_samples, a.seed, a.calib_text), (512, 0, "data_main/imatrix_calib.txt"))
        with self.assertRaises(SystemExit):
            q.build_parser().parse_args(["--method", "fp8", "--model", "m", "--out", "o"])

    def test_models_dir_is_gitignored(self):
        r = subprocess.run(["git", "check-ignore", "models/Qwen3.5-9B-GPTQ-W4A16"], cwd=ROOT,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)

    def test_script_records_calibration_in_meta(self):
        src = open(os.path.join(HERE, "quantize_w4a16.py"), encoding="utf-8").read()
        for key in ('"calibration"', '"num_samples"', '"seed"', '"samples_sha256"', '"source_text"', '"recipe"'):
            self.assertIn(key, src)


class TestTrackBOutputs(unittest.TestCase):
    def test_summary_b_does_not_touch_summary(self):
        with tempfile.TemporaryDirectory() as d:
            for c, v in (("A0", 0.5), ("B0", 0.9), ("B1", 0.8)):
                json.dump({"condition": c, "kind": "x", "metrics": {"jcsqa_acc": v}}, open(f"{d}/{c}.json", "w"))
            open(f"{d}/summary.tsv", "w").write("KEEP\n")
            subprocess.run([sys.executable, os.path.join(HERE, "make_summary.py"), d, "summary_b.tsv", "B"],
                           check=True, capture_output=True)
            self.assertEqual(open(f"{d}/summary.tsv").read(), "KEEP\n")
            rows = open(f"{d}/summary_b.tsv", encoding="utf-8").read().splitlines()
            self.assertEqual([r.split("\t")[0] for r in rows[1:]], ["B0", "B1"])

    def test_timings_file_env(self):
        with tempfile.TemporaryDirectory() as d:
            env = {**os.environ, "RESULTS_DIR": d, "TIMINGS_FILE": "timings_b.tsv"}
            subprocess.run([sys.executable, os.path.join(HERE, "run_timed.py"), "lbl", "--", "true"],
                           check=True, capture_output=True, env=env)
            self.assertTrue(os.path.exists(f"{d}/timings_b.tsv"))
            self.assertFalse(os.path.exists(f"{d}/timings.tsv"))


class TestNotebook(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "colab", "phase2_main_b.ipynb"), encoding="utf-8") as f:
            self.cells = json.load(f)["cells"]
        self.code = ["".join(c["source"]) for c in self.cells if c["cell_type"] == "code"]

    def test_code_cells_compile(self):
        for i, src in enumerate(self.code):
            body = "\n".join("pass" if ln.lstrip().startswith("!") else ln for ln in src.splitlines())
            compile(body, f"cell{i}", "exec")

    def test_order_push_and_isolation(self):
        allsrc = "\n".join(self.code)
        calls = [c for c in self.code if c.startswith("run_condition(")]
        self.assertEqual(calls, ['run_condition("B0")', 'run_condition("B1")', 'run_condition("B2")', 'run_condition("B3")'])
        self.assertIn("Add Phase 2 result ({cond})", allsrc)
        self.assertIn("PUSH FAILED", allsrc)
        self.assertIn("summary_b.tsv", allsrc)
        self.assertIn('TIMINGS = "timings_b.tsv"', allsrc)
        self.assertNotIn('"summary.tsv"', allsrc)

    def test_token_only_in_cells(self):
        self.assertEqual(sum("GITHUB_TOKEN" in c for c in self.code), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
