#!/usr/bin/env python3
"""ハイブリッド・キャリブレーションのテスト（torch / transformers 不要）。
    python scripts/test_calibration_hybrid.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_calibration as bc  # noqa: E402
import calib_pack as cp  # noqa: E402
import quantize_w4a16 as q  # noqa: E402


def docs(n, length, start=0):
    return [[(start + i * 7919 + j) % 50000 + i * 100003 for j in range(length)] for i in range(n)]


class TestPurePacking(unittest.TestCase):
    def test_chunk_drops_remainder(self):  # (c)
        ch = cp.chunk_fixed(list(range(2500)), 1024)
        self.assertEqual(len(ch), 2)
        self.assertEqual(ch[1][-1], 2047)

    def test_all_samples_exact_length(self):  # (b)
        r = cp.build_hybrid(docs(3, 700), docs(40, 900, 5), seq_len=64, n=50, seed=0)
        self.assertEqual(len(r["samples"]), 50)
        self.assertTrue(all(len(s) == 64 for s in r["samples"]))

    def test_reproducible_and_seed_sensitive(self):  # (a)
        a = cp.build_hybrid(docs(3, 700), docs(40, 900, 5), 64, 50, 0)
        b = cp.build_hybrid(docs(3, 700), docs(40, 900, 5), 64, 50, 0)
        c = cp.build_hybrid(docs(3, 700), docs(40, 900, 5), 64, 50, 1)
        self.assertEqual(cp.samples_digest_ids(a["samples"]), cp.samples_digest_ids(b["samples"]))
        self.assertNotEqual(a["chunk_indices"], c["chunk_indices"])

    def test_pool_shortage_raises(self):  # (d)
        with self.assertRaises(ValueError):
            cp.build_hybrid(docs(1, 100), docs(1, 100, 3), seq_len=64, n=20, seed=0)

    def test_no_duplicate_chunks(self):  # (e)
        r = cp.build_hybrid(docs(3, 700), docs(40, 900, 5), 64, 50, 0)
        self.assertEqual(len({tuple(s) for s in r["samples"]}), 50)
        self.assertEqual(len(set(r["chunk_indices"])), 50)
        # 内容が同一のチャンクは 1 つに畳まれる
        _, ch = cp.select_chunks([[1, 2], [1, 2], [3, 4], [5, 6]], n=3, seed=0)
        self.assertEqual(sorted(map(tuple, ch)), [(1, 2), (3, 4), (5, 6)])
        with self.assertRaises(ValueError):
            cp.select_chunks([[1, 2], [1, 2], [3, 4]], n=3, seed=0)

    def test_existing_fully_used_and_breakdown(self):
        e = docs(3, 700)  # 2100 tokens, seq 64 -> 33 chunks 必須
        r = cp.build_hybrid(e, docs(40, 900, 5), 64, 40, 0)
        self.assertEqual(r["existing_tokens_used"], 2100)
        self.assertEqual(r["supplement_tokens_used"], 40 * 64 - 2100)
        flat = cp.concat_ids(r["samples"])
        self.assertEqual(flat[:2100], cp.concat_ids(e))  # 既存は先頭にそのまま

    def test_existing_exceeding_n_raises(self):
        with self.assertRaises(ValueError):
            cp.build_hybrid(docs(1, 5000), docs(5, 900), seq_len=64, n=10, seed=0)

    def test_real_scale_arithmetic(self):
        self.assertEqual(256 * 1024 - 93306, 168838)

    def test_collect_supplement_filters_and_stops(self):
        stream = [{"id": "s", "text": "短い"}, {"id": "a", "text": "あ" * 300}, {"id": "a2", "text": "あ" * 300},
                  {"id": "bad", "text": "評価データの一節" * 40}] + [{"id": f"d{i}", "text": chr(0x3042 + i) * 400} for i in range(50)]
        sh = cp.shingle_set(["評価データの一節" * 40])
        r = cp.collect_supplement(iter(stream), lambda s: [ord(c) for c in s], existing_tokens=0, seq_len=100,
                                  target_chunks=10, shingles=sh)
        self.assertEqual(r["skipped"], {"short": 1, "duplicate": 1, "contaminated": 1})
        self.assertGreaterEqual(r["tokens"] // 100, 10)
        self.assertLess(len(r["used"]), 20)  # 目標到達で打ち切り
        self.assertNotIn("bad", [u["id"] for u in r["used"]])


class TestBuildScript(unittest.TestCase):
    def _setup(self, d, n_sup=60):
        with open(f"{d}/corpus.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(f"既存文書{i} " + "い" * 150 for i in range(10)) + "\n\n")
        with open(f"{d}/sup.jsonl", "w", encoding="utf-8") as f:
            for i in range(n_sup):
                f.write(json.dumps({"id": f"x{i}", "text": f"補充{i}" + chr(0x4E00 + i) * 500}, ensure_ascii=False) + "\n")
        with open(f"{d}/eval.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"question": "評価用の質問" * 10}, ensure_ascii=False) + "\n")

    def _run(self, d, *extra):
        return subprocess.run([sys.executable, "-I", os.path.join(HERE, "build_calibration.py"), "--corpus", f"{d}/corpus.txt",
                               "--out", f"{d}/out/p.json", "--dummy-tokenizer", "--supplement-jsonl", f"{d}/sup.jsonl",
                               "--seq-len", "128", "--num-samples", "16", "--pool-margin", "4", *extra],
                              capture_output=True, text=True)

    def test_end_to_end_with_dummy_tokenizer(self):  # (f)
        with tempfile.TemporaryDirectory() as d:
            self._setup(d)
            r = self._run(d, "--decontam", f"{d}/eval.jsonl")
            self.assertEqual(r.returncode, 0, r.stderr)
            meta = json.load(open(f"{d}/out/p.meta.json", encoding="utf-8"))
            packed = json.load(open(f"{d}/out/p.json"))
            self.assertEqual((meta["num_samples"], meta["seq_len"], meta["total_tokens"]), (16, 128, 2048))
            self.assertEqual(meta["existing"]["tokens_used"], meta["existing"]["tokens_total"])
            self.assertEqual(meta["existing"]["tokens_used"] + meta["supplement"]["tokens_used"], 2048)
            self.assertTrue(all(len(s) == 128 for s in packed["samples"]))
            self.assertEqual(meta["samples_sha256"], cp.samples_digest_ids(packed["samples"]))
            self.assertEqual(meta["packed_sha256"], bc.sha256_file(f"{d}/out/p.json"))
            self.assertEqual(len(meta["supplement"]["doc_ids"]), meta["supplement"]["docs_consumed"])
            # 検証モード OK → 改変で NG
            self.assertEqual(self._run(d, "--verify-only").returncode, 0)
            packed["samples"][0][0] += 1
            json.dump(packed, open(f"{d}/out/p.json", "w"))
            self.assertEqual(self._run(d, "--verify-only").returncode, 1)

    def test_deterministic_across_runs(self):
        with tempfile.TemporaryDirectory() as d:
            self._setup(d)
            self._run(d)
            a = json.load(open(f"{d}/out/p.meta.json"))["samples_sha256"]
            self._run(d)
            self.assertEqual(a, json.load(open(f"{d}/out/p.meta.json"))["samples_sha256"])

    def test_insufficient_supplement_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self._setup(d, n_sup=1)
            self.assertNotEqual(self._run(d).returncode, 0)

    def test_quantize_loads_packed_and_records_meta(self):
        with tempfile.TemporaryDirectory() as d:
            self._setup(d)
            self.assertEqual(self._run(d).returncode, 0)
            ids, L, fsha, dg = q.load_packed_calibration(f"{d}/out/p.json")
            meta = json.load(open(f"{d}/out/p.meta.json", encoding="utf-8"))
            cal = q.packed_calibration_meta(f"{d}/out/p.json", ids, L, fsha, dg, meta)
            self.assertEqual((cal["method"], cal["num_samples"], cal["seq_len"], cal["total_tokens"]), ("hybrid-pack", 16, 128, 2048))
            self.assertEqual(cal["existing_tokens"] + cal["supplement_tokens"], 2048)
            self.assertEqual((cal["packed_sha256"], cal["samples_sha256"]), (meta["packed_sha256"], meta["samples_sha256"]))
            with open(f"{d}/bad.json", "w") as f:
                json.dump({"format": cp.FORMAT, "seq_len": 128, "num_samples": 1, "samples": [[1, 2]]}, f)
            with self.assertRaises(SystemExit):
                q.load_packed_calibration(f"{d}/bad.json")

    def test_quantize_cli_has_packed_option(self):
        a = q.build_parser().parse_args(["--method", "gptq", "--model", "m", "--out", "o", "--calib-packed", "p.json"])
        self.assertEqual(a.calib_packed, "p.json")


class TestFineweb2Args(unittest.TestCase):
    """load_dataset へ渡す引数の固定（datasets 未導入でも通る。config 未指定で Colab 実機が失敗した回帰）。"""

    def test_load_kwargs(self):
        kw = bc.fineweb2_load_kwargs("REV")
        self.assertEqual(kw, {"name": "jpn_Jpan", "data_files": "data/jpn_Jpan/train/000_00000.parquet",
                              "split": "train", "streaming": True, "revision": "REV"})

    def test_iter_fineweb2_calls_load_dataset(self):
        from unittest import mock
        rows = [{"id": "a", "text": "t1", "url": "u"}, {"id": "b", "text": "t2", "url": "u"}]
        ds = mock.MagicMock()
        ds.shuffle.return_value = rows
        fake_datasets = mock.MagicMock()
        fake_datasets.load_dataset.return_value = ds
        fake_hub = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"datasets": fake_datasets, "huggingface_hub": fake_hub}):
            stream, src = bc.iter_fineweb2(7, "REV")
            out = list(stream)
        fake_datasets.load_dataset.assert_called_once_with(
            "HuggingFaceFW/fineweb-2", name="jpn_Jpan", data_files="data/jpn_Jpan/train/000_00000.parquet",
            split="train", streaming=True, revision="REV")
        fake_hub.HfApi.assert_not_called()   # revision 指定時は API 解決しない
        ds.shuffle.assert_called_once_with(seed=7, buffer_size=bc.SHUFFLE_BUFFER)
        self.assertEqual(out, [{"id": "a", "text": "t1"}, {"id": "b", "text": "t2"}])
        self.assertEqual((src["config"], src["revision"], src["shuffle_seed"]), ("jpn_Jpan", "REV", 7))


class TestGitignore(unittest.TestCase):
    def test_packed_calibration_is_tracked_but_data_main_is_not(self):
        root = os.path.dirname(HERE)
        for path, ignored in (("calib/hybrid_256x1024.json", 1), ("calib/hybrid_256x1024.meta.json", 1),
                              ("data_main/imatrix_calib.txt", 0)):
            r = subprocess.run(["git", "check-ignore", "-q", path], cwd=root)
            self.assertEqual(r.returncode, 0 if ignored == 0 else 1, path)


class TestNotebookB(unittest.TestCase):
    def test_b2_b3_use_packed_and_cell_exists(self):
        root = os.path.dirname(HERE)
        cells = json.load(open(os.path.join(root, "colab", "phase2_main_b.ipynb"), encoding="utf-8"))["cells"]
        code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
        self.assertEqual(code.count('"--calib-packed", CALIB_PACKED'), 2)
        self.assertNotIn("--calib-text", code)
        self.assertIn("build_calibration.py", code)
        self.assertTrue(all("id" in c for c in cells))


if __name__ == "__main__":
    unittest.main(verbosity=2)
