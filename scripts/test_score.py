#!/usr/bin/env python3
"""score.py の単体テスト（ダミーデータ）。  python scripts/test_score.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score  # noqa: E402


class TestJCommonsenseQA(unittest.TestCase):
    def test_extract(self):
        self.assertEqual(score.extract_choice("3"), 3)
        self.assertEqual(score.extract_choice("  ２: りんご"), 2)  # 全角数字
        self.assertEqual(score.extract_choice("答えは 4 です"), 4)
        self.assertIsNone(score.extract_choice("わかりません"))
        self.assertIsNone(score.extract_choice("9"))  # 範囲外

    def test_accuracy(self):
        items = [{"label": 0}, {"label": 1}, {"label": 4}, {"label": 2}]
        r = score.score_jcommonsenseqa(["0", "1: foo", "3", "不明"], items)
        self.assertEqual(r["n"], 4)
        self.assertAlmostEqual(r["acc"], 0.5)


class TestJSQuAD(unittest.TestCase):
    def test_exact_and_f1(self):
        items = [{"answers": ["東京都"]}, {"answers": ["富士山"]}, {"answers": ["大阪"]}]
        r = score.score_jsquad(["東京都", "答え: 富士山。", "京都"], items)
        # 1問目 EM=1/F1=1、2問目 接頭辞・句点除去で EM=1、3問目は不一致 (F1=0)
        self.assertAlmostEqual(r["em"], 2 / 3)
        self.assertAlmostEqual(r["f1"], 2 / 3)

    def test_partial_f1_and_multi_gold(self):
        # pred "東京タワー" vs gold "東京" : P=2/5, R=1 → F1=4/7
        f1, em = score._f1_em("東京タワー", "東京")
        self.assertAlmostEqual(f1, 2 * (2 / 5) * 1 / (2 / 5 + 1))
        self.assertEqual(em, 0.0)
        r = score.score_jsquad(["大阪府"], [{"answers": ["東京", "大阪府"]}])
        self.assertAlmostEqual(r["em"], 1.0)  # 複数正解は最大値

    def test_normalize_and_empty(self):
        self.assertEqual(score.normalize_ja(" Ａ、ｂ。 c "), "abc")
        r = score.score_jsquad([""], [{"answers": ["東京"]}])
        self.assertEqual((r["f1"], r["em"]), (0.0, 0.0))

    def test_multiline_uses_first_line(self):
        r = score.score_jsquad(["東京\n補足説明です"], [{"answers": ["東京"]}])
        self.assertAlmostEqual(r["em"], 1.0)


class TestMGSM(unittest.TestCase):
    def test_extract(self):
        self.assertEqual(score.extract_number("計算すると 3+4=7。\n答え: 18"), 18.0)
        self.assertEqual(score.extract_number("答え：１，２３４"), 1234.0)  # 全角・カンマ
        self.assertEqual(score.extract_number("答え: 12.5ドル"), 12.5)
        self.assertEqual(score.extract_number("答え: -3"), -3.0)
        self.assertEqual(score.extract_number("途中 5 と 9 で結果は 14"), 14.0)  # 「答え」無し→最後の数
        self.assertEqual(score.extract_number("答え: 1\n答え: 2"), 2.0)  # 最後の「答え」
        self.assertIsNone(score.extract_number("数値なし"))

    def test_accuracy(self):
        items = [{"answer": "18"}, {"answer": "1,234"}, {"answer": "7"}, {"answer": "3"}]
        preds = ["答え: 18", "答え: 1234", "答え: 8", "わからない"]
        r = score.score_mgsm(preds, items)
        self.assertEqual(r["n"], 4)
        self.assertAlmostEqual(r["acc"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
