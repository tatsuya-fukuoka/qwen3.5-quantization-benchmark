#!/usr/bin/env python3
"""make_gguf.sh の冪等性・再開安全性テスト（スタブのバイナリで GPU/モデル不要）。
    python scripts/test_make_gguf.py
"""
import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "make_gguf.sh")

# 各スタブは呼ばれた記録を $CALLS に追記し、出力ファイルを作る
STUB_QUANT = '''#!/usr/bin/env bash
echo "quantize $*" >> "$CALLS"
# llama-quantize [--imatrix X] IN OUT TYPE : 出力は末尾から 2 番目
out=""; n=$#; i=0
for a in "$@"; do i=$((i+1)); [ "$i" = "$((n-1))" ] && out="$a"; done
echo x > "$out"
'''
STUB_IMAT = '''#!/usr/bin/env bash
echo "imatrix $*" >> "$CALLS"
while [ $# -gt 0 ]; do [ "$1" = "-o" ] && echo x > "$2"; shift; done
'''
STUB_PY = '''#!/usr/bin/env bash
echo "convert $*" >> "$CALLS"
while [ $# -gt 0 ]; do [ "$1" = "--outfile" ] && echo x > "$2"; shift; done
'''


class TestMakeGguf(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.d = self.tmp.name
        os.makedirs(f"{d}/llama/build/bin")
        os.makedirs(f"{d}/pybin")
        for path, body in ((f"{d}/llama/build/bin/llama-quantize", STUB_QUANT),
                           (f"{d}/llama/build/bin/llama-imatrix", STUB_IMAT),
                           (f"{d}/pybin/python", STUB_PY)):
            with open(path, "w") as f:
                f.write(body)
            os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        open(f"{d}/llama/convert_hf_to_gguf.py", "w").close()
        open(f"{d}/imat.txt", "w").write("t")
        self.calls = f"{d}/calls.log"
        self.env = {**os.environ, "CALLS": self.calls, "LLAMA_DIR": f"{d}/llama", "MODEL_DIR": f"{d}/model",
                    "OUT_DIR": f"{d}/out", "IMATRIX_TEXT": f"{d}/imat.txt",
                    "PATH": f"{d}/pybin:" + os.environ["PATH"]}

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *conds):
        return subprocess.run(["bash", SCRIPT, *conds], env=self.env, capture_output=True, text=True)

    def calls_text(self):
        return open(self.calls).read() if os.path.exists(self.calls) else ""

    def test_all_phase1_conditions_build_and_are_idempotent(self):
        conds = ["A0", "A1", "A2", "A3", "A3i", "A4", "A4i", "A5", "A5i"]
        r = self.run_script(*conds)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = os.listdir(f"{self.d}/out")
        for name in ["qwen3.5-9b-F16.gguf", "qwen3.5-9b-Q8_0.gguf", "qwen3.5-9b-Q5_K_M.gguf",
                     "qwen3.5-9b-Q4_K_M.gguf", "qwen3.5-9b-Q4_K_M-imatrix.gguf", "qwen3.5-9b-Q3_K_M.gguf",
                     "qwen3.5-9b-Q3_K_M-imatrix.gguf", "qwen3.5-9b-Q2_K.gguf", "qwen3.5-9b-Q2_K-imatrix.gguf",
                     "imatrix_c4_en_ja_32chunks.dat"]:
            self.assertIn(name, out)
        self.assertFalse([f for f in out if "partial" in f])
        lines = self.calls_text().splitlines()
        self.assertEqual(len([ln for ln in lines if ln.startswith("convert ")]), 1)  # F16 変換は 1 回
        self.assertEqual(len([ln for ln in lines if ln.startswith("imatrix ")]), 1)  # imatrix 生成は 1 回（3 条件で共有）
        self.assertEqual(len([ln for ln in lines if ln.startswith("quantize ") and "--imatrix" in ln]), 3)
        before = self.calls_text()
        self.assertEqual(self.run_script(*conds).returncode, 0)
        self.assertEqual(self.calls_text(), before)                # 2 回目は何も再実行しない

    def test_partial_file_is_not_mistaken_for_done(self):
        self.assertEqual(self.run_script("A0").returncode, 0)
        open(f"{self.d}/out/qwen3.5-9b-Q4_K_M.partial.gguf", "w").write("truncated")
        r = self.run_script("A3")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("qwen3.5-9b-Q4_K_M.gguf", os.listdir(f"{self.d}/out"))
        self.assertNotIn("qwen3.5-9b-Q4_K_M.partial.gguf", os.listdir(f"{self.d}/out"))

    def test_unknown_condition(self):
        self.assertNotEqual(self.run_script("A9").returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
