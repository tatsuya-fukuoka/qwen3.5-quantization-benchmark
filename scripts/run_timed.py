#!/usr/bin/env python3
"""コマンドを実行して所要時間を results/timings.tsv に追記する。
使い方: python scripts/run_timed.py <label> -- <command...>
環境変数 RESULTS_DIR（既定 results）と TIMINGS_FILE（既定 timings.tsv。Track B は timings_b.tsv）で出力先を変える。
"""
import os
import subprocess
import sys
import time

if "--" not in sys.argv or len(sys.argv) < 4:
    sys.exit(__doc__)
i = sys.argv.index("--")
label, cmd = " ".join(sys.argv[1:i]), sys.argv[i + 1:]
path = os.path.join(os.environ.get("RESULTS_DIR", "results"),
                    os.environ.get("TIMINGS_FILE", "timings.tsv"))
os.makedirs(os.path.dirname(path), exist_ok=True)
t0 = time.time()
rc = subprocess.call(cmd)
dt = time.time() - t0
new = not os.path.exists(path)
with open(path, "a") as f:
    if new:
        f.write("label\tseconds\tminutes\texit_code\tstarted\n")
    started = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))
    f.write(f"{label}\t{dt:.0f}\t{dt / 60:.1f}\t{rc}\t{started}\n")
print(f"[timed] {label}: {dt / 60:.1f} min (exit {rc})")
sys.exit(rc)
