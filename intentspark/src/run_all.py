"""
run_all.py — download → classical → Qwen probe → plots
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    py = sys.executable
    run([py, "src/download_data.py"])
    run([py, "src/classical.py"])
    run([py, "src/embed_probe.py"])
    run([py, "src/plot_results.py"])
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
