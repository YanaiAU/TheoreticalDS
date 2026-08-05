"""
run_overnight.py — unattended GPU pipeline (Qwen, not Gemma).

1) download SCOTUS data
2) classical TF-IDF baselines (expect ~chance balanced accuracy)
3) Qwen3-0.6B vanilla
4) Qwen3-0.6B speculative (prompt-lookup)
5) Qwen3-4B assisted by Qwen3-0.6B (optional draft/target speculative)
6) plots

Usage:
  python src/run_overnight.py --n 300
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-classical", action="store_true")
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--with-assisted", action="store_true", help="Also run 4B+0.6B assisted")
    args = p.parse_args()
    py = sys.executable

    if not args.skip_download:
        run([py, "src/download_data.py"])
    if not args.skip_classical:
        run([py, "src/classical.py"])
    if not args.skip_llm:
        run([py, "src/llm_judge.py", "--mode", "vanilla", "--n", str(args.n)])
        run([py, "src/llm_judge.py", "--mode", "speculative", "--n", str(args.n)])
        if args.with_assisted:
            run([py, "src/llm_judge.py", "--mode", "assisted", "--n", str(args.n)])
    run([py, "src/plot_results.py"])
    print("\nDONE. See results/", flush=True)


if __name__ == "__main__":
    main()
