"""Copy the small, human-readable outputs of every finished run (args.json, log.jsonl, final.json)
from runs/ into results/ so they can be versioned and plotted in the notebook.

    python src/collect_results.py [--runs runs] [--results results]
"""
import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    n = 0
    for name in sorted(os.listdir(args.runs)):
        src = os.path.join(args.runs, name)
        if not os.path.exists(os.path.join(src, "final.json")):
            continue  # unfinished run
        dst = os.path.join(args.results, name)
        os.makedirs(dst, exist_ok=True)
        for f in ("args.json", "log.jsonl", "final.json"):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        n += 1
    print(f"collected {n} finished runs into {args.results}/")


if __name__ == "__main__":
    main()
