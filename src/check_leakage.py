"""
Leakage ablation: how much did star-count titles ("Three Stars") help the model?

Amazon auto-fills a review's title with its star count when the reviewer skips
the title, which hands the model the rating in words. score_batch.py hides such
titles by default. This script re-scores the SAME saved sample K times with the
titles hidden and K times with them left visible, and compares accuracy overall
and on just the reviews whose title is a star-count phrase.

Usage:
    python src/check_leakage.py [--runs 3] [--saved output/step6]
Writes <saved>/leakage_check.json.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from score_batch import load_all_rows, score_one


def run_once(saved, rows, mode, mask, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda r: score_one(r["index"], r["file_index"], rows[r["file_index"]], mode, mask), saved))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--saved", default="output/step6")
    parser.add_argument("--data", default="data/Gift_Cards.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    saved = json.load(open(f"{args.saved}/records.json"))
    mode = json.load(open(f"{args.saved}/summary.json"))["meta"]["mode"]
    rows = load_all_rows(args.data)
    star_positions = {r["index"] for r in saved if r["title_masked"]}

    result = {"n_reviews": len(saved), "n_star_titled": len(star_positions), "runs_per_condition": args.runs}
    for label, mask in [("titles_hidden", True), ("titles_visible", False)]:
        overall, subset = [], []
        for _ in range(args.runs):
            recs = run_once(saved, rows, mode, mask, args.workers)
            overall.append(sum(r["correct"] for r in recs) / len(recs))
            sub = [r for r in recs if r["index"] in star_positions]
            subset.append(sum(r["correct"] for r in sub) / len(sub))
        result[label] = {
            "overall_accuracy_runs": overall, "overall_accuracy_mean": sum(overall) / len(overall),
            "star_titled_subset_accuracy_runs": subset, "star_titled_subset_accuracy_mean": sum(subset) / len(subset),
        }
        print(f"{label:>15}: overall mean {result[label]['overall_accuracy_mean']:.1%} {['%.1f%%' % (100*x) for x in overall]}; "
              f"on the {len(star_positions)} star-titled reviews mean {result[label]['star_titled_subset_accuracy_mean']:.1%}")

    with open(f"{args.saved}/leakage_check.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.saved}/leakage_check.json")


if __name__ == "__main__":
    main()
