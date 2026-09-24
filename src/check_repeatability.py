"""
Repeatability check for the Step 6 balanced run.

The review sample is fixed by its seed, but the shared LLM endpoint is not
bit-deterministic (even at temperature 0 with a pinned seed, and even with one
request at a time), so a fresh run can flip a few borderline reviews. This
script re-scores the SAME sample K more times and reports how much the headline
numbers move, compared with the saved official run in output/step6/.

Usage:
    python src/check_repeatability.py [--runs 5] [--saved output/step6] [--workers 8]
Writes <saved>/repeatability.json.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from score_batch import load_all_rows, score_one, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--saved", default="output/step6")
    parser.add_argument("--data", default="data/Gift_Cards.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(f"{args.saved}/records.json") as f:
        saved = json.load(f)
    with open(f"{args.saved}/summary.json") as f:
        saved_summary = json.load(f)
    mode = saved_summary["meta"]["mode"]
    rows = load_all_rows(args.data)

    runs = []
    for k in range(1, args.runs + 1):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(lambda r: score_one(r["index"], r["file_index"], rows[r["file_index"]], mode, saved_summary["meta"]["mask_star_titles"]), saved))
        s = summarize(records, mode, saved_summary["meta"])
        flips = sum(1 for a, b in zip(saved, records) if a["predicted_label"] != b["predicted_label"])
        runs.append({
            "run": k,
            "overall_accuracy": s["overall_accuracy"],
            "per_class_recall": {c: v["recall"] for c, v in s["per_class"].items()},
            "sentiment_labels_differing_from_saved_run": flips,
        })
        print(f"run {k}: accuracy {s['overall_accuracy']:.1%}, "
              f"recall { {c: round(v['recall'], 2) for c, v in s['per_class'].items()} }, "
              f"{flips} sentiment labels differ from the saved run")

    accs = [r["overall_accuracy"] for r in runs] + [saved_summary["overall_accuracy"]]
    result = {
        "saved_run_accuracy": saved_summary["overall_accuracy"],
        "accuracy_min": min(accs), "accuracy_max": max(accs),
        "n_reviews": len(saved), "reruns": runs,
    }
    with open(f"{args.saved}/repeatability.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nAccuracy across the saved run + {args.runs} reruns: {min(accs):.1%} to {max(accs):.1%}")
    print(f"Wrote {args.saved}/repeatability.json")


if __name__ == "__main__":
    main()
