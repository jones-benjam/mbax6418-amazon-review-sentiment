"""
Step 2: Score a batch of reviews against the star rating.

Ground truth (binary, this step only): rating >= 4 -> POSITIVE, else NEGATIVE.
The model only ever receives title/text (via classify_review); the rating is
used here, after the fact, purely to check the model's answer.

Usage:
    python src/score_batch.py [--n 100] [--data data/Gift_Cards.jsonl] \
        [--out output/step2] [--workers 8]
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from classify import classify_review


def true_label_binary(rating: float) -> str:
    return "POSITIVE" if rating >= 4 else "NEGATIVE"


def load_rows(path: str, n: int) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    return rows


def score_one(index: int, row: dict) -> dict:
    truth = true_label_binary(row["rating"])
    result = classify_review(row.get("title", ""), row.get("text", ""))
    predicted = result["sentiment"]
    return {
        "index": index,
        "asin": row.get("asin"),
        "rating": row["rating"],
        "title": row.get("title", ""),
        "text": row.get("text", ""),
        "true_label": truth,
        "predicted_label": predicted,
        "correct": predicted == truth,
        "raw_model_output": result["raw"],
    }


def summarize(records: list[dict]) -> dict:
    total = len(records)
    parsed = [r for r in records if r["predicted_label"] is not None]
    unparsed = total - len(parsed)

    overall_correct = sum(r["correct"] for r in parsed)
    overall_accuracy = overall_correct / len(parsed) if parsed else 0.0

    classes = ["POSITIVE", "NEGATIVE"]
    per_class = {}
    confusion = {t: {p: 0 for p in classes} for t in classes}

    for r in parsed:
        t, p = r["true_label"], r["predicted_label"]
        if p in confusion[t]:
            confusion[t][p] += 1

    for c in classes:
        class_rows = [r for r in parsed if r["true_label"] == c]
        n_class = len(class_rows)
        n_correct = sum(r["correct"] for r in class_rows)
        per_class[c] = {
            "support": n_class,
            "correct": n_correct,
            "accuracy": (n_correct / n_class) if n_class else None,
        }

    misclassified = [r["index"] for r in parsed if not r["correct"]]

    return {
        "total_reviews": total,
        "unparsed_responses": unparsed,
        "overall_accuracy": overall_accuracy,
        "class_distribution_in_batch": {
            c: sum(1 for r in parsed if r["true_label"] == c) for c in classes
        },
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion,
        "misclassified_indices": misclassified,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--data", default="data/Gift_Cards.jsonl")
    parser.add_argument("--out", default="output/step2")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = load_rows(args.data, args.n)
    print(f"Loaded {len(rows)} rows from {args.data} (file order, first {args.n})")

    records = [None] * len(rows)
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_one, i, row): i for i, row in enumerate(rows)}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            records[i] = future.result()
            done += 1
            if done % 10 == 0 or done == len(rows):
                print(f"  scored {done}/{len(rows)}")
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")

    summary = summarize(records)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "records.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Step 2 Summary (100-row, file-order batch, binary POS/NEG) ===")
    print(f"Class distribution in this batch (from rating): {summary['class_distribution_in_batch']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.1%}  ({summary['total_reviews'] - summary['unparsed_responses']} scored, {summary['unparsed_responses']} unparsed)")
    for c, stats in summary["per_class_accuracy"].items():
        acc = f"{stats['accuracy']:.1%}" if stats["accuracy"] is not None else "n/a"
        print(f"  {c:<9} support={stats['support']:<4} accuracy={acc}")
    print("Confusion matrix (rows=true rating-derived label, cols=model prediction):")
    for t, row in summary["confusion_matrix"].items():
        print(f"  {t:<9} {row}")
    print(f"\nSaved per-review results to {out_dir}/records.json")
    print(f"Saved summary to {out_dir}/summary.json")


if __name__ == "__main__":
    main()
