"""
Scoring script (Steps 2 and 6): run reviews through the LLM classifier and
check its answers against the star rating.

The rating is the "correct answer" -- but the model never sees it. Ground
truth is worked out here, after the fact:

  --mode binary : rating >= 4 -> POSITIVE, else NEGATIVE          (Step 2)
  --mode three  : 4-5 POSITIVE, 3 NEUTRAL, 1-2 NEGATIVE           (Step 6)

Which reviews are scored:
  --sample first     : the first N rows of the file, in file order (Step 2)
  --sample balanced  : PER_CLASS reviews from each of the three rating classes,
                       drawn from the WHOLE file with a fixed random seed (Step 6)

Reproduce the two runs reported in the README:
  python src/score_batch.py --mode binary --sample first --n 100 --out output/step2
  python src/score_batch.py --mode three --sample balanced --per-class 50 --seed 42 --out output/step6
"""
import argparse
import collections
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from classify import CLASSES, MODEL, SETTINGS, classify_review, is_star_title, mask_star_title, star_title_number
from emotion_wordlist import primary_emotion as wordlist_primary_emotion

UNPARSED = "UNPARSED"


def true_label(rating: float, mode: str) -> str:
    if rating >= 4:
        return "POSITIVE"
    if mode == "three" and rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def load_all_rows(path: str) -> list:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def dataset_stats(rows: list) -> dict:
    """Whole-file counts, so every report can say how skewed the data is."""
    ratings = collections.Counter(int(r["rating"]) for r in rows)
    classes = collections.Counter(true_label(r["rating"], "three") for r in rows)
    star_titled = [(star_title_number(r.get("title")), int(r["rating"])) for r in rows if is_star_title(r.get("title"))]
    return {
        "total_rows": len(rows),
        "star_count_titles": {
            "total": len(star_titled),
            "title_number_matches_rating": sum(1 for stated, actual in star_titled if stated == actual),
        },
        "rating_counts": {str(k): ratings.get(k, 0) for k in range(1, 6)},
        "class_counts": {c: classes.get(c, 0) for c in CLASSES["three"]},
    }


def pick_first(rows: list, n: int) -> list:
    return list(range(min(n, len(rows))))


def pick_balanced(rows: list, per_class: int, seed: int) -> list:
    """File indices of `per_class` reviews from each class, sampled without
    replacement from the whole file with a fixed seed, returned in file order."""
    buckets = {c: [] for c in CLASSES["three"]}
    for i, row in enumerate(rows):
        buckets[true_label(row["rating"], "three")].append(i)
    too_small = {c: len(v) for c, v in buckets.items() if len(v) < per_class}
    if too_small:
        raise SystemExit(f"--per-class {per_class} is more than the file has for {too_small}")
    rng = random.Random(seed)
    picked = []
    for c in CLASSES["three"]:  # fixed class order keeps the draw deterministic
        picked.extend(rng.sample(buckets[c], per_class))
    return sorted(picked)


def score_one(position: int, file_index: int, row: dict, mode: str, mask_titles: bool = True) -> dict:
    title, text = row.get("title", ""), row.get("text", "")
    masked = mask_titles and is_star_title(title)
    # What the model (and the word list) actually get to read: a title that only
    # states the star count is the rating in disguise, so it is hidden.
    shown_title = mask_star_title(title) if mask_titles else title
    truth = true_label(row["rating"], mode)
    result = classify_review(shown_title, text, mode)
    predicted = result["sentiment"]
    wordlist = wordlist_primary_emotion(shown_title, text)
    return {
        "index": position,
        "file_index": file_index,
        "asin": row.get("asin"),
        "rating": row["rating"],
        "title": title,
        "title_masked": masked,
        "text": text,
        "true_label": truth,
        "predicted_label": predicted,
        "correct": predicted == truth,
        "llm_emotion": result["emotion"],
        "wordlist_emotion": wordlist["emotion"],
        "wordlist_emotion_scores": wordlist["scores"],
        "emotions_agree": (
            result["emotion"] is not None
            and wordlist["emotion"] is not None
            and result["emotion"] == wordlist["emotion"]
        ),
        "raw_model_output": result["raw"],
        "finish_reason": result["finish_reason"],
        "error": result["error"],
    }


def wilson_interval(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(records: list, mode: str, meta: dict) -> dict:
    meta = {**meta, "titles_hidden_from_model": sum(1 for r in records if r["title_masked"])}
    classes = CLASSES[mode]
    total = len(records)
    unparsed = sum(1 for r in records if r["predicted_label"] is None)
    truncated = sum(1 for r in records if r["finish_reason"] == "length")

    # Unparsed answers count as wrong -- never silently dropped from the denominator.
    correct_total = sum(r["correct"] for r in records)
    confusion = {t: {p: 0 for p in classes + [UNPARSED]} for t in classes}
    for r in records:
        confusion[r["true_label"]][r["predicted_label"] or UNPARSED] += 1

    per_class = {}
    for c in classes:
        support = sum(confusion[c].values())
        correct = confusion[c][c]
        predicted = sum(confusion[t][c] for t in classes)
        lo, hi = wilson_interval(correct, support)
        per_class[c] = {
            "support": support,
            "correct": correct,
            "recall": correct / support if support else None,
            "ci_low": lo,
            "ci_high": hi,
            "predicted": predicted,
            "precision": correct / predicted if predicted else None,
        }

    # Per-class recalls weighted by how common each class is in the WHOLE file: what
    # overall accuracy would look like on the real review mix rather than a balanced one.
    cc = meta["dataset"]["class_counts"]
    file_counts = ({"POSITIVE": cc["POSITIVE"], "NEGATIVE": cc["NEUTRAL"] + cc["NEGATIVE"]}
                   if mode == "binary" else {c: cc[c] for c in classes})
    file_total = sum(file_counts.values())
    weights = {c: file_counts[c] / file_total for c in classes}
    file_mix_accuracy = sum(per_class[c]["recall"] * weights[c] for c in classes if per_class[c]["recall"] is not None)

    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    supports = {c: per_class[c]["support"] for c in classes}
    majority = max(supports, key=supports.get)

    both = [r for r in records if r["llm_emotion"] and r["wordlist_emotion"]]
    agree = sum(1 for r in both if r["emotions_agree"])
    emotion_agreement = {
        "reviews_with_both_emotions": len(both),
        "reviews_wordlist_had_no_lexicon_words": sum(1 for r in records if r["wordlist_emotion"] is None),
        "reviews_llm_had_no_valid_emotion": sum(1 for r in records if r["llm_emotion"] is None),
        "agree_count": agree,
        "agreement_rate": agree / len(both) if both else None,
        "llm_emotion_distribution": dict(sorted(collections.Counter(
            r["llm_emotion"] for r in records if r["llm_emotion"]).items())),
        "wordlist_emotion_distribution": dict(sorted(collections.Counter(
            r["wordlist_emotion"] for r in records if r["wordlist_emotion"]).items())),
    }

    return {
        "meta": meta,
        "classes": classes,
        "total_reviews": total,
        "unparsed_responses": unparsed,
        "truncated_responses": truncated,
        "overall_accuracy": correct_total / total if total else 0.0,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "majority_baseline": {"class": majority, "accuracy": supports[majority] / total if total else 0.0},
        "file_mix_weighted_accuracy": {"value": file_mix_accuracy, "weights": weights},
        "class_distribution": supports,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "misclassified_indices": [r["index"] for r in records if not r["correct"]],
        "emotion_agreement": emotion_agreement,
    }


def print_report(summary: dict, out_dir: str) -> None:
    m = summary["meta"]
    print(f"\n=== {m['mode']}-class run, {m['sample']['method']} sample of {summary['total_reviews']} ===")
    print(f"Class counts in this sample (from rating): {summary['class_distribution']}")
    print(f"Whole-file class counts:                   {m['dataset']['class_counts']}")
    print(f"Overall accuracy:  {summary['overall_accuracy']:.1%}   "
          f"balanced accuracy: {summary['balanced_accuracy']:.1%}   "
          f"always-'{summary['majority_baseline']['class']}' baseline: {summary['majority_baseline']['accuracy']:.1%}")
    print(f"Accuracy if weighted to the whole file's real class mix: {summary['file_mix_weighted_accuracy']['value']:.1%}")
    print(f"Unparsed: {summary['unparsed_responses']}   Truncated: {summary['truncated_responses']}   "
          f"Star-count titles hidden from model: {m['titles_hidden_from_model']}")
    for c, s in summary["per_class"].items():
        rec = f"{s['recall']:.1%}" if s["recall"] is not None else "n/a"
        prec = f"{s['precision']:.1%}" if s["precision"] is not None else "n/a"
        ci = f"[{s['ci_low']:.0%}-{s['ci_high']:.0%}]" if s["ci_low"] is not None else ""
        print(f"  {c:<9} support={s['support']:<4} recall={rec:<6} 95%CI{ci:<10} precision={prec}")
    print("Confusion matrix (rows = rating-derived truth, cols = model answer):")
    for t, row in summary["confusion_matrix"].items():
        print(f"  {t:<9} {row}")
    ea = summary["emotion_agreement"]
    print("Emotion (LLM vs NRC word list): "
          f"agree {ea['agree_count']}/{ea['reviews_with_both_emotions']}"
          + (f" ({ea['agreement_rate']:.1%})" if ea["agreement_rate"] is not None else "")
          + f"; word list had no lexicon hits in {ea['reviews_wordlist_had_no_lexicon_words']}"
          + f"; LLM gave no valid emotion in {ea['reviews_llm_had_no_valid_emotion']}")
    print(f"Saved {out_dir}/records.json and {out_dir}/summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=list(CLASSES), default="three")
    parser.add_argument("--sample", choices=["first", "balanced"], default="balanced")
    parser.add_argument("--n", type=int, default=100, help="rows to read for --sample first")
    parser.add_argument("--per-class", type=int, default=50, help="rows per class for --sample balanced")
    parser.add_argument("--seed", type=int, default=42, help="random seed for --sample balanced")
    parser.add_argument("--data", default="data/Gift_Cards.jsonl")
    parser.add_argument("--out", default="output/step6")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resummarize", metavar="RUN_DIR",
                        help="rebuild RUN_DIR/summary.json from its saved records.json (no model calls) and exit")
    parser.add_argument("--keep-star-titles", action="store_true",
                        help="do NOT hide star-count titles like 'Three Stars' (leaky baseline, for the ablation only)")
    args = parser.parse_args()

    if args.resummarize:
        with open(os.path.join(args.resummarize, "records.json")) as f:
            saved = json.load(f)
        with open(os.path.join(args.resummarize, "summary.json")) as f:
            old_meta = json.load(f)["meta"]
        if os.path.exists(args.data):
            old_meta["dataset"] = dataset_stats(load_all_rows(args.data))
        summary = summarize(saved, old_meta["mode"], old_meta)
        with open(os.path.join(args.resummarize, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print_report(summary, args.resummarize)
        return

    if args.sample == "balanced" and args.mode != "three":
        parser.error("--sample balanced needs --mode three (it samples across all three rating classes)")

    rows = load_all_rows(args.data)
    stats = dataset_stats(rows)
    if args.sample == "first":
        indices = pick_first(rows, args.n)
        sample_meta = {"method": "first", "n": len(indices)}
    else:
        indices = pick_balanced(rows, args.per_class, args.seed)
        sample_meta = {"method": "balanced", "per_class": args.per_class, "seed": args.seed, "n": len(indices)}
    print(f"Loaded {len(rows):,} rows; scoring {len(indices)} ({sample_meta['method']} sample, mode={args.mode})")

    records = [None] * len(indices)
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_one, pos, fi, rows[fi], args.mode, not args.keep_star_titles): pos for pos, fi in enumerate(indices)}
        for done, future in enumerate(as_completed(futures), 1):
            records[futures[future]] = future.result()
            if done % 25 == 0 or done == len(indices):
                print(f"  scored {done}/{len(indices)}")
    print(f"Done in {time.time() - start:.1f}s")

    meta = {"mode": args.mode, "sample": sample_meta, "model": MODEL, "settings": SETTINGS, "dataset": stats,
            "mask_star_titles": not args.keep_star_titles}
    summary = summarize(records, args.mode, meta)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "records.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print_report(summary, args.out)


if __name__ == "__main__":
    main()
