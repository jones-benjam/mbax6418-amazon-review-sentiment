"""
Independent check of the saved results.

Recomputes every headline number from records.json alone -- deriving the answer
key from the star ratings itself and NOT calling score_batch.summarize() -- and
compares it with summary.json. Also recounts the whole-file rating distribution
from the raw data file when it is available.

Usage:
    python src/verify_numbers.py [--runs output/step2 output/step6] [--data data/Gift_Cards.jsonl]
Exit code 1 if anything disagrees.
"""
import argparse
import collections
import json
import os
import re
import sys


def answer_key(rating: float, classes: list) -> str:
    if rating >= 4:
        return "POSITIVE"
    if "NEUTRAL" in classes and rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def check_run(run_dir: str, failures: list) -> dict:
    with open(os.path.join(run_dir, "records.json")) as f:
        recs = json.load(f)
    with open(os.path.join(run_dir, "summary.json")) as f:
        s = json.load(f)
    classes = s["classes"]

    def expect(name, got, want):
        ok = got == want or (isinstance(got, float) and isinstance(want, float) and abs(got - want) < 1e-9)
        if isinstance(got, list):
            detail = f"{len(got)} labels, {sum(1 for a, b in zip(got, want) if a != b)} differ"
        else:
            detail = f"recomputed {got!r} vs saved {want!r}"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            failures.append(f"{run_dir}: {name}")

    print(f"{run_dir}  ({len(recs)} records, classes {classes})")
    keys = [answer_key(r["rating"], classes) for r in recs]
    expect("stored true_label matches rating rule", [r["true_label"] for r in recs], keys)
    n = len(recs)
    expect("total_reviews", n, s["total_reviews"])
    right = sum(1 for r, k in zip(recs, keys) if r["predicted_label"] == k)
    expect("overall_accuracy", right / n, s["overall_accuracy"])

    cm = {t: collections.Counter() for t in classes}
    for r, k in zip(recs, keys):
        cm[k][r["predicted_label"] or "UNPARSED"] += 1
    for t in classes:
        for p in classes:
            expect(f"confusion[{t}->{p}]", cm[t][p], s["confusion_matrix"][t][p])
    recalls = []
    for c in classes:
        support = sum(cm[c].values())
        recall = cm[c][c] / support
        recalls.append(recall)
        expect(f"recall[{c}]", recall, s["per_class"][c]["recall"])
        predicted = sum(cm[t][c] for t in classes)
        expect(f"model_said[{c}]", predicted, s["per_class"][c]["predicted"])
    expect("balanced_accuracy", sum(recalls) / len(recalls), s["balanced_accuracy"])

    cc = s["meta"]["dataset"]["class_counts"]
    share = ({"POSITIVE": cc["POSITIVE"], "NEGATIVE": cc["NEUTRAL"] + cc["NEGATIVE"]} if "NEUTRAL" not in classes
             else {c: cc[c] for c in classes})
    weighted = sum(rc * share[c] / sum(share.values()) for c, rc in zip(classes, recalls))
    expect("file_mix_weighted_accuracy", weighted, s["file_mix_weighted_accuracy"]["value"])

    both = [r for r in recs if r["llm_emotion"] and r["wordlist_emotion"]]
    agree = sum(1 for r in both if r["llm_emotion"] == r["wordlist_emotion"])
    ea = s["emotion_agreement"]
    expect("emotion agree_count", agree, ea["agree_count"])
    expect("emotion reviews_with_both", len(both), ea["reviews_with_both_emotions"])
    expect("wordlist found nothing", sum(1 for r in recs if r["wordlist_emotion"] is None),
           ea["reviews_wordlist_had_no_lexicon_words"])

    stars = collections.Counter(int(r["rating"]) for r in recs)
    print(f"  info  star counts in this sample: {dict(sorted(stars.items()))}")

    expected = s["meta"]["dataset"]
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["output/step2", "output/step6"])
    parser.add_argument("--data", default="data/Gift_Cards.jsonl")
    args = parser.parse_args()

    failures = []
    dataset = None
    for run in args.runs:
        dataset = check_run(run, failures)
        print()

    if os.path.exists(args.data):
        counts = collections.Counter()
        with open(args.data) as f:
            for line in f:
                counts[int(json.loads(line)["rating"])] += 1
        print("Whole-file rating counts recounted from the raw data file")
        for star in range(1, 6):
            want = dataset["rating_counts"][str(star)]
            ok = counts[star] == want
            print(f"  {'PASS' if ok else 'FAIL'}  {star} star: recounted {counts[star]:,} vs saved {want:,}")
            if not ok:
                failures.append(f"file rating count {star}")
        pat = re.compile(r"^\s*(one|two|three|four|five|[1-5])[\s-]*stars?[.!]*\s*$", re.IGNORECASE)
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        total = matches = 0
        with open(args.data) as f:
            for line in f:
                r = json.loads(line)
                m = pat.match(r.get("title") or "")
                if m:
                    total += 1
                    matches += (words.get(m.group(1).lower()) or int(m.group(1))) == int(r["rating"])
        want = dataset["star_count_titles"]
        for name, got, exp in [("star-count titles", total, want["total"]), ("...that state the true rating", matches, want["title_number_matches_rating"])]:
            ok = got == exp
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: recounted {got:,} vs saved {exp:,}")
            if not ok:
                failures.append(name)
    else:
        print(f"(skipping whole-file recount: {args.data} not found)")

    print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
