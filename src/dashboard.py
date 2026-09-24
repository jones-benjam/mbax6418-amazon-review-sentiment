"""
Interactive Gradio dashboard (Steps 3-7).

Shows two saved scoring runs side by side in tabs:
  * Balanced three-class run (Step 6)  -- output/step6
  * Lopsided first-100 binary run (Step 2) -- output/step2
Everything on the page is computed from the saved records.json / summary.json;
nothing is scored live.

Run:
    ./venv/bin/python src/dashboard.py [--balanced output/step6] [--lopsided output/step2]
        [--start-tab balanced|lopsided] [--port 7860]
"""
import argparse
import html
import json
from pathlib import Path

import gradio as gr

# --- palette (light-mode slice of the project's validated dataviz palette) ---
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BORDER = "rgba(11,11,11,0.10)"
BLUE = "#2a78d6"    # POSITIVE  (diverging pair: blue <-> red, gray midpoint)
RED = "#e34948"     # NEGATIVE
GRAY = "#898781"    # NEUTRAL
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
CLASS_COLOR = {"POSITIVE": BLUE, "NEUTRAL": GRAY, "NEGATIVE": RED}
CLASS_TEXT = {"POSITIVE": BLUE, "NEUTRAL": TEXT_SECONDARY, "NEGATIVE": RED}
UNPARSED = "UNPARSED"
NAVY = "#0d366b"      # single-hue ink for the two-treatment bar charts (outline = reference, solid = model/sample)
TRACK = "#efeee9"

CUSTOM_CSS = f"""
.gradio-container {{
    background: {PAGE} !important;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
    /* Gradio's default theme injects dark-mode CSS variables that turn our text
       near-white on a light page. Override its theme variables directly. */
    color-scheme: light !important;
    --body-text-color: {TEXT_PRIMARY} !important;
    --body-text-color-subdued: {TEXT_MUTED} !important;
    --background-fill-primary: {SURFACE} !important;
    --background-fill-secondary: {PAGE} !important;
    --block-background-fill: {SURFACE} !important;
    --block-border-color: {BORDER} !important;
    --border-color-primary: {BORDER} !important;
    --body-background-fill: {PAGE} !important;
    --input-background-fill: {SURFACE} !important;
    --input-background-fill-focus: {SURFACE} !important;
    --input-background-fill-hover: {SURFACE} !important;
    --input-border-color: #c3c2b7 !important;
    --input-border-color-hover: {TEXT_MUTED} !important;
    --input-border-color-focus: {BLUE} !important;
    --input-shadow-focus: 0 0 0 1px {BLUE} !important;
    --input-placeholder-color: {TEXT_MUTED} !important;
    --table-row-focus: {PAGE} !important;
    --link-text-color: {BLUE} !important;
}}
/* Force dark text on Gradio's own Markdown blocks only. A blanket .prose rule also flattened every inline
   color inside our gr.HTML blocks (dark matrix cells became black-on-navy, status colors turned black). */
.gradio-container, .gradio-container .md, .gradio-container .md * {{ color: {TEXT_PRIMARY} !important; }}
.gradio-container table, .gradio-container tr, .gradio-container th, .gradio-container td {{
    border: 0 !important; border-color: {GRIDLINE} !important;
}}
.stat-tile {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 18px; }}
.stat-tile .label {{ font-size: 12px; color: {TEXT_SECONDARY}; margin-bottom: 6px; }}
.stat-tile .value {{ font-size: 28px; font-weight: 700; color: {TEXT_PRIMARY}; letter-spacing: -0.01em; }}
.stat-tile .sub {{ font-size: 12px; color: {TEXT_MUTED}; margin-top: 4px; line-height: 1.4; }}
.tile-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }}
"""


# ----------------------------------------------------------------- helpers --
def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def pct(v, digits: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def tile(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="stat-tile"><div class="label">{esc(label)}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def card(inner: str, pad: str = "18px 20px") -> str:
    return f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;padding:{pad};">{inner}</div>'


def load_run(run_dir: str) -> dict:
    d = Path(run_dir)
    with open(d / "records.json") as f:
        records = json.load(f)
    with open(d / "summary.json") as f:
        summary = json.load(f)
    sens = d / "seed_sensitivity.json"
    sensitivity = None
    if sens.exists():
        with open(sens) as f:
            sensitivity = json.load(f)
    return {"records": records, "summary": summary, "sensitivity": sensitivity}


# ------------------------------------------------------------ page sections --
def hidden_note(meta: dict) -> str:
    n = meta.get("titles_hidden_from_model", 0)
    if not meta.get("mask_star_titles", True) or not n:
        return ""
    return (f' &mdash; including in review titles: {n} reviews had a title that only states the star count '
            f'(e.g. &ldquo;Three Stars&rdquo;), which would leak the answer, so those titles were hidden from the model')


def render_run_header(summary: dict) -> str:
    m = summary["meta"]
    s = m["sample"]
    if s["method"] == "balanced":
        how = (f'{s["per_class"]} reviews per class drawn at random from the whole file '
               f'(seed {s["seed"]}), {len(summary["classes"])}-class labels')
    else:
        how = f'the first {s["n"]} reviews in file order, {len(summary["classes"])}-class labels'
    return (f'<div style="font-size:13px;color:{TEXT_SECONDARY};line-height:1.6;">'
            f'<strong style="color:{TEXT_PRIMARY};">Sample:</strong> {esc(how)}. '
            f'<strong style="color:{TEXT_PRIMARY};">Model:</strong> {esc(m["model"])} '
            f'(temperature {m["settings"]["temperature"]}). '
            f'The star rating is the answer key (4-5 POSITIVE, 3 NEUTRAL, 1-2 NEGATIVE) and is never shown to the model'
            f'{hidden_note(m)}.</div>')


def render_headline_tiles(summary: dict) -> str:
    base = summary["majority_baseline"]
    unp = summary["unparsed_responses"]
    tiles = [
        tile("Reviews scored", str(summary["total_reviews"]),
             f"{unp} unparsed (counted as wrong)" if unp else "all answers parsed"),
        tile("Overall accuracy", pct(summary["overall_accuracy"]), "share of reviews where the model matched the rating-derived label"),
        tile("Balanced accuracy", pct(summary["balanced_accuracy"]), "average of the per-class recalls, so every class counts equally"),
        tile("Always-guess-" + base["class"], pct(base["accuracy"]), "what a model that ignores the review entirely would score"),
        tile("On the real review mix", pct(summary["file_mix_weighted_accuracy"]["value"]),
             "per-class recalls weighted by how common each class is in the whole file"),
    ]
    return f'<div class="tile-grid">{"".join(tiles)}</div>'


def render_class_tiles(summary: dict) -> str:
    tiles = []
    for c, s in summary["per_class"].items():
        ci = f'95% CI {pct(s["ci_low"], 0)}&ndash;{pct(s["ci_high"], 0)}' if s["ci_low"] is not None else ""
        sub = f'{s["correct"]} of {s["support"]} {esc(c)} reviews correct &middot; {ci}'
        tiles.append(
            f'<div class="stat-tile" data-tile="recall" data-label="{esc(c)}" data-value="{(s["recall"] or 0):.6f}" style="border-top:3px solid {CLASS_COLOR.get(c, GRAY)};">'
            f'<div class="label">{esc(c)} recall (how often a true {esc(c)} review is answered right)</div>'
            f'<div class="value">{pct(s["recall"])}</div><div class="sub">{sub}</div></div>')
    return f'<div class="tile-grid">{"".join(tiles)}</div>'


def render_callout(run: dict) -> str:
    summary, sens = run["summary"], run["sensitivity"]
    m = summary["meta"]
    ds = m["dataset"]
    total = ds["total_rows"]
    file_mix = ", ".join(f'{c} {n / total:.1%}' for c, n in ds["class_counts"].items())
    base = summary["majority_baseline"]
    if m["sample"]["method"] == "first":
        dist = summary["class_distribution"]
        n = summary["total_reviews"]
        mix = ", ".join(f"{c} {k} ({k / n:.0%})" for c, k in dist.items())
        minority = min(summary["per_class"].items(), key=lambda kv: kv[1]["support"])
        ci = minority[1]
        body = (f"This batch is {mix}. A model that just answered {base['class']} every time would already "
                f"score {pct(base['accuracy'])}, so the model's {pct(summary['overall_accuracy'])} overall is only "
                f"{(summary['overall_accuracy'] - base['accuracy']) * 100:.0f} points above that. Balanced accuracy "
                f"(every class weighted equally) is {pct(summary['balanced_accuracy'])}, and the "
                f"{minority[0]} result rests on just {ci['support']} reviews (95% CI "
                f"{pct(ci['ci_low'], 0)}&ndash;{pct(ci['ci_high'], 0)}). Whole file: {file_mix}.")
        title, color = "Imbalance warning.", STATUS_CRITICAL
    else:
        s = m["sample"]
        body = (f"Balanced by design: {s['per_class']} reviews per class, seed {s['seed']}. The real file is "
                f"{file_mix} &mdash; these per-class numbers say how the model does on each kind of review, "
                f"<em>not</em> how it would score on the real mix. With only {s['per_class']} per class the "
                f"intervals are wide.")
        if sens:
            accs = [v["overall_accuracy"] for v in sens.values()]
            body += (f" Across {len(sens)} different random draws (this seed included), overall accuracy ranged from "
                     f"{pct(min(accs))} to {pct(max(accs))}.")
        title, color = "Read these numbers carefully.", STATUS_GOOD
    return (f'<div style="display:flex;gap:10px;align-items:flex-start;background:{SURFACE};border:1px solid {BORDER};'
            f'border-left:3px solid {color};border-radius:8px;padding:14px 16px;font-size:13px;color:{TEXT_SECONDARY};'
            f'line-height:1.55;"><div><strong style="color:{TEXT_PRIMARY};">{title}</strong> {body}</div></div>')


def render_confusion(summary: dict) -> str:
    cm = summary["confusion_matrix"]
    classes = summary["classes"]
    show_unparsed = any(cm[t][UNPARSED] for t in classes)
    cols = classes + ([UNPARSED] if show_unparsed else [])
    th = f'padding:10px 18px;color:{TEXT_MUTED};font-size:11px;text-transform:uppercase;text-align:center;'
    head = "".join(f'<th style="{th}">{esc(c)}</th>' for c in cols)
    rows = ""
    for t in classes:
        total = sum(cm[t].values()) or 1
        cells = ""
        for p in cols:
            v = cm[t][p]
            frac = v / total
            shade = SEQ_STEPS[min(len(SEQ_STEPS) - 1, int(frac * len(SEQ_STEPS)))]
            fg = "#fff" if frac > 0.55 else TEXT_PRIMARY
            tip = f"{v} of {total} true {t} reviews were answered {p} ({frac:.0%})"
            cells += (f'<td data-cell="{esc(t)}>{esc(p)}" data-n="{v}" title="{esc(tip)}" style="padding:12px 18px;text-align:center;border-radius:6px;'
                      f'background:{shade};color:{fg};"><div style="font-size:18px;font-weight:700;color:{fg};">{v}</div>'
                      f'<div style="font-size:11px;opacity:.85;color:{fg};">{frac:.0%}</div></td>')
        rows += (f'<tr><td style="padding:12px 14px;text-align:right;font-weight:600;color:{TEXT_SECONDARY};">'
                 f'{esc(t)}</td>{cells}</tr>')
    note = (f'<div style="font-size:12px;color:{TEXT_MUTED};margin-top:10px;max-width:520px;line-height:1.6;">'
            f'Rows = the answer key from the star rating. Columns = what the model said. The diagonal is correct; '
            f'everything off it is a mistake and shows which class got mistaken for which. Percentages are of that row; '
            f'darker = larger share.</div>')
    return card(f'<table style="border-collapse:separate;border-spacing:3px;font-size:13px;">'
                f'<tr><th></th>{head}</tr>{rows}</table>{note}', "20px 22px")


def render_emotion_summary(summary: dict) -> str:
    ea = summary["emotion_agreement"]
    emotions = sorted(set(ea["llm_emotion_distribution"]) | set(ea["wordlist_emotion_distribution"]))
    th = f'padding:8px 16px;color:{TEXT_MUTED};font-size:11px;text-transform:uppercase;'
    line = f"border-bottom:1px solid {GRIDLINE} !important;"
    rows = "".join(
        f'<tr><td style="padding:8px 16px;{line}color:{TEXT_SECONDARY};font-weight:600;">{esc(e)}</td>'
        f'<td style="padding:8px 16px;{line}text-align:right;">{ea["llm_emotion_distribution"].get(e, 0)}</td>'
        f'<td style="padding:8px 16px;{line}text-align:right;">{ea["wordlist_emotion_distribution"].get(e, 0)}</td></tr>'
        for e in emotions)
    tiles = "".join([
        tile("Emotion agreement", pct(ea["agreement_rate"]),
             f'{ea["agree_count"]} of {ea["reviews_with_both_emotions"]} reviews where both methods gave an answer'),
        tile("Word list found nothing", str(ea["reviews_wordlist_had_no_lexicon_words"]), "reviews with zero lexicon-word hits"),
        tile("LLM gave no valid emotion", str(ea["reviews_llm_had_no_valid_emotion"]), "answer was outside the 8 allowed emotions"),
    ])
    table = card(f'<table style="border-collapse:collapse;font-size:13px;width:100%;">'
                 f'<tr><th style="{th}text-align:left;">Emotion</th><th style="{th}text-align:right;">LLM</th>'
                 f'<th style="{th}text-align:right;">Word list</th></tr>{rows}</table>', "12px 14px")
    return (f'<div style="display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start;">'
            f'<div class="tile-grid" style="flex:2;min-width:300px;">{tiles}</div>'
            f'<div style="flex:1;min-width:260px;">{table}</div></div>')



# ------------------------------------------------------------------- charts --
# Hand-built HTML bars. Values are printed in their own column beside each bar
# (never overlaid on it) and every non-zero bar has a minimum width, so small
# values cannot collapse to nothing. Each mark carries data-* attributes with its
# exact value so the page can be checked against the saved JSON in the browser.
MIN_BAR_PX = 6


def swatch(kind: str) -> str:
    if kind == "ref":
        return f'<span style="display:inline-block;width:22px;height:10px;border:2px solid {NAVY};border-radius:0 3px 3px 0;box-sizing:border-box;"></span>'
    return f'<span style="display:inline-block;width:22px;height:10px;background:{NAVY};border-radius:0 3px 3px 0;"></span>'


def legend(items: list) -> str:
    inner = "".join(f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px;">{swatch(k)}{esc(t)}</span>'
                    for k, t in items)
    return f'<div style="font-size:12px;color:{TEXT_SECONDARY};margin:2px 0 10px;">{inner}</div>'


def chart_card(title: str, blurb: str, body: str) -> str:
    return card(f'<div style="font-weight:700;font-size:14px;margin-bottom:4px;">{esc(title)}</div>'
                f'<div style="font-size:12px;color:{TEXT_SECONDARY};line-height:1.5;margin-bottom:10px;">{blurb}</div>{body}')


def solid_or_outline(kind: str, frac: float, attrs: str) -> str:
    """One bar. A zero value draws nothing (an outlined bar would otherwise show its 4px border)."""
    if frac <= 0:
        return f'<div {attrs} style="height:0;"></div>'
    w = f"width:{frac * 100:.2f}%;min-width:{MIN_BAR_PX}px;height:12px;"
    if kind == "ref":
        return f'<div {attrs} style="{w}border:2px solid {NAVY};border-radius:0 3px 3px 0;box-sizing:border-box;"></div>'
    return f'<div {attrs} style="{w}background:{NAVY};border-radius:0 3px 3px 0;"></div>'


def grouped_bars(chart_id: str, rows: list, scale_max: float, ref_name: str, val_name: str) -> str:
    """rows: dicts with label, ref, ref_text, val, val_text, note. Outlined bar = reference, solid = the run."""
    out = legend([("ref", ref_name), ("val", val_name)])
    for r in rows:
        line = ""
        for kind, v, text, name in [("ref", r["ref"], r["ref_text"], ref_name), ("val", r["val"], r["val_text"], val_name)]:
            attrs = (f'data-chart="{chart_id}" data-series="{kind}" data-label="{esc(r["key"])}" '
                     f'data-value="{v}" title="{esc(r["key"])} \u2014 {esc(name)}: {esc(text)}"')
            line += (f'<div style="display:grid;grid-template-columns:1fr 128px;gap:10px;align-items:center;margin:3px 0;">'
                     f'<div>{solid_or_outline(kind, (v / scale_max) if scale_max else 0, attrs)}</div>'
                     f'<div style="font-size:12px;color:{TEXT_SECONDARY};white-space:nowrap;">{text}</div></div>')
        note = f'<div style="font-size:11px;color:{TEXT_MUTED};margin:0 0 2px;">{r["note"]}</div>' if r.get("note") else ""
        out += f'<div style="margin:12px 0 14px;"><div style="font-size:13px;font-weight:600;">{r["label"]}</div>{line}{note}</div>'
    return out


def render_recall_chart(summary: dict) -> str:
    k = len(summary["classes"])
    rows = ""
    for c, s in summary["per_class"].items():
        rec = s["recall"] or 0.0
        color = CLASS_COLOR.get(c, GRAY)
        title = f'{c}: {s["correct"]} of {s["support"]} answered right ({pct(rec)})'
        bar = (f'<div data-chart="recall" data-label="{esc(c)}" data-value="{rec:.6f}" title="{esc(title)}" '
               f'style="height:18px;width:{rec * 100:.2f}%;min-width:{MIN_BAR_PX if rec > 0 else 0}px;'
               f'background:{color};border-radius:0 4px 4px 0;"></div>')
        whisker = ""
        if s["ci_low"] is not None:
            lo, hi = s["ci_low"] * 100, s["ci_high"] * 100
            whisker = (f'<div title="95% confidence interval {pct(s["ci_low"], 0)}\u2013{pct(s["ci_high"], 0)}" '
                       f'style="position:absolute;left:{lo:.2f}%;width:max({hi - lo:.2f}%,2px);top:50%;height:2px;margin-top:-1px;background:{TEXT_PRIMARY};"></div>'
                       f'<div style="position:absolute;left:{lo:.2f}%;top:3px;bottom:3px;width:2px;background:{TEXT_PRIMARY};"></div>'
                       f'<div style="position:absolute;left:calc({hi:.2f}% - 2px);top:3px;bottom:3px;width:2px;background:{TEXT_PRIMARY};"></div>')
        chance = (f'<div style="position:absolute;left:{100 / k:.2f}%;top:-4px;bottom:-4px;border-left:1px dashed {TEXT_MUTED};"></div>')
        rows += (f'<div style="display:grid;grid-template-columns:86px 1fr 96px;gap:10px;align-items:center;margin:10px 0;">'
                 f'<div style="font-size:13px;font-weight:600;color:{CLASS_TEXT.get(c, TEXT_SECONDARY)};">{esc(c)}</div>'
                 f'<div style="position:relative;height:18px;background:{TRACK};border-radius:4px;">{bar}{chance}{whisker}</div>'
                 f'<div style="font-size:13px;white-space:nowrap;"><strong>{pct(rec, 0)}</strong> '
                 f'<span style="color:{TEXT_MUTED};font-size:12px;">{s["correct"]}/{s["support"]}</span></div></div>')
    key = (f'<div style="font-size:11px;color:{TEXT_MUTED};margin-top:6px;">Whisker = 95% confidence interval. '
           f'Dashed line = {100 / k:.0f}%, what random guessing among {k} classes would get.</div>')
    return chart_card("How often each class was answered right",
                      "Of the reviews that truly belong to each class, the share the model got right. Short bars are where it fails.",
                      rows + key)


def render_key_vs_model_chart(summary: dict) -> str:
    cm, classes = summary["confusion_matrix"], summary["classes"]
    n = summary["total_reviews"]
    key_n = {c: sum(cm[c].values()) for c in classes}
    said_n = {c: sum(cm[t][c] for t in classes) for c in classes}
    scale = max(list(key_n.values()) + list(said_n.values())) or 1
    rows = []
    for c in classes:
        diff = said_n[c] - key_n[c]
        note = (f"model over-used this label by {diff}" if diff > 0 else
                f"model under-used this label by {-diff}" if diff < 0 else "model used it exactly as often as the key")
        rows.append({"key": c, "label": f'<span style="color:{CLASS_TEXT.get(c, TEXT_SECONDARY)};">{esc(c)}</span>',
                     "ref": key_n[c], "ref_text": f'{key_n[c]} <span style="color:{TEXT_MUTED};">({key_n[c] / n:.0%})</span>',
                     "val": said_n[c], "val_text": f'{said_n[c]} <span style="color:{TEXT_MUTED};">({said_n[c] / n:.0%})</span>',
                     "note": note})
    return chart_card("Answer key vs. what the model said",
                      "How many reviews truly belong to each class (from the stars) next to how many the model put there. "
                      "A solid bar longer than its outline means the model over-uses that label.",
                      grouped_bars("keyvsmodel", rows, scale, "Answer key (from stars)", "Model said"))


def render_star_chart(run: dict) -> str:
    summary, records = run["summary"], run["records"]
    ds = summary["meta"]["dataset"]
    file_total = ds["total_rows"]
    n = len(records)
    rows = []
    for star in range(1, 6):
        fc = ds["rating_counts"][str(star)]
        sc = sum(1 for r in records if int(r["rating"]) == star)
        rows.append({"key": f"{star} star{'s' if star > 1 else ''}", "label": "\u2605" * star + "\u2606" * (5 - star),
                     "ref": fc / file_total, "ref_text": f'{fc / file_total:.1%} <span style="color:{TEXT_MUTED};">({fc:,})</span>',
                     "val": sc / n, "val_text": f'{sc / n:.1%} <span style="color:{TEXT_MUTED};">({sc})</span>'})
    return chart_card("Star-rating distribution",
                      f"Share of reviews at each star rating: the whole file ({file_total:,} reviews) next to this run's sample. "
                      "The real data is overwhelmingly 5-star; sampling changes that on purpose.",
                      grouped_bars("stars", rows, 1.0, "Whole file", "This run's sample"))


def render_charts(run: dict) -> str:
    cards = "".join([render_recall_chart(run["summary"]), render_key_vs_model_chart(run["summary"]), render_star_chart(run)])
    return f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px;align-items:start;">{cards}</div>'


# -------------------------------------------------------------- review table --
def star_string(rating: float) -> str:
    full = max(0, min(5, round(rating)))
    return "★" * full + "☆" * (5 - full)


def build_filters(records: list, summary: dict) -> dict:
    classes = summary["classes"]
    cm = summary["confusion_matrix"]
    f = {
        "All": lambda r: True,
        "Correct only": lambda r: r["correct"],
        "Incorrect only": lambda r: not r["correct"],
    }
    for c in classes:
        f[f"True label: {c}"] = lambda r, c=c: r["true_label"] == c
    for k in sorted({int(r["rating"]) for r in records}):
        f[f"Rating: {k} star{'s' if k != 1 else ''}"] = lambda r, k=k: int(r["rating"]) == k
    for t in classes:
        for p in classes + [UNPARSED]:
            n = cm[t][p]
            if t != p and n:
                f[f"Mistake: true {t} → answered {p} ({n})"] = (
                    lambda r, t=t, p=p: r["true_label"] == t and (r["predicted_label"] or UNPARSED) == p)
    f["Emotions agree"] = lambda r: r["emotions_agree"]
    f["Emotions differ"] = lambda r: bool(r["llm_emotion"] and r["wordlist_emotion"] and not r["emotions_agree"])
    return f


def hidden_tag(r: dict) -> str:
    """Marks titles that only state the star count: the model never saw them."""
    if not r.get("title_masked"):
        return ""
    return (f'<div style="font-weight:400;font-size:11px;color:{TEXT_MUTED};" '
            f'title="A title that only states the star count would reveal the rating, so it was hidden from the model.">'
            f'hidden from model</div>')


def render_review_rows(records: list) -> str:
    cell = f'padding:10px 14px;border-bottom:1px solid {GRIDLINE} !important;'
    rows = ""
    for r in records:
        pred = r["predicted_label"]
        text = r["text"] or ""
        short = (text[:200].strip() + "…") if len(text) > 200 else text
        agree = r["emotions_agree"]
        # Review text is untrusted user content (and often contains literal "<br />"): always escape.
        rows += (
            f'<tr>'
            f'<td title="row {r["file_index"]} of Gift_Cards.jsonl" style="{cell}color:{TEXT_MUTED};">{r["index"]}</td>'
            f'<td style="{cell}color:{TEXT_MUTED};white-space:nowrap;">{star_string(r["rating"])}</td>'
            f'<td style="{cell}font-weight:600;">{esc(r["title"])}{hidden_tag(r)}</td>'
            f'<td title="{esc(text)}" style="{cell}color:{TEXT_SECONDARY};min-width:260px;max-width:380px;">{esc(short)}</td>'
            f'<td style="{cell}color:{CLASS_TEXT.get(r["true_label"], TEXT_SECONDARY)};font-weight:600;white-space:nowrap;">{esc(r["true_label"])}</td>'
            f'<td style="{cell}color:{CLASS_TEXT.get(pred, TEXT_MUTED)};font-weight:600;white-space:nowrap;">{esc(pred or UNPARSED)}</td>'
            f'<td style="{cell}color:{STATUS_GOOD if r["correct"] else STATUS_CRITICAL};font-weight:600;white-space:nowrap;">'
            f'{"✓ Correct" if r["correct"] else "✗ Incorrect"}</td>'
            f'<td style="{cell}color:{TEXT_SECONDARY};white-space:nowrap;">{esc(r["llm_emotion"] or "—")}</td>'
            f'<td style="{cell}color:{TEXT_SECONDARY};white-space:nowrap;">{esc(r["wordlist_emotion"] or "—")}</td>'
            f'<td style="{cell}color:{STATUS_GOOD if agree else TEXT_MUTED};white-space:nowrap;">{"✓ Agree" if agree else "—"}</td>'
            f'</tr>')
    heads = ["#", "Rating", "Title", "Text", "True label", "Model said", "Result", "LLM emotion", "Word-list emotion", "Emotions"]
    header = "".join(
        f'<th style="padding:10px 14px;text-align:left;font-size:11px;text-transform:uppercase;color:{TEXT_MUTED};'
        f'border-bottom:1px solid {GRIDLINE} !important;">{h}</th>' for h in heads)
    return (f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;overflow:hidden;">'
            f'<div style="max-height:520px;overflow:auto;">'
            f'<table style="border-collapse:collapse;width:100%;min-width:900px;font-size:13px;">'
            f'<thead style="position:sticky;top:0;background:{SURFACE};"><tr>{header}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div></div>')


# ------------------------------------------------------------------- the app --
def build_run_view(run: dict) -> None:
    """Lay out one run's sections inside the current Gradio container."""
    records, summary = run["records"], run["summary"]
    filters = build_filters(records, summary)

    gr.HTML(render_run_header(summary))
    gr.HTML(render_headline_tiles(summary))
    gr.HTML(render_class_tiles(summary))
    gr.HTML(render_callout(run))
    gr.Markdown("### The data and the model's results at a glance")
    gr.HTML(render_charts(run))
    gr.Markdown("### Which answers were right, and which class got mistaken for which")
    gr.HTML(render_confusion(summary))
    gr.Markdown("### Primary emotion: LLM vs. NRC word list")
    gr.HTML(render_emotion_summary(summary))

    gr.Markdown("### Every review")
    with gr.Row():
        dropdown = gr.Dropdown(choices=list(filters), value="All", label="Show", scale=1)
        count = gr.Markdown(f"Showing **{len(records)}** of **{len(records)}** reviews")
    table = gr.HTML(render_review_rows(records))

    def on_change(choice):
        kept = [r for r in records if filters[choice](r)]
        return render_review_rows(kept), f"Showing **{len(kept)}** of **{len(records)}** reviews"

    dropdown.change(on_change, inputs=dropdown, outputs=[table, count])


def build_app(balanced_dir: str, lopsided_dir: str, start_tab: str = "balanced") -> gr.Blocks:
    balanced, lopsided = load_run(balanced_dir), load_run(lopsided_dir)
    with gr.Blocks(title="Gift Card Review Sentiment Dashboard") as demo:
        gr.Markdown("# Gift Card Review Sentiment Dashboard")
        gr.Markdown(
            "How well does an LLM read the sentiment and emotion of Amazon gift-card reviews, judged against "
            "the reviewers' own star ratings? Two runs are shown: a balanced three-class run (the fair test) "
            "and the original lopsided first-100 run (which flatters the model).")
        with gr.Tabs(selected=start_tab):
            with gr.Tab("Balanced 3-class run (Step 6)", id="balanced"):
                build_run_view(balanced)
            with gr.Tab("Lopsided first-100 run (Step 2)", id="lopsided"):
                build_run_view(lopsided)
        gr.Markdown(
            "---\nData: Amazon Reviews '23, “Gift Cards” category (McAuley Lab, UC San Diego) — "
            "https://amazon-reviews-2023.github.io. Sentiment and emotion are predicted by an LLM from the "
            "review title and text only; the star rating is used solely to check the model afterward.")
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced", default="output/step6")
    parser.add_argument("--lopsided", default="output/step2")
    parser.add_argument("--start-tab", choices=["balanced", "lopsided"], default="balanced")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_app(args.balanced, args.lopsided, args.start_tab).launch(server_port=args.port, share=args.share, css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
