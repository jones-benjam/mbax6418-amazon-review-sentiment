"""
Interactive Gradio dashboard (Step 3 + Step 4).

Replaces the earlier static-HTML dashboard generator. Reads the saved
scoring output (records.json + summary.json) and serves an interactive
app: headline KPIs, a confusion-matrix heatmap, and a per-review table
the reader can filter (Step 4) with a live count.

Run:
    ./venv/bin/python src/dashboard.py [--records output/step2/records.json]
        [--summary output/step2/summary.json] [--port 7860]
"""
import argparse
import json

import gradio as gr

# --- palette (light-mode slice of the project's validated palette) --------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BORDER = "rgba(11,11,11,0.10)"
SERIES_POS = "#2a78d6"   # blue
SERIES_NEG = "#e34948"   # red
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]

CUSTOM_CSS = f"""
.gradio-container {{
    background: {PAGE} !important;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
    /* Force the light palette regardless of the viewer's OS theme -- Gradio's
       default theme otherwise pulls in dark-mode CSS variables that make our
       inline text colors unreadable against a light surface. */
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
    --link-text-color: {SERIES_POS} !important;
}}
.gradio-container, .gradio-container .prose, .gradio-container .prose * {{ color: {TEXT_PRIMARY} !important; }}
#kpi-row {{ gap: 14px; }}
.stat-tile {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 18px 20px; }}
.stat-tile .label {{ font-size: 12px; color: {TEXT_SECONDARY}; margin-bottom: 6px; }}
.stat-tile .value {{ font-size: 28px; font-weight: 700; color: {TEXT_PRIMARY}; letter-spacing: -0.01em; }}
.stat-tile .sub {{ font-size: 12px; color: {TEXT_MUTED}; margin-top: 4px; }}
"""


def load_dataset(records_path: str, summary_path: str) -> dict:
    with open(records_path) as f:
        records = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)
    return {"records": records, "summary": summary}


def imbalance_note(summary: dict) -> str:
    dist = summary["class_distribution_in_batch"]
    total = sum(dist.values())
    parts = [f"{cls} {count} ({count/total:.0%})" for cls, count in dist.items()]
    dominant = max(dist, key=dist.get)
    return (
        f"This batch's rating-derived labels are {', '.join(parts)}. "
        f"With {dominant} dominating, a high overall-accuracy number is easy to reach just by "
        f"getting the majority class right — it is not by itself evidence the model handles the "
        f"minority class well. Balanced sampling across classes arrives in Step 6."
    )


def render_kpis(summary: dict) -> str:
    pos = summary["per_class_accuracy"]["POSITIVE"]
    neg = summary["per_class_accuracy"]["NEGATIVE"]

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.1f}%"

    tiles = [
        ("Reviews scored", str(summary["total_reviews"]),
         f"{summary['unparsed_responses']} unparsed" if summary["unparsed_responses"] else "all parsed"),
        ("Overall accuracy", f"{summary['overall_accuracy'] * 100:.1f}%", "vs. rating-derived label"),
        ("POSITIVE accuracy", pct(pos["accuracy"]), f"support {pos['support']}"),
        ("NEGATIVE accuracy", pct(neg["accuracy"]), f"support {neg['support']}"),
    ]
    cells = "".join(
        f'<div class="stat-tile"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
        for label, value, sub in tiles
    )
    return f'<div id="kpi-row" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));">{cells}</div>'


def render_callout(summary: dict) -> str:
    return (
        f'<div style="display:flex;gap:10px;align-items:flex-start;background:{SURFACE};'
        f'border:1px solid {BORDER};border-left:3px solid {STATUS_CRITICAL};border-radius:8px;'
        f'padding:14px 16px;font-size:13px;color:{TEXT_SECONDARY};line-height:1.55;">'
        f'<span style="color:{STATUS_CRITICAL};font-weight:700;flex-shrink:0;">&#9888;</span>'
        f'<div><strong style="color:{TEXT_PRIMARY};">Imbalance warning.</strong> {imbalance_note(summary)}</div>'
        f'</div>'
    )


def render_confusion(summary: dict) -> str:
    cm = summary["confusion_matrix"]
    classes = list(cm.keys())
    header = "".join(f'<th style="padding:12px 18px;color:{TEXT_MUTED};font-size:11px;'
                      f'text-transform:uppercase;">{c}</th>' for c in classes)
    rows = ""
    for t in classes:
        row = cm[t]
        row_total = sum(row.values()) or 1
        rowlabel = (f'<td style="padding:12px 18px;text-align:right;color:{TEXT_SECONDARY};'
                    f'font-weight:600;">{t}</td>')
        cells = ""
        for p in classes:
            v = row.get(p, 0)
            frac = v / row_total
            idx = min(len(SEQ_STEPS) - 1, int(frac * len(SEQ_STEPS)))
            color = SEQ_STEPS[idx]
            text_color = "#fff" if frac > 0.55 else TEXT_PRIMARY
            cells += (f'<td style="padding:12px 18px;text-align:center;font-weight:700;'
                      f'font-size:16px;border-radius:6px;background:{color};color:{text_color};">{v}</td>')
        rows += f"<tr>{rowlabel}{cells}</tr>"
    return (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;padding:20px 22px;">'
        f'<table style="border-collapse:collapse;font-size:13px;">'
        f'<tr><th></th>{header}</tr>{rows}</table>'
        f'<div style="font-size:12px;color:{TEXT_MUTED};margin-top:10px;max-width:420px;line-height:1.6;">'
        f"Rows = rating-derived ground truth. Columns = model prediction. "
        f"Darker cells hold more reviews (shaded relative to that row's total).</div></div>"
    )


def star_string(rating: float) -> str:
    full = round(rating)
    return "★" * full + "☆" * (5 - full)


def filter_records(records: list[dict], mode: str) -> list[dict]:
    if mode == "Correct only":
        return [r for r in records if r["correct"]]
    if mode == "Incorrect only":
        return [r for r in records if not r["correct"]]
    if mode == "True label: POSITIVE":
        return [r for r in records if r["true_label"] == "POSITIVE"]
    if mode == "True label: NEGATIVE":
        return [r for r in records if r["true_label"] == "NEGATIVE"]
    return records


def render_review_rows(records: list[dict]) -> str:
    rows = ""
    for r in records:
        badge_color = SERIES_POS if r["true_label"] == "POSITIVE" else SERIES_NEG
        pred = r["predicted_label"]
        pred_color = SERIES_POS if pred == "POSITIVE" else (SERIES_NEG if pred == "NEGATIVE" else TEXT_MUTED)
        result_color = STATUS_GOOD if r["correct"] else STATUS_CRITICAL
        result_text = "✓ Correct" if r["correct"] else "✗ Incorrect"
        text = r["text"] or ""
        text = (text[:160].strip() + "…") if len(text) > 160 else text
        rows += (
            f'<tr>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {GRIDLINE};color:{TEXT_MUTED};">{r["index"]}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {GRIDLINE};color:{TEXT_MUTED};'
            f'white-space:nowrap;">{star_string(r["rating"])}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {GRIDLINE};font-weight:600;">{r["title"]}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {GRIDLINE};color:{TEXT_SECONDARY};'
            f'max-width:340px;">{text}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {GRIDLINE};color:{badge_color};'
            f'font-weight:600;white-space:nowrap;">{r["true_label"]}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {GRIDLINE};color:{pred_color};'
            f'font-weight:600;white-space:nowrap;">{pred or "UNPARSED"}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid {GRIDLINE};color:{result_color};'
            f'font-weight:600;white-space:nowrap;">{result_text}</td>'
            f'</tr>'
        )
    header = "".join(
        f'<th style="padding:10px 14px;text-align:left;font-size:11px;text-transform:uppercase;'
        f'color:{TEXT_MUTED};border-bottom:1px solid {GRIDLINE};">{h}</th>'
        for h in ["#", "Rating", "Title", "Text", "True label", "Predicted", "Result"]
    )
    return (
        f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;overflow:hidden;">'
        f'<div style="max-height:520px;overflow:auto;">'
        f'<table style="border-collapse:collapse;width:100%;min-width:760px;font-size:13px;">'
        f'<thead style="position:sticky;top:0;background:{SURFACE};"><tr>{header}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div></div>'
    )


def build_app(records_path: str, summary_path: str) -> gr.Blocks:
    dataset = load_dataset(records_path, summary_path)
    records, summary = dataset["records"], dataset["summary"]

    filter_choices = [
        "All", "Correct only", "Incorrect only",
        "True label: POSITIVE", "True label: NEGATIVE",
    ]

    with gr.Blocks(css=CUSTOM_CSS, title="Gift Card Review Sentiment Dashboard") as demo:
        gr.Markdown("# Gift Card Review Sentiment Dashboard")
        gr.Markdown(
            "LLM sentiment classification vs. star-rating ground truth — "
            "Step 2 data: first 100 reviews, file order (binary POSITIVE/NEGATIVE)."
        )

        gr.HTML(render_kpis(summary))
        gr.HTML(render_callout(summary))
        gr.Markdown("### Confusion matrix")
        gr.HTML(render_confusion(summary))

        gr.Markdown("### Per-review detail")
        with gr.Row():
            filter_dropdown = gr.Dropdown(choices=filter_choices, value="All", label="Filter", scale=1)
            count_label = gr.Markdown(f"Showing **{len(records)}** of **{len(records)}** reviews", elem_id="count-label")
        review_table = gr.HTML(render_review_rows(records))

        def on_filter_change(mode):
            filtered = filter_records(records, mode)
            count_text = f"Showing **{len(filtered)}** of **{len(records)}** reviews"
            return render_review_rows(filtered), count_text

        filter_dropdown.change(
            fn=on_filter_change,
            inputs=filter_dropdown,
            outputs=[review_table, count_label],
        )

        gr.Markdown(
            "---\n"
            "Data: Amazon Reviews '23 “Gift Cards” category (McAuley Lab, UC San Diego). "
            "Sentiment predicted by an LLM from review title/text only — the star rating is used "
            "solely to check the model afterward."
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="output/step2/records.json")
    parser.add_argument("--summary", default="output/step2/summary.json")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_app(args.records, args.summary)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
