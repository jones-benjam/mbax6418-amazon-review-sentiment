"""
Dashboard generator (Step 3, extended in Steps 4-7).

Reads the saved scoring output (records.json + summary.json) and renders a
single self-contained HTML file: dashboard/index.html. No server, no network
calls at view time -- all data is baked in as an embedded JSON blob and
rendered client-side with vanilla JS, so later steps (filtering, emotions,
three-class, descriptive charts) can extend the same file by feeding it
richer data rather than starting over.

Usage:
    python src/generate_dashboard.py --records output/step2/records.json \
        --summary output/step2/summary.json --out dashboard/index.html
"""
import argparse
import json
import os

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gift Card Review Sentiment Dashboard</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page-plane:     #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --gridline:       #e1e0d9;
    --baseline:       #c3c2b7;
    --border:         rgba(11,11,11,0.10);
    --series-1:       #2a78d6;   /* blue - sequential / positive class */
    --series-8:       #e34948;   /* red - negative class */
    --status-good:      #0ca30c;
    --status-critical:  #d03b3b;
    --seq-100: #cde2fb; --seq-200: #9ec5f4; --seq-300: #6da7ec;
    --seq-400: #3987e5; --seq-450: #2a78d6; --seq-500: #256abf;
    --seq-600: #184f95; --seq-700: #0d366b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page-plane:     #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --gridline:       #2c2c2a;
      --baseline:       #383835;
      --border:         rgba(255,255,255,0.10);
      --series-1:       #3987e5;
      --series-8:       #e66767;
      --status-good:      #0ca30c;
      --status-critical:  #e66767;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page-plane:     #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --series-1:       #3987e5;
    --series-8:       #e66767;
    --status-good:      #0ca30c;
    --status-critical:  #e66767;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--page-plane);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1120px; margin: 0 auto; padding: 40px 24px 80px; }}

  header.page-header {{ margin-bottom: 32px; }}
  header.page-header h1 {{ font-size: 26px; font-weight: 700; margin: 0 0 6px; letter-spacing: -0.01em; }}
  header.page-header p {{ margin: 0; color: var(--text-secondary); font-size: 14px; line-height: 1.5; max-width: 720px; }}

  .card {{
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
  }}
  section {{ margin-bottom: 28px; }}
  section > h2 {{ font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--text-muted); margin: 0 0 12px; }}

  .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
  .stat-tile .label {{ font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; }}
  .stat-tile .value {{ font-size: 30px; font-weight: 700; letter-spacing: -0.01em; }}
  .stat-tile .sub {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}

  .callout {{
    display: flex; gap: 10px; align-items: flex-start;
    background: var(--surface-1); border: 1px solid var(--border); border-left: 3px solid var(--status-critical);
    border-radius: 8px; padding: 14px 16px; font-size: 13px; color: var(--text-secondary); line-height: 1.55;
  }}
  .callout .icon {{ color: var(--status-critical); font-weight: 700; flex-shrink: 0; }}
  .callout strong {{ color: var(--text-primary); }}

  .matrix-wrap {{ display: flex; gap: 32px; flex-wrap: wrap; align-items: flex-start; }}
  table.confusion {{ border-collapse: collapse; font-size: 13px; }}
  table.confusion th, table.confusion td {{ padding: 12px 18px; text-align: center; }}
  table.confusion th {{ color: var(--text-muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em; }}
  table.confusion th.corner {{ visibility: hidden; }}
  table.confusion td.cell {{ font-weight: 700; font-size: 16px; border-radius: 6px; color: var(--text-primary); }}
  table.confusion td.rowlabel {{ text-align: right; color: var(--text-secondary); font-weight: 600; }}
  .matrix-legend {{ font-size: 12px; color: var(--text-muted); max-width: 240px; line-height: 1.6; }}

  .table-card {{ padding: 0; overflow: hidden; }}
  .table-scroll {{ max-height: 560px; overflow-y: auto; overflow-x: auto; }}
  table.reviews {{ border-collapse: collapse; width: 100%; min-width: 760px; font-size: 13px; }}
  table.reviews thead th {{
    position: sticky; top: 0; background: var(--surface-1); z-index: 1;
    text-align: left; padding: 10px 14px; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.03em; color: var(--text-muted); border-bottom: 1px solid var(--gridline);
  }}
  table.reviews tbody td {{ padding: 10px 14px; border-bottom: 1px solid var(--gridline); vertical-align: top; }}
  table.reviews tbody tr:last-child td {{ border-bottom: none; }}
  .rating {{ color: var(--text-muted); white-space: nowrap; font-variant-numeric: tabular-nums; }}
  .review-title {{ font-weight: 600; color: var(--text-primary); }}
  .review-text {{ color: var(--text-secondary); max-width: 360px; }}
  .badge {{ display: inline-flex; align-items: center; gap: 5px; font-size: 12px; font-weight: 600; white-space: nowrap; }}
  .badge.pos {{ color: var(--series-1); }}
  .badge.neg {{ color: var(--series-8); }}
  .result {{ display: inline-flex; align-items: center; gap: 5px; font-size: 12px; font-weight: 600; white-space: nowrap; }}
  .result.correct {{ color: var(--status-good); }}
  .result.incorrect {{ color: var(--status-critical); }}

  footer {{ margin-top: 40px; font-size: 12px; color: var(--text-muted); line-height: 1.6; }}
  footer a {{ color: var(--text-secondary); }}
</style>
</head>
<body>
<div class="wrap">
  <header class="page-header">
    <h1>Gift Card Review Sentiment Dashboard</h1>
    <p>{subtitle}</p>
  </header>

  <section>
    <h2>Headline numbers</h2>
    <div class="kpi-row" id="kpi-row"></div>
  </section>

  <section>
    <div class="callout">
      <span class="icon">&#9888;</span>
      <div><strong>Imbalance warning.</strong> {imbalance_note}</div>
    </div>
  </section>

  <section>
    <h2>Confusion matrix</h2>
    <div class="card matrix-wrap">
      <table class="confusion" id="confusion-table"></table>
      <div class="matrix-legend">Rows = rating-derived ground truth. Columns = model prediction. Darker cells hold more reviews (shaded relative to that row's total).</div>
    </div>
  </section>

  <section>
    <h2>Per-review detail ({review_count} reviews)</h2>
    <div class="card table-card">
      <div class="table-scroll">
        <table class="reviews">
          <thead>
            <tr><th>#</th><th>Rating</th><th>Title</th><th>Text</th><th>True label</th><th>Predicted</th><th>Result</th></tr>
          </thead>
          <tbody id="review-body"></tbody>
        </table>
      </div>
    </div>
  </section>

  <footer>
    Data: Amazon Reviews&nbsp;'23 &ldquo;Gift Cards&rdquo; category (McAuley Lab, UC San Diego).
    Sentiment predicted by an LLM from review title/text only &mdash; the star rating is used solely to check the model afterward.
  </footer>
</div>

<script id="app-data" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('app-data').textContent);

function starString(rating) {{
  const full = Math.round(rating);
  return '&#9733;'.repeat(full) + '&#9734;'.repeat(5 - full);
}}

function escapeHtml(s) {{
  return (s || '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

function truncate(s, n) {{
  if (!s) return '';
  return s.length > n ? s.slice(0, n).trim() + '\\u2026' : s;
}}

function renderKpis() {{
  const s = DATA.summary;
  const tiles = [
    {{ label: 'Reviews scored', value: s.total_reviews, sub: s.unparsed_responses ? `${{s.unparsed_responses}} unparsed` : 'all parsed' }},
    {{ label: 'Overall accuracy', value: (s.overall_accuracy * 100).toFixed(1) + '%', sub: 'vs. rating-derived label' }},
    {{ label: 'POSITIVE accuracy', value: fmtPct(s.per_class_accuracy.POSITIVE.accuracy), sub: `support ${{s.per_class_accuracy.POSITIVE.support}}` }},
    {{ label: 'NEGATIVE accuracy', value: fmtPct(s.per_class_accuracy.NEGATIVE.accuracy), sub: `support ${{s.per_class_accuracy.NEGATIVE.support}}` }},
  ];
  document.getElementById('kpi-row').innerHTML = tiles.map(t => `
    <div class="card stat-tile">
      <div class="label">${{t.label}}</div>
      <div class="value">${{t.value}}</div>
      <div class="sub">${{t.sub}}</div>
    </div>`).join('');
}}
function fmtPct(v) {{ return v === null || v === undefined ? 'n/a' : (v * 100).toFixed(1) + '%'; }}

function seqColor(frac) {{
  const steps = ['--seq-100','--seq-200','--seq-300','--seq-400','--seq-450','--seq-500','--seq-600','--seq-700'];
  const idx = Math.min(steps.length - 1, Math.floor(frac * steps.length));
  return `var(${{steps[idx]}})`;
}}

function renderConfusion() {{
  const cm = DATA.summary.confusion_matrix;
  const classes = Object.keys(cm);
  let html = '<tr><th class="corner">true \\\\ pred</th>' + classes.map(c => `<th>${{c}}</th>`).join('') + '</tr>';
  classes.forEach(t => {{
    const row = cm[t];
    const rowTotal = Object.values(row).reduce((a,b) => a+b, 0) || 1;
    html += `<tr><td class="rowlabel">${{t}}</td>`;
    classes.forEach(p => {{
      const v = row[p] || 0;
      const frac = v / rowTotal;
      const textColor = frac > 0.55 ? '#fff' : 'var(--text-primary)';
      html += `<td class="cell" style="background:${{seqColor(frac)}}; color:${{textColor}}">${{v}}</td>`;
    }});
    html += '</tr>';
  }});
  document.getElementById('confusion-table').innerHTML = html;
}}

function renderReviews() {{
  const rows = DATA.records.map(r => {{
    const badgeClass = r.true_label === 'POSITIVE' ? 'pos' : 'neg';
    const predClass = r.predicted_label === 'POSITIVE' ? 'pos' : (r.predicted_label === 'NEGATIVE' ? 'neg' : '');
    const resultClass = r.correct ? 'correct' : 'incorrect';
    const resultIcon = r.correct ? '&#10003;' : '&#10007;';
    const resultLabel = r.correct ? 'Correct' : 'Incorrect';
    return `<tr>
      <td>${{r.index}}</td>
      <td class="rating">${{starString(r.rating)}}</td>
      <td class="review-title">${{escapeHtml(r.title)}}</td>
      <td class="review-text">${{escapeHtml(truncate(r.text, 160))}}</td>
      <td><span class="badge ${{badgeClass}}">${{r.true_label}}</span></td>
      <td><span class="badge ${{predClass}}">${{r.predicted_label ?? 'UNPARSED'}}</span></td>
      <td><span class="result ${{resultClass}}">${{resultIcon}} ${{resultLabel}}</span></td>
    </tr>`;
  }}).join('');
  document.getElementById('review-body').innerHTML = rows;
}}

renderKpis();
renderConfusion();
renderReviews();
</script>
</body>
</html>
"""


def build_imbalance_note(summary: dict) -> str:
    dist = summary["class_distribution_in_batch"]
    total = sum(dist.values())
    parts = [f"{cls} {count} ({count/total:.0%})" for cls, count in dist.items()]
    dominant = max(dist, key=dist.get)
    return (
        f"This batch's rating-derived labels are {', '.join(parts)}. "
        f"With {dominant} dominating, a high overall-accuracy number is easy to reach just by "
        f"getting the majority class right -- it is not by itself evidence the model handles the "
        f"minority class well. Balanced sampling across classes arrives in Step 6."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="output/step2/records.json")
    parser.add_argument("--summary", default="output/step2/summary.json")
    parser.add_argument("--out", default="dashboard/index.html")
    parser.add_argument("--subtitle", default=(
        "LLM sentiment classification vs. star-rating ground truth &mdash; "
        "Step 2: first 100 reviews, file order (binary POSITIVE/NEGATIVE)."
    ))
    args = parser.parse_args()

    with open(args.records) as f:
        records = json.load(f)
    with open(args.summary) as f:
        summary = json.load(f)

    data_json = json.dumps({"records": records, "summary": summary})
    imbalance_note = build_imbalance_note(summary)

    html = TEMPLATE.format(
        subtitle=args.subtitle,
        imbalance_note=imbalance_note,
        review_count=len(records),
        data_json=data_json,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"Wrote {args.out} ({len(html):,} bytes, {len(records)} reviews embedded)")


if __name__ == "__main__":
    main()
