"""
Step 1: A structured, reusable prompt that classifies an Amazon review's
title + text as POSITIVE or NEGATIVE using an LLM.

The model is never shown the star rating -- classification must be based
only on the review's own language.
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)
MODEL = os.environ["OPENAI_MODEL"]

SYSTEM_PROMPT = """You are a strict sentiment classifier for Amazon product reviews.

Given a review's title and text, decide whether the reviewer's overall sentiment
toward the product/experience is POSITIVE or NEGATIVE.

Rules:
- Base your judgment only on the words in the title and text. You are never given
  a star rating, and you must not guess or reason about one.
- If the title and text seem to disagree (e.g. a sarcastic or misleading title),
  weight the body text more heavily -- it usually carries the real signal.
- Short, terse, or angry reviews still get a clear answer: judge the tone and
  content that is present, don't default to NEGATIVE just because a review is
  short, and don't default to POSITIVE just because it's polite.
- Sarcasm and backhanded compliments count as NEGATIVE if the underlying
  complaint is negative, even if individual words sound positive.
- There is no NEUTRAL option at this stage. If the review is mixed, pick the
  side the reviewer leans toward more strongly.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{"sentiment": "POSITIVE"} or {"sentiment": "NEGATIVE"}
"""

USER_TEMPLATE = """Title: {title}
Text: {text}"""


def classify_review(title: str, text: str) -> dict:
    """Classify one review's sentiment. Returns dict with 'sentiment' and
    'raw' (the raw model output, for debugging parse failures)."""
    title = (title or "").strip()
    text = (text or "").strip()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(title=title, text=text)},
        ],
        temperature=0,
        max_tokens=20,
        # The backing model (Qwen3) is a reasoning model that otherwise burns
        # the token budget on a hidden "thinking" trace before ever writing
        # the JSON answer, leaving content=None. We don't need chain-of-thought
        # for a two-way classification, so we turn it off.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = (response.choices[0].message.content or "").strip()

    sentiment = _parse_sentiment(raw)
    return {"sentiment": sentiment, "raw": raw}


def _parse_sentiment(raw: str) -> str | None:
    """Pull POSITIVE/NEGATIVE out of the model's reply. Tries strict JSON
    first, falls back to a substring search so minor formatting slips
    (stray text, markdown fences) don't silently break scoring."""
    try:
        parsed = json.loads(raw)
        value = str(parsed.get("sentiment", "")).strip().upper()
        if value in ("POSITIVE", "NEGATIVE"):
            return value
    except (json.JSONDecodeError, AttributeError):
        pass

    upper = raw.upper()
    has_pos = "POSITIVE" in upper
    has_neg = "NEGATIVE" in upper
    if has_pos and not has_neg:
        return "POSITIVE"
    if has_neg and not has_pos:
        return "NEGATIVE"
    return None


if __name__ == "__main__":
    spot_checks = [
        {
            "title": "Terrible experience",
            "text": "The gift card never arrived and support ignored three emails. Waste of money.",
            "expected": "NEGATIVE",
        },
        {
            "title": "Perfect every time",
            "text": "Always the easiest gift to send. Instant delivery, no issues at all.",
            "expected": "POSITIVE",
        },
        {
            "title": "Five stars, sure",
            "text": "Great, ANOTHER gift card that showed up a week late and after the birthday. Real classy.",
            "expected": "NEGATIVE",
        },
        {
            "title": "fine",
            "text": "it's a gift card. it worked.",
            "expected": "POSITIVE",
        },
        {
            "title": "DO NOT BUY",
            "text": "Card was already redeemed by someone else when I tried to use it. Scam.",
            "expected": "NEGATIVE",
        },
    ]

    print(f"Model: {MODEL}\n")
    correct = 0
    for case in spot_checks:
        result = classify_review(case["title"], case["text"])
        ok = result["sentiment"] == case["expected"]
        correct += ok
        print(f"{'OK ' if ok else 'FAIL'} expected={case['expected']:<8} got={str(result['sentiment']):<8} "
              f"title={case['title']!r}")
        if result["sentiment"] is None:
            print(f"     (unparsed raw output: {result['raw']!r})")

    print(f"\n{correct}/{len(spot_checks)} spot checks correct")
