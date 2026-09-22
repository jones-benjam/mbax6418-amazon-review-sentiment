"""
Step 1: A structured, reusable prompt that classifies an Amazon review's
title + text as POSITIVE or NEGATIVE using an LLM.

Step 5 (Part A): the same call also asks for the review's primary emotion,
using the 8 NRC emotion categories, so it can be compared against the
word-list-derived emotion in emotion_wordlist.py.

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

# Same 8 categories as the NRC Emotion Lexicon, so the LLM's answer and the
# word-list answer (emotion_wordlist.py) are directly comparable.
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

SYSTEM_PROMPT = """You are a strict sentiment and emotion classifier for Amazon product reviews.

Given a review's title and text, do two things:

1. Decide whether the reviewer's overall SENTIMENT toward the product/experience
   is POSITIVE or NEGATIVE.
2. Pick the single primary EMOTION the reviewer is expressing, from exactly this
   list: anger, anticipation, disgust, fear, joy, sadness, surprise, trust.

Rules for sentiment:
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

Rules for emotion:
- Pick exactly one emotion from the list above, in lowercase, even if the review
  blends several -- choose the strongest or most central one.
- Base it only on the words in the title and text, never on a rating.
- Praise, a good gift, fast/easy delivery: usually "joy", or "trust" when the
  review is about relying on the product/brand working as expected.
- An unexpectedly good (or bad) twist: "surprise".
- Looking forward to giving/using the gift: "anticipation".
- Complaints and failures: "anger" (frustration, feeling cheated), "sadness"
  (disappointment, letdown), "disgust" (feeling scammed or grossed out), or
  "fear" (worry -- e.g. about fraud, a card not working, losing money).
- If nothing above clearly fits, fall back to the sentiment: "joy" for a
  POSITIVE review, "anger" for a NEGATIVE one.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{"sentiment": "POSITIVE", "emotion": "joy"}
"""

USER_TEMPLATE = """Title: {title}
Text: {text}"""


def classify_review(title: str, text: str) -> dict:
    """Classify one review's sentiment + primary emotion. Returns dict with
    'sentiment', 'emotion', and 'raw' (the raw model output, for debugging
    parse failures)."""
    title = (title or "").strip()
    text = (text or "").strip()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(title=title, text=text)},
        ],
        temperature=0,
        # Measured actual usage tops out around 22 tokens for this schema, but
        # that only left an ~8-token margin at max_tokens=30 -- too tight to
        # trust across a much larger, more varied sample (Step 6). A truncated
        # response silently loses the emotion field (or worse, breaks JSON
        # parsing entirely) since finish_reason='length' isn't checked here.
        max_tokens=60,
        # The backing model (Qwen3) is a reasoning model that otherwise burns
        # the token budget on a hidden "thinking" trace before ever writing
        # the JSON answer, leaving content=None. We don't need chain-of-thought
        # for this classification, so we turn it off.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = (response.choices[0].message.content or "").strip()

    sentiment, emotion = _parse_response(raw)
    return {"sentiment": sentiment, "emotion": emotion, "raw": raw}


def _parse_response(raw: str) -> tuple[str | None, str | None]:
    """Pull sentiment + emotion out of the model's reply. Tries strict JSON
    first, falls back to substring search so minor formatting slips (stray
    text, markdown fences) don't silently break scoring."""
    try:
        parsed = json.loads(raw)
        sentiment = str(parsed.get("sentiment", "")).strip().upper()
        emotion = str(parsed.get("emotion", "")).strip().lower()
        sentiment = sentiment if sentiment in ("POSITIVE", "NEGATIVE") else None
        emotion = emotion if emotion in EMOTIONS else None
        if sentiment is not None or emotion is not None:
            return sentiment, emotion
    except (json.JSONDecodeError, AttributeError):
        pass

    upper = raw.upper()
    has_pos = "POSITIVE" in upper
    has_neg = "NEGATIVE" in upper
    sentiment = "POSITIVE" if (has_pos and not has_neg) else ("NEGATIVE" if (has_neg and not has_pos) else None)

    lower = raw.lower()
    found = [e for e in EMOTIONS if e in lower]
    emotion = found[0] if len(found) == 1 else None

    return sentiment, emotion


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
              f"emotion={str(result['emotion']):<12} title={case['title']!r}")
        if result["sentiment"] is None or result["emotion"] is None:
            print(f"     (unparsed raw output: {result['raw']!r})")

    print(f"\n{correct}/{len(spot_checks)} spot checks correct")
