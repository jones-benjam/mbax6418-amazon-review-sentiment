"""
Reusable LLM classifier for Amazon reviews (Steps 1, 5 and 6).

classify_review(title, text, mode) asks the model, in ONE call, for the
review's sentiment and its primary emotion (one of the 8 NRC categories).

  mode="binary"  -> POSITIVE / NEGATIVE           (Steps 1-5)
  mode="three"   -> POSITIVE / NEUTRAL / NEGATIVE (Step 6)

The prompts live in prompts/*.txt. The model is only ever given the review's
title and text -- never the star rating.
"""
import json
import os
import re
from pathlib import Path

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

CLASSES = {
    "binary": ["POSITIVE", "NEGATIVE"],
    "three": ["POSITIVE", "NEUTRAL", "NEGATIVE"],
}

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_PROMPTS = {
    "binary": (_PROMPT_DIR / "sentiment_emotion_binary.txt").read_text().strip(),
    "three": (_PROMPT_DIR / "sentiment_emotion_three_class.txt").read_text().strip(),
}

USER_TEMPLATE = """Title: {title}
Text: {text}"""

# Amazon auto-fills the title with the star count ("Three Stars") when a reviewer
# skips it. That is the rating itself, in words -- ~21% of this dataset, and the
# number matches the true rating 99.9% of the time. Callers must pass such
# titles through mask_star_title() so the model never sees the rating.
_STAR_TITLE = re.compile(r"^\s*(one|two|three|four|five|[1-5])[\s-]*stars?[.!]*\s*$", re.IGNORECASE)


def is_star_title(title: str) -> bool:
    return bool(_STAR_TITLE.match(title or ""))


def mask_star_title(title: str) -> str:
    """Blank out a title that only states the star count; leave real titles alone."""
    return "" if is_star_title(title) else (title or "")

# Fixed request settings -- recorded in every run's summary.json so a result
# can be reproduced. temperature=0 plus a pinned seed keeps the endpoint's
# output stable from run to run.
SETTINGS = {
    "temperature": 0,
    "seed": 0,
    # Measured usage for this schema tops out around 22 tokens; 60 leaves a
    # wide margin. A truncated reply (finish_reason == "length") is recorded.
    "max_tokens": 60,
    # The backing model (Qwen3) is a reasoning model that otherwise burns the
    # token budget on a hidden "thinking" trace and returns content=None.
    "enable_thinking": False,
}


def classify_review(title: str, text: str, mode: str = "three") -> dict:
    """Classify one review. Returns a dict with 'sentiment', 'emotion',
    'raw' (raw model output), 'finish_reason', and 'error' (None unless the
    API call itself failed, in which case sentiment/emotion are None)."""
    classes = CLASSES[mode]
    title = (title or "").strip()
    text = (text or "").strip()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[mode]},
                {"role": "user", "content": USER_TEMPLATE.format(title=title or "(none)", text=text)},
            ],
            temperature=SETTINGS["temperature"],
            seed=SETTINGS["seed"],
            max_tokens=SETTINGS["max_tokens"],
            extra_body={"chat_template_kwargs": {"enable_thinking": SETTINGS["enable_thinking"]}},
        )
    except Exception as exc:  # network/server failure: record it, don't kill the whole run
        return {"sentiment": None, "emotion": None, "raw": "", "finish_reason": "error",
                "error": f"{type(exc).__name__}: {exc}"}

    choice = response.choices[0]
    raw = (choice.message.content or "").strip()
    sentiment, emotion = _parse_response(raw, classes)
    return {"sentiment": sentiment, "emotion": emotion, "raw": raw,
            "finish_reason": choice.finish_reason, "error": None}


def _parse_response(raw: str, classes: list) -> tuple:
    """Pull sentiment + emotion out of the model's reply. Tries strict JSON
    first, falls back to substring search so minor formatting slips (stray
    text, markdown fences) don't silently break scoring."""
    try:
        parsed = json.loads(raw)
        sentiment = str(parsed.get("sentiment", "")).strip().upper()
        emotion = str(parsed.get("emotion", "")).strip().lower()
        sentiment = sentiment if sentiment in classes else None
        emotion = emotion if emotion in EMOTIONS else None
        if sentiment is not None or emotion is not None:
            return sentiment, emotion
    except (json.JSONDecodeError, AttributeError):
        pass

    upper = raw.upper()
    hits = [c for c in classes if c in upper]
    sentiment = hits[0] if len(hits) == 1 else None

    lower = raw.lower()
    found = [e for e in EMOTIONS if e in lower]
    emotion = found[0] if len(found) == 1 else None

    return sentiment, emotion


if __name__ == "__main__":
    spot_checks = {
        "binary": [
            ("Terrible experience", "The gift card never arrived and support ignored three emails. Waste of money.", "NEGATIVE"),
            ("Perfect every time", "Always the easiest gift to send. Instant delivery, no issues at all.", "POSITIVE"),
            ("Five stars, sure", "Great, ANOTHER gift card that showed up a week late and after the birthday. Real classy.", "NEGATIVE"),
            ("fine", "it's a gift card. it worked.", "POSITIVE"),
            ("DO NOT BUY", "Card was already redeemed by someone else when I tried to use it. Scam.", "NEGATIVE"),
        ],
        "three": [
            ("Terrible experience", "The gift card never arrived and support ignored three emails. Waste of money.", "NEGATIVE"),
            ("Perfect every time", "Always the easiest gift to send. Instant delivery, no issues at all.", "POSITIVE"),
            ("It's okay", "Arrived on time but the box was a bit crushed and the design was plain. Does the job, nothing special.", "NEUTRAL"),
            ("Mixed", "Loved the tin, but the card took a week to arrive and one was missing its PIN. Half good, half bad.", "NEUTRAL"),
        ],
    }

    print(f"Model: {MODEL}\nSettings: {SETTINGS}\n")
    for mode, cases in spot_checks.items():
        correct = 0
        print(f"--- mode={mode} ---")
        for title, text, expected in cases:
            result = classify_review(title, text, mode)
            ok = result["sentiment"] == expected
            correct += ok
            print(f"{'OK ' if ok else 'FAIL'} expected={expected:<8} got={str(result['sentiment']):<8} "
                  f"emotion={str(result['emotion']):<12} title={title!r}")
            if result["sentiment"] is None or result["emotion"] is None:
                print(f"     (raw output: {result['raw']!r}, error: {result['error']})")
        print(f"{correct}/{len(cases)} spot checks correct\n")
