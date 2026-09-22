"""
Step 5 (Part B): word-list-derived primary emotion.

Scores a review's words against the NRC Word-Emotion Association Lexicon
(EmoLex) -- no model calls, runs over already-scored reviews. For each of
the 8 NRC emotions, count how many of the review's words are associated
with that emotion; the emotion with the highest count wins.

Lexicon source: the NRC Emotion Lexicon, Saif Mohammad, NRC Canada.
http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
Obtained via the `nrclex` PyPI package, which bundles the lexicon as JSON
(word -> list of associated emotions, including "positive"/"negative" tags
we ignore here). We use only that raw word->emotion data; the scoring below
is our own, not nrclex's built-in TextBlob-based scorer.
"""
import json
import re
from importlib import resources

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

_WORD_RE = re.compile(r"[a-z']+")


def _load_lexicon() -> dict:
    import nrclex
    data_path = resources.files(nrclex).joinpath("data", "nrc_en.json")
    with data_path.open() as f:
        return json.load(f)


LEXICON = _load_lexicon()


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def emotion_scores(title: str, text: str) -> dict:
    """Word-hit counts per NRC emotion for a review's title+text combined."""
    words = tokenize(title) + tokenize(text)
    scores = {e: 0 for e in EMOTIONS}
    for w in words:
        for e in LEXICON.get(w, []):
            if e in scores:
                scores[e] += 1
    return scores


def primary_emotion(title: str, text: str) -> dict:
    """Top-scoring emotion (None if no lexicon words matched at all), plus
    the full score breakdown. Ties resolve to the first emotion (in the
    fixed EMOTIONS order) that reaches the max count, so results are
    deterministic."""
    scores = emotion_scores(title, text)
    total = sum(scores.values())
    top = max(scores, key=scores.get) if total > 0 else None
    return {"emotion": top, "scores": scores}


if __name__ == "__main__":
    samples = [
        ("Terrible experience", "The gift card never arrived and support ignored three emails. Waste of money."),
        ("Perfect every time", "Always the easiest gift to send. Instant delivery, no issues at all."),
        ("DO NOT BUY", "Card was already redeemed by someone else when I tried to use it. Scam."),
        ("Good product", "Good product"),
        ("fine", "it's a gift card. it worked."),
    ]
    for title, text in samples:
        result = primary_emotion(title, text)
        print(f"{title!r} / {text!r}\n  -> emotion={result['emotion']}  scores={result['scores']}\n")
