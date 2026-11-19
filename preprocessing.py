"""
Custom preprocessing for the CineSense sentiment model.

These functions live in their own module so that the pickled (joblib) pipeline
can be loaded outside Jupyter. ``mark_negation`` is copied unchanged from the
notebook; ``clean_text`` reproduces the notebook's cleaning steps
(<br> removal, lowercasing, URL removal, whitespace normalisation) so the saved
pipeline accepts RAW review text.
"""
import re

NEG_WORDS = {"not", "no", "never", "none", "nobody", "nothing", "neither",
             "nor", "cannot", "hardly", "barely", "without"}


def mark_negation(text: str, window: int = 3) -> str:
    """'not very good' -> 'not NOT_very NOT_good' (until the next punctuation mark)."""
    tokens = re.findall(r"[a-z']+|[.,;:!?]", text.lower())
    out, left = [], 0
    for t in tokens:
        if t in ".,;:!?":
            left = 0
            continue
        if t in NEG_WORDS or t.endswith("n't"):
            out.append(t)
            left = window
        elif left > 0:
            out.append("NOT_" + t)
            left -= 1
        else:
            out.append(t)
    return " ".join(out)


def clean_text(text: str) -> str:
    """Same cleaning steps as the notebook's `clean_review` column."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def full_preprocess(text: str) -> str:
    """Notebook cleaning followed by negation marking (used as the TF-IDF preprocessor)."""
    return mark_negation(clean_text(text))
