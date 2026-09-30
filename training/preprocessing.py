"""Canonical text preprocessing shared by training and production inference."""

import re


def preprocess_text(text: str) -> str:
    """Normalize text using the exact production inference rules."""
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
