"""Canonical text preprocessing shared by training and production inference."""

import re


def preprocess_text(text: str) -> str:
    """Normalize text using the exact production inference rules."""
    text = re.sub(r"\s+", " ", text).strip()
    return text
