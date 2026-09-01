"""
text_cleaning.py
Text cleaning utilities for the Fake News Detector project.

Handles:
- Dataset leakage removal (Reuters dateline prefix)
- Heavy cleaning for TF-IDF / classical ML models
- Light cleaning for DistilBERT (uncased) fine-tuning
"""

import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

# Loaded once at import time for efficiency
STOP_WORDS = set(stopwords.words("english"))


def strip_reuters_prefix(text):
    """
    Removes the leading Reuters dateline pattern (e.g. "WASHINGTON (Reuters) - ")
    from the start of real-news articles. This is a dataset collection artifact,
    not a genuine signal of real vs fake news, so it must be removed before modeling.

    Args:
        text (str): The raw input text.

    Returns:
        str: Text with the Reuters dateline prefix removed, if present.
    """
    text = re.sub(r"^[A-Za-z\s/,.]*\(Reuters\)\s*-\s*", "", text)
    return text


def clean_text_heavy(text):
    """
    Cleans text for TF-IDF / classical ML models: lowercase, strip punctuation
    and digits, remove stopwords, collapse whitespace.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The heavily cleaned text.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    cleaned_text = " ".join(words)

    return cleaned_text


def clean_text_light(text):
    """
    Cleans text for DistilBERT (uncased) fine-tuning: lowercase and whitespace
    normalization only. Punctuation, stopwords, and numbers are preserved since
    the transformer relies on full sentence structure.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The lightly cleaned text.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess(text, mode="heavy"):
    """
    Full preprocessing pipeline: strips the Reuters dateline leak first,
    then applies either heavy (TF-IDF) or light (DistilBERT) cleaning.

    Args:
        text (str): The raw input text.
        mode (str): "heavy" for TF-IDF/classical ML, "light" for DistilBERT.

    Returns:
        str: The fully preprocessed text.
    """
    text = str(text)
    text = strip_reuters_prefix(text)

    if mode == "heavy":
        return clean_text_heavy(text)
    elif mode == "light":
        return clean_text_light(text)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'heavy' or 'light'.")