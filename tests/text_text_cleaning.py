"""
test_text_cleaning.py
Unit tests for src/text_cleaning.py using pytest.
"""

import sys
import os

# allow imports from src/ when running pytest from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.text_cleaning import (
    strip_reuters_prefix,
    clean_text_heavy,
    clean_text_light,
    preprocess,
)


# ---------- strip_reuters_prefix ----------

def test_strip_reuters_prefix_removes_dateline_with_location():
    input_text = "WASHINGTON (Reuters) - The president said today..."
    result = strip_reuters_prefix(input_text)
    assert result == "The president said today..."


def test_strip_reuters_prefix_removes_dateline_without_location():
    input_text = "(Reuters) - President Donald Trump's team said..."
    result = strip_reuters_prefix(input_text)
    assert result == "President Donald Trump's team said..."


def test_strip_reuters_prefix_leaves_normal_text_unchanged():
    input_text = "This is a normal sentence with no dateline."
    result = strip_reuters_prefix(input_text)
    assert result == input_text


# ---------- clean_text_heavy ----------

def test_clean_text_heavy_lowercases_and_removes_punctuation():
    input_text = "Hello, World! This is GREAT."
    result = clean_text_heavy(input_text)
    assert "," not in result
    assert "!" not in result
    assert result == result.lower()


def test_clean_text_heavy_removes_stopwords():
    input_text = "this is a test of the stopword removal"
    result = clean_text_heavy(input_text)
    assert "is" not in result.split()
    assert "the" not in result.split()
    assert "of" not in result.split()


def test_clean_text_heavy_removes_digits():
    input_text = "Trump won in 2020 election"
    result = clean_text_heavy(input_text)
    assert "2020" not in result


# ---------- clean_text_light ----------

def test_clean_text_light_collapses_whitespace():
    input_text = "  Hello   World  "
    result = clean_text_light(input_text)
    assert result == "hello world"


def test_clean_text_light_lowercases():
    input_text = "Hello World"
    result = clean_text_light(input_text)
    assert result == "hello world"


def test_clean_text_light_keeps_punctuation():
    input_text = "Hello, World!"
    result = clean_text_light(input_text)
    assert "," in result
    assert "!" in result


# ---------- preprocess ----------

def test_preprocess_heavy_and_light_differ():
    input_text = "WASHINGTON (Reuters) - The President said, 'This is great!' in 2020."
    heavy_result = preprocess(input_text, mode="heavy")
    light_result = preprocess(input_text, mode="light")
    assert heavy_result != light_result


def test_preprocess_invalid_mode_raises_error():
    try:
        preprocess("some text", mode="medium")
        assert False, "Expected ValueError but none was raised"
    except ValueError:
        pass