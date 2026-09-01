"""
data_loader.py
Reusable functions to load and prepare the Fake/Real news dataset.
"""

import pandas as pd


def load_raw_data(raw_dir="../data/raw"):
    """
    Loads Fake.csv and True.csv from the raw data directory,
    adds a label column to each (1 = real, 0 = fake).
    """
    real_news = pd.read_csv(f"{raw_dir}/True.csv")
    fake_news = pd.read_csv(f"{raw_dir}/Fake.csv")

    real_news["label"] = 1
    fake_news["label"] = 0

    return real_news, fake_news


def merge_and_shuffle(real_news, fake_news, random_state=42):
    """
    Merges real and fake dataframes, shuffles the combined dataset,
    and resets the index.
    """
    merged = pd.concat([real_news, fake_news], ignore_index=True)
    merged = merged.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return merged


def drop_leaky_columns(df, columns=("subject", "date")):
    """
    Drops columns known to leak the label (e.g. subject, date).
    """
    return df.drop(columns=[c for c in columns if c in df.columns])


def remove_near_empty_rows(df, min_word_count=10, text_col="text"):
    """
    Removes rows where the text column has fewer than min_word_count words.
    """
    word_counts = df[text_col].apply(lambda x: len(str(x).split()))
    return df[word_counts >= min_word_count].reset_index(drop=True)


def load_clean_dataset(raw_dir="../data/raw"):
    """
    Full pipeline: load raw CSVs -> label -> merge -> shuffle ->
    drop leaky columns -> remove near-empty rows.
    Returns the cleaned, merged dataframe.
    """
    real_news, fake_news = load_raw_data(raw_dir)
    merged = merge_and_shuffle(real_news, fake_news)
    merged = drop_leaky_columns(merged)
    merged = remove_near_empty_rows(merged)
    return merged


def load_processed_dataset(path="../data/processed/merged_news.csv"):
    """
    Loads the already-processed/saved dataset directly (skips raw pipeline).
    """
    return pd.read_csv(path)