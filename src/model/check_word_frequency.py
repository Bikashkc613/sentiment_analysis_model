from pathlib import Path
from collections import Counter

import pandas as pd

from src.preprocessing.clean_text import clean_text


# Project folder
project_folder = Path(__file__).resolve().parents[2]

# Load training dataset
train_path = project_folder / "data" / "interim" / "train_3class.csv"

df = pd.read_csv(train_path)

# Count words in training reviews
word_counts = Counter()

for review in df["review"]:
    cleaned_review = clean_text(review)
    word_counts.update(cleaned_review.split())


# Words to investigate
words = [
    "hated",
    "hate",
    "terrible",
    "boring",
    "wonderful",
    "amazing",
    "loved"
]

print("\nWORD FREQUENCY IN TRAINING DATA")

for word in words:
    print(f"{word}: {word_counts[word]}")