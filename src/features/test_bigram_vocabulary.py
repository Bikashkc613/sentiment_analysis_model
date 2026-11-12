from pathlib import Path
import pandas as pd

from src.features.bigram_vocabulary import build_bigram_vocabulary


# Project folder
project_folder = Path(__file__).resolve().parents[2]

# Load training data only
train_path = project_folder / "data" / "interim" / "train_3class.csv"

train_df = pd.read_csv(train_path)

# Build bigram vocabulary
bigram_vocabulary = build_bigram_vocabulary(
    train_df["review"],
    min_frequency=100,
    max_bigrams=10000
)

print("Bigram vocabulary created!")

print("Total bigrams:", len(bigram_vocabulary))

print("\nFirst 20 bigrams:")

for bigram, index in list(bigram_vocabulary.items())[:20]:
    print(f"{bigram}: {index}")


# Check some known bigrams
print("\nChecking selected bigrams:")

test_bigrams = [
    "this garbage",
    "nothing special",
    "highly recommend",
    "waste of",
    "just perfect"
]

for bigram in test_bigrams:

    if bigram in bigram_vocabulary:
        print(f"{bigram}: Found")
    else:
        print(f"{bigram}: Not found")

from collections import Counter
from src.preprocessing.clean_text import clean_text

all_bigrams = Counter()

for review in train_df["review"]:
    words = clean_text(review).split()

    all_bigrams.update(
        " ".join(pair)
        for pair in zip(words, words[1:])
    )

print("\nActual training frequencies:")

for bigram in test_bigrams:
    print(f"{bigram}: {all_bigrams[bigram]}")