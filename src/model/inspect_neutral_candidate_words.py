from pathlib import Path

import pandas as pd

from src.preprocessing.clean_text import clean_text


project_folder = Path(__file__).resolve().parents[2]

train_path = (
    project_folder
    / "data"
    / "interim"
    / "train_3class.csv"
)

df = pd.read_csv(train_path)

candidate_words = [
    "uneven",
    "weakest",
    "formulaic",
    "setup",
]

for word in candidate_words:

    print("\n" + "=" * 70)
    print(f"WORD: {word}")
    print("=" * 70)

    matching = df[
        df["review"].apply(
            lambda review: word in clean_text(review).split()
        )
    ]

    print("\nSentiment distribution:")
    print(matching["sentiment"].value_counts())

    print("\nNeutral review examples:")

    neutral_examples = matching[
        matching["sentiment"] == "neutral"
    ]["review"].head(5)

    for i, review in enumerate(neutral_examples, start=1):
        print(f"\nExample {i}:")
        print(review[:700])