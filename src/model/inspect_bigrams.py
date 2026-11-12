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

selected_bigrams = [
    "this garbage",
    "nothing special",
    "not great",
    "is okay",
    "just perfect",
    "like iti",
     "this garbage",
    "nothing special",
    "not great",
    "is okay",
    "just perfect",
    "like iti",
    "this trash",
    "this crap",
    "highly recommend",
    "waste of",
    "must watch",
    "very good",
    "very bad",
    "not worth",
    "highly overrated"
]

for bigram in selected_bigrams:

    print("\n" + "=" * 60)
    print("BIGRAM:", bigram)
    print("=" * 60)

    matches = df[
        df["review"].apply(
            lambda review: bigram in clean_text(review)
        )
    ]

    print("Total reviews:", len(matches))

    print("\nSentiment distribution:")
    print(matches["sentiment"].value_counts())

    print("\nReview examples:")

    for _, row in matches.head(3).iterrows():

        print("\nSentiment:", row["sentiment"])
        print("Review:", row["review"][:400])