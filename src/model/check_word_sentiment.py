from pathlib import Path

import pandas as pd

from src.preprocessing.clean_text import clean_text


project_folder = Path(__file__).resolve().parents[2]

train_path = project_folder / "data" / "interim" / "train_3class.csv"

df = pd.read_csv(train_path)

words = ["hated", "hate", "terrible", "boring"]

for word in words:

    print(f"\n{'=' * 50}")
    print(f"WORD: {word}")
    print("=" * 50)

    mask = df["review"].apply(
        lambda review: word in clean_text(review).split()
    )

    matching_reviews = df[mask]

    print("Total reviews:", len(matching_reviews))

    print("\nSentiment distribution:")
    distribution = matching_reviews["sentiment"].value_counts(
    normalize=True
) * 100

    print(distribution.round(2))

    print("\nExample reviews:")

    for _, row in matching_reviews.head(5).iterrows():
        print(f"\nSentiment: {row['sentiment']}")
        print(f"Review: {row['review']}")