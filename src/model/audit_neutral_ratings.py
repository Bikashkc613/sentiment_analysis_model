
from pathlib import Path

import pandas as pd

from src.preprocessing.clean_text import clean_text


PROJECT_FOLDER = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_FOLDER / "data" / "raw" / "Data.csv"

OUTPUT_PATH = (
    PROJECT_FOLDER
    / "data"
    / "processed"
    / "rating_label_audit.csv"
)


def main():

    print("=" * 65)
    print("NEUTRAL RATING LABEL AUDIT")
    print("=" * 65)

    # Step 1: Load original dataset
    df = pd.read_csv(RAW_PATH)

    print("\nOriginal dataset:", df.shape)

    # Step 2: Prepare reviews and ratings
    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    df = df.dropna(subset=["rating", "review"]).copy()

    df["cleaned_review"] = (
        df["review"]
        .astype(str)
        .apply(clean_text)
    )

    df = df[df["cleaned_review"] != ""]

    # Step 3: Remove reviews with conflicting ratings
    conflicting_reviews = (
        df.groupby("cleaned_review")["rating"]
        .nunique()
    )

    conflicting_reviews = conflicting_reviews[
        conflicting_reviews > 1
    ].index

    df = df[
        ~df["cleaned_review"].isin(conflicting_reviews)
    ].copy()

    # Step 4: Remove duplicate reviews
    df = df.drop_duplicates(
        subset="cleaned_review"
    ).copy()

    # Step 5: Assign sentiment labels
    df["sentiment"] = pd.cut(
        df["rating"],
        bins=[0, 3, 7, 10],
        labels=["negative", "neutral", "positive"]
    )

    print("\nCleaned dataset:", df.shape)

    # Step 6: Display rating distribution
    print("\nRATING DISTRIBUTION")
    print("-" * 65)

    print(
        df["rating"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # Step 7: Filter neutral ratings
    neutral_df = df[
        df["rating"].between(4, 7)
    ].copy()

    print("\nNEUTRAL CLASS RATING COUNTS")
    print("-" * 65)

    print(
        neutral_df["rating"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # Step 8: Show random examples for each rating
    print("\nRANDOM REVIEW EXAMPLES")
    print("=" * 65)

    for rating in [4, 5, 6, 7]:

        subset = neutral_df[
            neutral_df["rating"] == rating
        ]

        print(f"\nRATING: {rating}")
        print("-" * 65)

        if subset.empty:
            print("No reviews found.")
            continue

        samples = subset.sample(
            n=min(5, len(subset)),
            random_state=42
        )

        for index, row in enumerate(
            samples.itertuples(),
            start=1
        ):

            print(f"\nExample {index}")
            print("Rating:", row.rating)
            print("Assigned label:", row.sentiment)
            print("Review:", str(row.review)[:500])

    # Step 9: Save audit dataset
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    neutral_df[
        ["review", "rating", "sentiment"]
    ].to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("AUDIT COMPLETED")
    print("=" * 65)

    print("Saved:", OUTPUT_PATH)
    print("Total neutral-labelled reviews:", len(neutral_df))


if __name__ == "__main__":
    main()
