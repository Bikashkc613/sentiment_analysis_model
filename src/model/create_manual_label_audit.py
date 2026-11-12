
from pathlib import Path
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_FOLDER
    / "data"
    / "processed"
    / "rating_label_audit.csv"
)

OUTPUT_PATH = (
    PROJECT_FOLDER
    / "data"
    / "processed"
    / "manual_label_audit.csv"
)


def main():

    df = pd.read_csv(INPUT_PATH)

    samples = []

    # Select 25 examples from each rating
    for rating in [4, 5, 6, 7]:

        subset = df[df["rating"] == rating]

        sample = subset.sample(
            n=min(25, len(subset)),
            random_state=42
        )

        samples.append(sample)

    audit_df = pd.concat(
        samples,
        ignore_index=True
    )

    # Add an empty column for manual human labels
    audit_df["human_sentiment"] = ""

    audit_df["notes"] = ""

    audit_df = audit_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    audit_df.insert(
        0,
        "sample_id",
        range(1, len(audit_df) + 1)
    )

    audit_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("=" * 60)
    print("MANUAL LABEL AUDIT CREATED")
    print("=" * 60)

    print("Total samples:", len(audit_df))

    print("\nRating distribution:")
    print(audit_df["rating"].value_counts().sort_index())

    print("\nFile saved:")
    print(OUTPUT_PATH)

    print("\nFill human_sentiment with:")
    print("Negative / Neutral / Positive")


if __name__ == "__main__":
    main()
