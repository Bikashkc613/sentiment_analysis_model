
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# Project directory
PROJECT_FOLDER = Path(__file__).resolve().parents[2]

# Input file
ERROR_FILE = (
    PROJECT_FOLDER
    / "data"
    / "processed"
    / "prediction_errors.csv"
)

# Load prediction errors
errors_df = pd.read_csv(ERROR_FILE)

print("=" * 65)
print("DETAILED ERROR PATTERN ANALYSIS")
print("=" * 65)

print("\nTotal incorrect reviews:", len(errors_df))


# --------------------------------------------------
# STEP 1: Count error types
# --------------------------------------------------

print("\n1. ERROR TYPE DISTRIBUTION")
print("-" * 65)

error_counts = errors_df["error_type"].value_counts()

for error_type, count in error_counts.items():

    percentage = count / len(errors_df) * 100

    print(
        f"{error_type:25} "
        f"{count:5} "
        f"({percentage:.2f}%)"
    )


# --------------------------------------------------
# STEP 2: Analyze errors by actual sentiment
# --------------------------------------------------

print("\n\n2. ERRORS BY ACTUAL SENTIMENT")
print("-" * 65)

for sentiment in ["Negative", "Neutral", "Positive"]:

    subset = errors_df[
        errors_df["actual_sentiment"] == sentiment
    ]

    print(f"\n{sentiment} actual reviews: {len(subset)}")

    print(
        subset["predicted_sentiment"]
        .value_counts()
        .to_string()
    )


# --------------------------------------------------
# STEP 3: Find frequent words in error reviews
# --------------------------------------------------

print("\n\n3. MOST COMMON WORDS IN ERROR REVIEWS")
print("-" * 65)

vectorizer = CountVectorizer(
    stop_words="english",
    max_features=100
)

X = vectorizer.fit_transform(
    errors_df["review"].fillna("").astype(str)
)

word_counts = X.sum(axis=0).A1

words = vectorizer.get_feature_names_out()

word_frequency = pd.DataFrame({
    "word": words,
    "frequency": word_counts
})

word_frequency = word_frequency.sort_values(
    by="frequency",
    ascending=False
)

print("\nTop 30 frequent words:")

print(
    word_frequency.head(30).to_string(index=False)
)


# --------------------------------------------------
# STEP 4: Compare frequent words by actual sentiment
# --------------------------------------------------

print("\n\n4. FREQUENT WORDS BY ACTUAL SENTIMENT")
print("-" * 65)

for sentiment in ["Negative", "Neutral", "Positive"]:

    subset = errors_df[
        errors_df["actual_sentiment"] == sentiment
    ]

    if subset.empty:
        continue

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=50
    )

    X = vectorizer.fit_transform(
        subset["review"].fillna("").astype(str)
    )

    word_counts = X.sum(axis=0).A1

    words = vectorizer.get_feature_names_out()

    counts = Counter(dict(zip(words, word_counts)))

    print(f"\n{sentiment} misclassified reviews:")

    for word, count in counts.most_common(20):

        print(f"{word:20} {count}")


# --------------------------------------------------
# STEP 5: Show examples from each error type
# --------------------------------------------------

print("\n\n5. EXAMPLE REVIEWS BY ERROR TYPE")
print("-" * 65)

for error_type in error_counts.index:

    subset = errors_df[
        errors_df["error_type"] == error_type
    ].sort_values("confidence")

    print(f"\n\n{error_type}")
    print("-" * 65)

    for _, row in subset.head(3).iterrows():

        print(f"\nConfidence: {row['confidence']:.2f}%")

        print(
            "Review:",
            str(row["review"])[:350]
        )


# --------------------------------------------------
# STEP 6: Save word frequency report
# --------------------------------------------------

output_path = (
    PROJECT_FOLDER
    / "data"
    / "processed"
    / "error_word_frequency.csv"
)

word_frequency.to_csv(
    output_path,
    index=False
)

print("\n\n" + "=" * 65)
print("ERROR PATTERN ANALYSIS COMPLETED")
print("=" * 65)

print("Word frequency report saved to:")
print(output_path)
