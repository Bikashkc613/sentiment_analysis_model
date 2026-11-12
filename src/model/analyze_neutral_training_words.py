from pathlib import Path
from collections import Counter

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


def count_document_frequency(reviews):
    """
    Count how many reviews contain each word.
    A word is counted only once per review.
    """
    counter = Counter()

    for review in reviews:
        words = set(clean_text(review).split())
        counter.update(words)

    return counter


# Separate reviews by sentiment
negative_reviews = df.loc[
    df["sentiment"] == "negative", "review"
]

neutral_reviews = df.loc[
    df["sentiment"] == "neutral", "review"
]

positive_reviews = df.loc[
    df["sentiment"] == "positive", "review"
]


# Count document frequencies
negative_counts = count_document_frequency(negative_reviews)
neutral_counts = count_document_frequency(neutral_reviews)
positive_counts = count_document_frequency(positive_reviews)


# Number of reviews in each class
negative_total = len(negative_reviews)
neutral_total = len(neutral_reviews)
positive_total = len(positive_reviews)


print("\nNORMALIZED NEUTRAL WORD ANALYSIS")
print("=" * 70)

results = []

for word in neutral_counts:

    neg_count = negative_counts.get(word, 0)
    neu_count = neutral_counts.get(word, 0)
    pos_count = positive_counts.get(word, 0)

    total_count = neg_count + neu_count + pos_count

    # Ignore words with very little evidence
    if total_count < 100 or neu_count < 30:
        continue

    # Word occurrence rate per 1,000 reviews
    neg_rate = neg_count / negative_total * 1000
    neu_rate = neu_count / neutral_total * 1000
    pos_rate = pos_count / positive_total * 1000

    # Compare Neutral rate against the stronger other class
    other_rate = max(neg_rate, pos_rate)

    neutral_lift = neu_rate / other_rate

    results.append(
        (
            word,
            neu_rate,
            neg_rate,
            pos_rate,
            neutral_lift,
            total_count
        )
    )


# Highest Neutral association first
results.sort(
    key=lambda item: item[4],
    reverse=True
)


print("\nWORDS MORE COMMON IN NEUTRAL REVIEWS")
print("=" * 90)

for word, neu, neg, pos, lift, total in results[:40]:

    print(
        f"{word:<18}"
        f"Neutral: {neu:7.2f} | "
        f"Negative: {neg:7.2f} | "
        f"Positive: {pos:7.2f} | "
        f"Lift: {lift:.2f}"
    )


print("\nAnalysis completed!")