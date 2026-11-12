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

bigram_counts = {
    "negative": Counter(),
    "neutral": Counter(),
    "positive": Counter()
}

for _, row in df.iterrows():

    review = clean_text(row["review"])
    words = review.split()

    bigrams = zip(words, words[1:])

    bigram_counts[row["sentiment"]].update(
        " ".join(pair) for pair in bigrams
    )


# Total bigram occurrences per class
class_totals = {
    sentiment: sum(counts.values())
    for sentiment, counts in bigram_counts.items()
}

# All unique bigrams
all_bigrams = set()

for counts in bigram_counts.values():
    all_bigrams.update(counts.keys())

vocabulary_size = len(all_bigrams)


for sentiment in bigram_counts:

    print(f"\n{'=' * 50}")
    print(f"TOP DISTINCTIVE BIGRAMS: {sentiment.upper()}")
    print("=" * 50)

    other_sentiments = [
        s for s in bigram_counts if s != sentiment
    ]

    scores = []

    for bigram, count in bigram_counts[sentiment].items():

        if count < 100:
            continue

        other_count = sum(
            bigram_counts[s][bigram]
            for s in other_sentiments
        )

        other_total = sum(
            class_totals[s]
            for s in other_sentiments
        )

        # Add-one smoothing
        current_rate = (
            (count + 1) /
            (class_totals[sentiment] + vocabulary_size)
        )

        other_rate = (
            (other_count + 1) /
            (other_total + vocabulary_size)
        )

        lift = current_rate / other_rate

        scores.append((bigram, count, lift))

    scores.sort(key=lambda x: x[2], reverse=True)

    for bigram, count, lift in scores[:20]:

        print(
            f"{bigram:25} "
            f"Count: {count:6} "
            f"Lift: {lift:.2f}"
        )