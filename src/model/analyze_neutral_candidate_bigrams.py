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


def count_bigrams(reviews):

    counter = Counter()

    for review in reviews:

        words = clean_text(review).split()

        bigrams = set(
            " ".join(pair)
            for pair in zip(words, words[1:])
        )

        counter.update(bigrams)

    return counter


class_counts = {}

for sentiment in ["negative", "neutral", "positive"]:

    reviews = df.loc[
        df["sentiment"] == sentiment,
        "review"
    ]

    class_counts[sentiment] = count_bigrams(reviews)


print("\nMOST COMMON NEUTRAL BIGRAMS")
print("=" * 70)

neutral_counts = class_counts["neutral"]

for bigram, count in neutral_counts.most_common(40):

    print(f"{bigram:<30} {count}")


print("\nNEUTRAL-SPECIFIC BIGRAMS")
print("=" * 70)

results = []

for bigram, neutral_count in neutral_counts.items():

    negative_count = class_counts["negative"].get(bigram, 0)
    positive_count = class_counts["positive"].get(bigram, 0)

    if neutral_count < 20:
        continue

    neutral_rate = neutral_count / len(
        df[df["sentiment"] == "neutral"]
    )

    negative_rate = negative_count / len(
        df[df["sentiment"] == "negative"]
    )

    positive_rate = positive_count / len(
        df[df["sentiment"] == "positive"]
    )

    strongest_other = max(negative_rate, positive_rate)

    lift = neutral_rate / strongest_other if strongest_other else 0

    results.append((bigram, neutral_count, lift))


results.sort(key=lambda item: item[2], reverse=True)

for bigram, count, lift in results[:40]:

    print(
        f"{bigram:<30}"
        f"Count: {count:<6}"
        f"Lift: {lift:.2f}"
    )


print("\nAnalysis completed!")