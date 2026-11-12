from collections import Counter

import pandas as pd

from src.preprocessing.clean_text import clean_text

DATA_PATH = "data/interim/train_3class.csv"

df = pd.read_csv(DATA_PATH)

target_bigrams = [
    "this movie",
    "movie is",
    "is average"
]

counts = {
    bigram: Counter()
    for bigram in target_bigrams
}

for _, row in df.iterrows():
    words = clean_text(row["review"]).split()
    bigrams = [
        f"{words[i]} {words[i + 1]}"
        for i in range(len(words) - 1)
    ]

    for bigram in target_bigrams:
        if bigram in bigrams:
            counts[bigram][row["sentiment"]] += 1

for bigram, sentiment_counts in counts.items():
    print(f"\n{bigram.upper()}")
    print(sentiment_counts)