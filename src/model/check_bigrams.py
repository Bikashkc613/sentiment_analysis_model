from pathlib import Path
from collections import Counter

import pandas as pd

from src.preprocessing.clean_text import clean_text


project_folder = Path(__file__).resolve().parents[2]

train_path = project_folder / "data" / "interim" / "train_3class.csv"

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


for sentiment in bigram_counts:

    print(f"\n{'=' * 45}")
    print(f"TOP BIGRAMS: {sentiment.upper()}")
    print("=" * 45)

    for bigram, count in bigram_counts[sentiment].most_common(15):
        print(f"{bigram:25} {count}")