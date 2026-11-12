
from collections import Counter

from src.preprocessing.clean_text import clean_text


def build_bigram_vocabulary(reviews, min_frequency=100, max_bigrams=2000):

    bigram_counts = Counter()

    for review in reviews:

        review = clean_text(review)
        words = review.split()

        bigrams = zip(words, words[1:])

        bigram_counts.update(
            " ".join(pair) for pair in bigrams
        )

    # Remove rare bigrams
    frequent_bigrams = [
        (bigram, count)
        for bigram, count in bigram_counts.items()
        if count >= min_frequency
    ]

    # Keep the most frequent bigrams
    frequent_bigrams.sort(
        key=lambda item: item[1],
        reverse=True
    )

    frequent_bigrams = frequent_bigrams[:max_bigrams]

    # Assign each bigram a unique index
    vocabulary = {}

    for bigram, count in frequent_bigrams:
        vocabulary[bigram] = len(vocabulary)

    return vocabulary
