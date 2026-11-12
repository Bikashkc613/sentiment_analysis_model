
from collections import Counter
from src.preprocessing.clean_text import clean_text


def build_vocabulary(reviews, max_words=5000):
    """
    Build vocabulary from cleaned training reviews.
    """

    word_counts = Counter()

    for review in reviews:

        # Clean text before extracting words
        review = clean_text(review)

        words = review.split()
        word_counts.update(words)

    most_common_words = word_counts.most_common(max_words)

    vocabulary = {}

    for word, count in most_common_words:
        vocabulary[word] = len(vocabulary)

    return vocabulary
