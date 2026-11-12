import numpy as np

from collections import Counter
from scipy.sparse import csr_matrix

from src.preprocessing.clean_text import clean_text


def reviews_to_bigram_matrix(reviews, bigram_vocabulary):

    data = []
    indices = []
    indptr = [0]

    for review in reviews:

        # Clean review
        review = clean_text(review)

        # Split into words
        words = review.split()

        # Generate consecutive two-word phrases
        bigrams = [
            " ".join(pair)
            for pair in zip(words, words[1:])
        ]

        # Count bigrams
        bigram_counts = Counter(bigrams)

        # Keep only bigrams in vocabulary
        for bigram, count in bigram_counts.items():

            if bigram in bigram_vocabulary:

                index = bigram_vocabulary[bigram]

                indices.append(index)
                data.append(count)

        indptr.append(len(indices))

    # Create sparse matrix
    matrix = csr_matrix(
        (data, indices, indptr),
        shape=(len(reviews), len(bigram_vocabulary)),
        dtype=np.float32
    )

    return matrix