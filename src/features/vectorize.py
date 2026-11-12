
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix

from src.preprocessing.clean_text import clean_text


def review_to_vector(review, vocabulary):
    """
    Convert one cleaned review into a numerical vector.
    """

    review = clean_text(review)

    vector = [0] * len(vocabulary)

    for word in review.split():
        if word in vocabulary:
            index = vocabulary[word]
            vector[index] += 1

    return vector


def reviews_to_matrix(reviews, vocabulary):
    """
    Convert multiple reviews into a sparse numerical matrix.
    """

    data = []
    indices = []
    indptr = [0]

    for review in reviews:

        # Clean each review before counting words
        review = clean_text(review)

        word_counts = Counter(review.split())

        for word, count in word_counts.items():

            if word in vocabulary:
                indices.append(vocabulary[word])
                data.append(count)

        indptr.append(len(indices))

    matrix = csr_matrix(
        (data, indices, indptr),
        shape=(len(reviews), len(vocabulary)),
        dtype=np.float32
    )

    return matrix
