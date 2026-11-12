from scipy.sparse import hstack

from src.features.vectorize import reviews_to_matrix
from src.features.bigram_vectorize import reviews_to_bigram_matrix


def reviews_to_combined_matrix(
    reviews,
    unigram_vocabulary,
    bigram_vocabulary
):

    # Step 1: Create unigram matrix
    unigram_matrix = reviews_to_matrix(
        reviews,
        unigram_vocabulary
    )

    # Step 2: Create bigram matrix
    bigram_matrix = reviews_to_bigram_matrix(
        reviews,
        bigram_vocabulary
    )

    # Step 3: Combine both matrices horizontally
    combined_matrix = hstack(
        [unigram_matrix, bigram_matrix],
        format="csr"
    )

    return combined_matrix