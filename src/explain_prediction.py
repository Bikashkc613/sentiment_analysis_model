import numpy as np

from src.predict import (
    W,
    b,
    unigram_vocabulary,
    bigram_vocabulary
)

from src.features.combined_vectorize import (
    reviews_to_combined_matrix
)


review = "This movie is average"

X = reviews_to_combined_matrix(
    [review],
    unigram_vocabulary,
    bigram_vocabulary
)

row = X.getrow(0)

unigram_reverse = {
    index: word
    for word, index in unigram_vocabulary.items()
}

bigram_reverse = {
    index: phrase
    for phrase, index in bigram_vocabulary.items()
}

print("\nACTIVE FEATURE CONTRIBUTIONS")
print("=" * 75)

for index, count in zip(row.indices, row.data):

    if index < len(unigram_vocabulary):
        feature = unigram_reverse[index]
    else:
        feature = bigram_reverse[
            index - len(unigram_vocabulary)
        ]

    contribution = count * W[index]

    print(
        f"{feature:20}",
        f"NEG: {contribution[0]: .4f}",
        f"NEU: {contribution[1]: .4f}",
        f"POS: {contribution[2]: .4f}"
    )

scores = np.asarray(row @ W).ravel() + b

print("\nFINAL CLASS SCORES")
print("Negative:", scores[0])
print("Neutral :", scores[1])
print("Positive:", scores[2])