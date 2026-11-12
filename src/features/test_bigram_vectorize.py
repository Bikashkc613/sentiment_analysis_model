from pathlib import Path
import pandas as pd

from src.features.bigram_vocabulary import build_bigram_vocabulary
from src.features.bigram_vectorize import reviews_to_bigram_matrix


# Project folder
project_folder = Path(__file__).resolve().parents[2]

# Load training data
train_path = project_folder / "data" / "interim" / "train_3class.csv"

train_df = pd.read_csv(train_path)

# Build bigram vocabulary
bigram_vocabulary = build_bigram_vocabulary(
    train_df["review"],
    min_frequency=100,
    max_bigrams=10000
)

# Test reviews
test_reviews = [
    "This garbage movie is terrible",
    "Nothing special but highly recommend",
    "This movie is just perfect"
]

# Convert reviews into bigram matrix
bigram_matrix = reviews_to_bigram_matrix(
    test_reviews,
    bigram_vocabulary
)

print("Bigram vectorization successful!")

print("\nMatrix shape:")
print(bigram_matrix.shape)

print("\nNon-zero values:")
print(bigram_matrix.nnz)

# Inspect recognized bigrams
print("\nRecognized bigrams:")

for review_number, review in enumerate(test_reviews):

    print(f"\nReview: {review}")

    row = bigram_matrix.getrow(review_number)

    for index, count in zip(row.indices, row.data):

        bigram = next(
            word for word, vocab_index in bigram_vocabulary.items()
            if vocab_index == index
        )

        print(f"{bigram}: {int(count)}")