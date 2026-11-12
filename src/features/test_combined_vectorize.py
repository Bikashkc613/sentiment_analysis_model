from pathlib import Path
import pandas as pd
import json

from src.features.combined_vectorize import reviews_to_combined_matrix


# Project folder
project_folder = Path(__file__).resolve().parents[2]

# Load training data
train_path = project_folder / "data" / "interim" / "train_3class.csv"
train_df = pd.read_csv(train_path)

# Load existing unigram vocabulary
unigram_path = project_folder / "models" / "vocabulary.json"

with open(unigram_path, "r", encoding="utf-8") as file:
    unigram_vocabulary = json.load(file)

# Build bigram vocabulary from training reviews
from src.features.bigram_vocabulary import build_bigram_vocabulary

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

# Create combined matrix
combined_matrix = reviews_to_combined_matrix(
    test_reviews,
    unigram_vocabulary,
    bigram_vocabulary
)

print("Combined vectorization successful!")

print("\nUnigram vocabulary size:", len(unigram_vocabulary))
print("Bigram vocabulary size:", len(bigram_vocabulary))

print("\nCombined matrix shape:")
print(combined_matrix.shape)

print("\nNon-zero values:")
print(combined_matrix.nnz)

print("\nMatrix type:")
print(type(combined_matrix))