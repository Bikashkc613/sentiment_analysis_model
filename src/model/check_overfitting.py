from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.features.combined_vectorize import (
    reviews_to_combined_matrix
)

from src.model.softmax_model import (
    forward_pass,
    cross_entropy_loss
)


project_folder = Path(__file__).resolve().parents[2]


# Load training and validation datasets
train_df = pd.read_csv(
    project_folder / "data/interim/train_3class.csv"
)

validation_df = pd.read_csv(
    project_folder / "data/interim/validation_3class.csv"
)


# Load vocabularies
with open(
    project_folder / "models/vocabulary.json",
    "r",
    encoding="utf-8"
) as file:
    unigram_vocabulary = json.load(file)


with open(
    project_folder / "models/bigram_vocabulary.json",
    "r",
    encoding="utf-8"
) as file:
    bigram_vocabulary = json.load(file)


# Load trained bigram model
model = np.load(
    project_folder / "models/sentiment_bigram_model.npz"
)

W = model["W"]
b = model["b"]


# Prepare features
X_train = reviews_to_combined_matrix(
    train_df["review"],
    unigram_vocabulary,
    bigram_vocabulary
)

X_validation = reviews_to_combined_matrix(
    validation_df["review"],
    unigram_vocabulary,
    bigram_vocabulary
)


# Convert labels
label_mapping = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

y_train = train_df["sentiment"].map(label_mapping).to_numpy()
y_validation = validation_df["sentiment"].map(label_mapping).to_numpy()


# Evaluate both datasets
for name, X, y in [
    ("TRAINING", X_train, y_train),
    ("VALIDATION", X_validation, y_validation)
]:

    _, probabilities = forward_pass(X, W, b)

    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y, predictions)

    loss = cross_entropy_loss(probabilities, y)

    print(f"\n{name} RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Loss: {loss:.4f}")


print("\nOverfitting check completed!")