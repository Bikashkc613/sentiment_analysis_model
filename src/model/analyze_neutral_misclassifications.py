
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from src.model.softmax_model import forward_pass

from src.features.prepare_3class_data_with_bigrams import (
    prepare_training_data,
    prepare_validation_data
)


# Project folder
project_folder = Path(__file__).resolve().parents[2]


# Step 1: Prepare training data and vocabulary
(
    X_train,
    y_train,
    unigram_vocabulary,
    bigram_vocabulary
) = prepare_training_data()


# Step 2: Prepare validation data
X_validation, y_validation = prepare_validation_data(
    unigram_vocabulary,
    bigram_vocabulary
)


# Step 3: Load trained model
model_path = (
    project_folder
    / "models"
    / "sentiment_bigram_model.npz"
)

model = np.load(model_path)

W = model["W"]
b = model["b"]


# Step 4: Get predictions
_, probabilities = forward_pass(
    X_validation,
    W,
    b
)

predictions = np.argmax(
    probabilities,
    axis=1
)


# Step 5: Load original validation reviews
validation_path = (
    project_folder
    / "data"
    / "interim"
    / "validation_3class.csv"
)

df = pd.read_csv(validation_path)


# Step 6: Find Neutral reviews predicted incorrectly
neutral_mask = y_validation == 1

neutral_reviews = df.loc[
    neutral_mask,
    "review"
].reset_index(drop=True)

neutral_predictions = predictions[neutral_mask]

neutral_probabilities = probabilities[neutral_mask]


# Neutral predicted as Negative
negative_errors = neutral_predictions == 0

# Neutral predicted as Positive
positive_errors = neutral_predictions == 2


print("\nNEUTRAL MISCLASSIFICATION ANALYSIS")
print("=" * 60)

print("Total Neutral reviews:", len(neutral_reviews))

print(
    "Neutral predicted Negative:",
    np.sum(negative_errors)
)

print(
    "Neutral predicted Positive:",
    np.sum(positive_errors)
)


# Step 7: Display examples
print("\nNEUTRAL PREDICTED AS NEGATIVE")
print("=" * 60)

for index in np.where(negative_errors)[0][:10]:

    print("\nReview:")
    print(neutral_reviews[index][:500])

    print(
        "Probabilities:",
        np.round(neutral_probabilities[index], 4)
    )


print("\nNEUTRAL PREDICTED AS POSITIVE")
print("=" * 60)

for index in np.where(positive_errors)[0][:10]:

    print("\nReview:")
    print(neutral_reviews[index][:500])

    print(
        "Probabilities:",
        np.round(neutral_probabilities[index], 4)
    )


print("\nAnalysis completed!")
