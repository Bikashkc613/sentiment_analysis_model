
from pathlib import Path
import numpy as np
import pandas as pd

from src.model.softmax_model import forward_pass

from src.features.prepare_3class_data_with_bigrams import (
    prepare_training_data,
    prepare_testing_data
)


# Project folder
project_folder = Path(__file__).resolve().parents[2]


# Step 1: Prepare vocabularies
(
    X_train,
    y_train,
    unigram_vocabulary,
    bigram_vocabulary
) = prepare_training_data()


# Step 2: Load test reviews
test_path = (
    project_folder
    / "data"
    / "interim"
    / "test_3class.csv"
)

test_df = pd.read_csv(test_path)

X_test, y_test = prepare_testing_data(
    unigram_vocabulary,
    bigram_vocabulary
)


# Step 3: Load model
model_path = (
    project_folder
    / "models"
    / "sentiment_bigram_model.npz"
)

model = np.load(model_path)

W = model["W"]
b = model["b"]


# Step 4: Predict
_, probabilities = forward_pass(
    X_test,
    W,
    b
)

predictions = np.argmax(
    probabilities,
    axis=1
)

neutral_threshold = 0.30

neutral_mask = probabilities[:, 1] >= neutral_threshold
predictions[neutral_mask] = 1


# Step 5: Find Positive reviews predicted as Neutral
mistake_mask = (
    (y_test == 2) &
    (predictions == 1)
)

mistake_indices = np.where(mistake_mask)[0]

print("\nPOSITIVE REVIEWS PREDICTED AS NEUTRAL")
print("=" * 60)

print("Total mistakes:", len(mistake_indices))


# Step 6: Display examples
for index in mistake_indices[:20]:

    review = test_df.iloc[index]["review"]

    print("\n" + "-" * 60)

    print("Review:")
    print(review)

    print("\nActual sentiment: Positive")
    print("Predicted sentiment: Neutral")

    print("\nProbabilities:")

    print(
        f"Negative: {probabilities[index, 0]:.4f}"
    )

    print(
        f"Neutral: {probabilities[index, 1]:.4f}"
    )

    print(
        f"Positive: {probabilities[index, 2]:.4f}"
    )


print("\nDiagnostic completed!")
