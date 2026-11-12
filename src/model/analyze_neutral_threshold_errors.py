
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from src.model.softmax_model import forward_pass

from src.features.prepare_3class_data_with_bigrams import (
    prepare_training_data,
    prepare_testing_data
)


# Project folder
project_folder = Path(__file__).resolve().parents[2]


# Step 1: Prepare vocabulary
(
    X_train,
    y_train,
    unigram_vocabulary,
    bigram_vocabulary
) = prepare_training_data()


# Step 2: Prepare test data
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


# Step 4: Get probabilities
_, probabilities = forward_pass(
    X_test,
    W,
    b
)


# Method 1: Normal argmax prediction
normal_predictions = np.argmax(
    probabilities,
    axis=1
)


# Method 2: Neutral threshold prediction
threshold_predictions = normal_predictions.copy()

neutral_mask = probabilities[:, 1] >= 0.30

threshold_predictions[neutral_mask] = 1


# Step 5: Compare results
print("\nNEUTRAL THRESHOLD ERROR ANALYSIS")
print("=" * 60)

for name, predictions in [
    ("Normal argmax", normal_predictions),
    ("Neutral threshold 0.30", threshold_predictions)
]:

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            labels=[0, 1, 2],
            zero_division=0
        )
    )

    print(f"\n{name}")
    print("-" * 40)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Neutral Precision: {precision[1]:.3f}")
    print(f"Neutral Recall: {recall[1]:.3f}")
    print(f"Neutral F1: {f1[1]:.3f}")


# Step 6: Count Positive reviews changed to Neutral
changed_mask = (
    (y_test == 2) &
    (normal_predictions == 2) &
    (threshold_predictions == 1)
)

print("\nPositive reviews changed to Neutral by threshold:")
print(np.sum(changed_mask))


# Step 7: Count Negative reviews changed to Neutral
negative_changed_mask = (
    (y_test == 0) &
    (normal_predictions == 0) &
    (threshold_predictions == 1)
)

print("\nNegative reviews changed to Neutral by threshold:")
print(np.sum(negative_changed_mask))


# Step 8: Count all threshold-induced changes
print("\nTotal predictions changed by threshold:")
print(np.sum(normal_predictions != threshold_predictions))


print("\nAnalysis completed!")
