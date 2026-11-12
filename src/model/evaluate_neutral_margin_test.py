
from pathlib import Path
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.model.softmax_model import forward_pass

from src.features.prepare_3class_data_with_bigrams import (
    prepare_training_data,
    prepare_testing_data
)


# Project folder
project_folder = Path(__file__).resolve().parents[2]


# Step 1: Prepare training vocabulary
(
    X_train,
    y_train,
    unigram_vocabulary,
    bigram_vocabulary
) = prepare_training_data()


# Step 2: Prepare testing data
X_test, y_test = prepare_testing_data(
    unigram_vocabulary,
    bigram_vocabulary
)


# Step 3: Load bigram model
model_path = (
    project_folder
    / "models"
    / "sentiment_bigram_model.npz"
)

model = np.load(model_path)

W = model["W"]
b = model["b"]

print("\nModel information:")
print("Weight shape:", W.shape)
print("Bias shape:", b.shape)
print("Test matrix shape:", X_test.shape)


# Step 4: Calculate probabilities
_, probabilities = forward_pass(
    X_test,
    W,
    b
)


# Method 1: Normal argmax
normal_predictions = np.argmax(
    probabilities,
    axis=1
)


# Method 2: Neutral threshold 0.30
threshold_predictions = normal_predictions.copy()

threshold_mask = probabilities[:, 1] >= 0.30

threshold_predictions[threshold_mask] = 1


# Method 3: Neutral margin 0.12
margin_predictions = normal_predictions.copy()

negative_probability = probabilities[:, 0]
neutral_probability = probabilities[:, 1]
positive_probability = probabilities[:, 2]

strongest_other = np.maximum(
    negative_probability,
    positive_probability
)

margin = 0.12

margin_mask = (
    neutral_probability >= strongest_other - margin
)

margin_predictions[margin_mask] = 1


# Step 5: Evaluate all three methods
methods = [
    ("Normal argmax", normal_predictions),
    ("Neutral threshold 0.30", threshold_predictions),
    ("Neutral margin 0.12", margin_predictions)
]


for name, predictions in methods:

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=[
                "Negative",
                "Neutral",
                "Positive"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1, 2]
        )
    )


print("\nTest evaluation completed!")
