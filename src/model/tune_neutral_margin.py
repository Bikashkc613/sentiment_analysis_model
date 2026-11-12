
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from src.model.softmax_model import forward_pass

from src.features.prepare_3class_data_with_bigrams import (
    prepare_training_data,
    prepare_validation_data
)


# Project folder
project_folder = Path(__file__).resolve().parents[2]


# Step 1: Prepare data
(
    X_train,
    y_train,
    unigram_vocabulary,
    bigram_vocabulary
) = prepare_training_data()

X_validation, y_validation = prepare_validation_data(
    unigram_vocabulary,
    bigram_vocabulary
)


# Step 2: Load trained model
model_path = (
    project_folder
    / "models"
    / "sentiment_bigram_model.npz"
)

model = np.load(model_path)

W = model["W"]
b = model["b"]


# Step 3: Get probabilities
_, probabilities = forward_pass(
    X_validation,
    W,
    b
)

normal_predictions = np.argmax(
    probabilities,
    axis=1
)


# Step 4: Test different Neutral margins
print("\nNEUTRAL MARGIN TUNING")
print("=" * 65)

print(
    f"{'Margin':<12}"
    f"{'Accuracy':<12}"
    f"{'Neutral P':<12}"
    f"{'Neutral R':<12}"
    f"{'Neutral F1':<12}"
)

for margin in [
    0.10,
    0.11,
    0.12,
    0.13,
    0.14
]:

    predictions = normal_predictions.copy()

    negative_probability = probabilities[:, 0]
    neutral_probability = probabilities[:, 1]
    positive_probability = probabilities[:, 2]

    strongest_other_probability = np.maximum(
        negative_probability,
        positive_probability
    )

    # Predict Neutral when its probability is
    # close enough to the strongest other class.
    neutral_mask = (
        neutral_probability >= strongest_other_probability - margin
    )

    predictions[neutral_mask] = 1

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_validation,
        predictions,
        labels=[0, 1, 2],
        zero_division=0
    )

    print(
        f"{margin:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{precision[1]:<12.4f}"
        f"{recall[1]:<12.4f}"
        f"{f1[1]:<12.4f}"
    )


print("\nTuning completed!")
