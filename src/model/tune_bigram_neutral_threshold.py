
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


project_folder = Path(__file__).resolve().parents[2]

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

model_path = (
    project_folder
    / "models"
    / "sentiment_bigram_model.npz"
)

model = np.load(model_path)

W = model["W"]
b = model["b"]

_, probabilities = forward_pass(
    X_validation,
    W,
    b
)

print("\nBIGRAM NEUTRAL THRESHOLD TUNING")

for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:

    predictions = np.argmax(probabilities, axis=1)

    # Predict Neutral when its probability reaches threshold
    neutral_mask = probabilities[:, 1] >= threshold

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

    print(f"\nNeutral threshold: {threshold}")

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(
        f"Neutral Precision: {precision[1]:.3f}"
    )

    print(
        f"Neutral Recall: {recall[1]:.3f}"
    )

    print(
        f"Neutral F1: {f1[1]:.3f}"
    )
