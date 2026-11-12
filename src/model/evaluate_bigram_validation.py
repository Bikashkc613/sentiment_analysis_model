
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
    prepare_validation_data
)


# Project folder
project_folder = Path(__file__).resolve().parents[2]

# Prepare vocabularies and training data
(
    X_train,
    y_train,
    unigram_vocabulary,
    bigram_vocabulary
) = prepare_training_data()

# Prepare validation data
X_validation, y_validation = prepare_validation_data(
    unigram_vocabulary,
    bigram_vocabulary
)

# Load experimental model
model_path = project_folder / "models" / "sentiment_bigram_model.npz"

model = np.load(model_path)

W = model["W"]
b = model["b"]

print("\nBigram model loaded!")
print("Weight shape:", W.shape)
print("Bias shape:", b.shape)

# Predict
_, probabilities = forward_pass(
    X_validation,
    W,
    b
)

predictions = np.argmax(probabilities, axis=1)

# Evaluate
accuracy = accuracy_score(
    y_validation,
    predictions
)

print("\nValidation Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        predictions,
        target_names=["Negative", "Neutral", "Positive"]
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_validation,
        predictions
    )
)

print("\nPrediction distribution:")

print(np.bincount(predictions, minlength=3))

print("\nAverage class probabilities:")

print(np.mean(probabilities, axis=0))
