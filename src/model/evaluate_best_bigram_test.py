
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.features.prepare_3class_data_with_bigrams import (
    prepare_training_data,
    prepare_testing_data
)

from src.model.softmax_model import forward_pass


# Project folder
project_folder = Path(__file__).resolve().parents[2]


# Step 1: Load best saved model
model_path = project_folder / "models" / "sentiment_bigram_model_best.npz"

model = np.load(model_path)

W = model["W"]
b = model["b"]

print("Best model loaded successfully!")
print("Weight shape:", W.shape)
print("Bias shape:", b.shape)
print("Learning rate:", model["learning_rate"])
print("Best epoch:", model["best_epoch"])


# Step 2: Prepare vocabulary and test data
(
    X_train,
    y_train,
    unigram_vocabulary,
    bigram_vocabulary
) = prepare_training_data()

X_test, y_test = prepare_testing_data(
    unigram_vocabulary,
    bigram_vocabulary
)


# Step 3: Make predictions
scores, probabilities = forward_pass(X_test, W, b)

predictions = np.argmax(probabilities, axis=1)


# Step 4: Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print("BEST BIGRAM MODEL - TEST EVALUATION")
print("=" * 60)

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Step 5: Classification report
print("\nCLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=["Negative", "Neutral", "Positive"],
        digits=4
    )
)


# Step 6: Confusion matrix
print("\nCONFUSION MATRIX")
print("=" * 60)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1, 2]
)

print(matrix)

print("\nRows = Actual sentiment")
print("Columns = Predicted sentiment")


# Step 7: Prediction distribution
print("\nPREDICTION DISTRIBUTION")
print("=" * 60)

class_names = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

for class_id, class_name in class_names.items():

    count = np.sum(predictions == class_id)

    percentage = count / len(predictions) * 100

    print(
        f"{class_name:<10}: "
        f"{count:>6} ({percentage:.2f}%)"
    )
