
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.model.load_model import load_model
from src.model.softmax_model import forward_pass

from src.features.prepare_3class_data import (
    prepare_testing_data,
    prepare_training_data
)


# Project folder
project_folder = Path(__file__).resolve().parents[2]


# Step 1: Load trained model
W, b, vocabulary = load_model()


# Step 2: Check training vocabulary
print("\nChecking training vocabulary...")

X_train, y_train, train_vocabulary = prepare_training_data()


# Step 3: Load testing data
X_test, y_test = prepare_testing_data(vocabulary)

test_path = (
    project_folder
    / "data"
    / "interim"
    / "test_3class.csv"
)

test_df = pd.read_csv(test_path)

print("\nTesting reviews:", X_test.shape[0])


# Step 4: Make predictions
scores, probabilities = forward_pass(X_test, W, b)

# Normal predictions
y_pred = np.argmax(probabilities, axis=1)

# Apply selected Neutral threshold
neutral_mask = probabilities[:, 1] >= 0.30
y_pred[neutral_mask] = 1


# Step 5: Inspect incorrectly predicted Neutral reviews
class_names = ["Negative", "Neutral", "Positive"]

neutral_label = 1

wrong_neutral_indices = np.where(
    (y_test == neutral_label) &
    (y_pred != neutral_label)
)[0]

print("\nIncorrectly predicted Neutral reviews:")
print("Total:", len(wrong_neutral_indices))

for index in wrong_neutral_indices[:10]:

    print("\nReview:")
    print(test_df["review"].iloc[index])

    print("Actual: Neutral")

    print("Predicted:", class_names[y_pred[index]])

    print("Probabilities:", probabilities[index])


# Step 6: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")


# Step 7: Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        zero_division=0
    )
)


# Step 8: Confusion matrix
print("\nConfusion Matrix:")

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)

print(matrix)


# Step 9: Prediction distribution
print("\nPrediction Distribution:")

unique_classes, prediction_counts = np.unique(
    y_pred,
    return_counts=True
)

label_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

for class_id, count in zip(unique_classes, prediction_counts):
    print(label_mapping[class_id], ":", count)


# Step 10: Average predicted probabilities
print("\nAverage Predicted Probabilities:")

average_probabilities = np.mean(probabilities, axis=0)

for class_id, probability in enumerate(average_probabilities):
    print(label_mapping[class_id], ":", probability)
