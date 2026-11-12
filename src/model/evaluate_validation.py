
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.model.load_model import load_model
from src.model.softmax_model import forward_pass

from src.features.prepare_3class_data import (
    prepare_validation_data
)


# Step 1: Load trained model
W, b, vocabulary = load_model()


# Step 2: Load validation data
X_validation, y_validation = prepare_validation_data(
    vocabulary
)

print("\nValidation reviews:", X_validation.shape[0])


# Step 3: Make predictions
scores, probabilities = forward_pass(
    X_validation,
    W,
    b
)

y_pred = np.argmax(probabilities, axis=1)


# Step 4: Class names
class_names = ["Negative", "Neutral", "Positive"]


# Step 5: Calculate accuracy
accuracy = accuracy_score(
    y_validation,
    y_pred
)

print("\nValidation Accuracy:", accuracy * 100, "%")


# Step 6: Classification report
print("\nValidation Classification Report:")

print(
    classification_report(
        y_validation,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        zero_division=0
    )
)


# Step 7: Confusion matrix
print("\nValidation Confusion Matrix:")

matrix = confusion_matrix(
    y_validation,
    y_pred,
    labels=[0, 1, 2]
)

print(matrix)


# Step 8: Prediction distribution
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

for class_id, count in zip(
    unique_classes,
    prediction_counts
):
    print(label_mapping[class_id], ":", count)


# Step 9: Average predicted probabilities
print("\nAverage Predicted Probabilities:")

average_probabilities = np.mean(
    probabilities,
    axis=0
)

for class_id, probability in enumerate(
    average_probabilities
):
    print(label_mapping[class_id], ":", probability)
