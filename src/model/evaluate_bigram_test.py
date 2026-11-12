
from pathlib import Path
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
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


# Step 2: Prepare test data
X_test, y_test = prepare_testing_data(
    unigram_vocabulary,
    bigram_vocabulary
)

print("\nTest data prepared!")
print("Test matrix shape:", X_test.shape)


# Step 3: Load trained bigram model
model_path = (
    project_folder
    / "models"
    / "sentiment_bigram_model.npz"
)

model = np.load(model_path)

W = model["W"]
b = model["b"]

print("\nModel loaded successfully!")
print("Weight shape:", W.shape)
print("Bias shape:", b.shape)


# Step 4: Calculate probabilities
_, probabilities = forward_pass(
    X_test,
    W,
    b
)


# Step 5: Make predictions using threshold 0.30
neutral_threshold = 0.30

predictions = np.argmax(
    probabilities,
    axis=1
)

neutral_mask = (
    probabilities[:, 1] >= neutral_threshold
)

predictions[neutral_mask] = 1


# Step 6: Calculate accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "=" * 60)
print("BIGRAM MODEL TEST RESULTS")
print("=" * 60)

print(f"\nNeutral threshold: {neutral_threshold}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# Step 7: Classification report
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


# Step 8: Confusion matrix
print("\nConfusion Matrix:")
print("Rows = Actual, Columns = Predicted")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )
)


# Step 9: Individual class metrics
precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_test,
        predictions,
        labels=[0, 1, 2],
        zero_division=0
    )
)

print("\nIndividual Class Performance:")

class_names = [
    "Negative",
    "Neutral",
    "Positive"
]

for index, class_name in enumerate(class_names):

    print(f"\n{class_name}")
    print(f"Precision: {precision[index]:.3f}")
    print(f"Recall: {recall[index]:.3f}")
    print(f"F1 Score: {f1[index]:.3f}")
    print(f"Support: {support[index]}")


# Step 10: Prediction distribution
print("\nPrediction Distribution:")

prediction_counts = np.bincount(
    predictions,
    minlength=3
)

print("Negative:", prediction_counts[0])
print("Neutral:", prediction_counts[1])
print("Positive:", prediction_counts[2])


print("\nTest evaluation completed!")
