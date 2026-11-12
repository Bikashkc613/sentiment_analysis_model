import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from src.model.load_model import load_model
from src.model.softmax_model import forward_pass
from src.features.prepare_3class_data import prepare_validation_data


# Step 1: Load trained model
W, b, vocabulary = load_model()

# Step 2: Load validation data
X_validation, y_validation = prepare_validation_data(vocabulary)

# Step 3: Get probabilities
scores, probabilities = forward_pass(
    X_validation, W, b
)

# Step 4: Try different Neutral thresholds
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

print("\nNEUTRAL THRESHOLD EXPERIMENT")
print("=" * 65)

for threshold in thresholds:

    # Start with normal predictions
    y_pred = np.argmax(probabilities, axis=1)

    # Force Neutral when its probability reaches the threshold
    neutral_mask = probabilities[:, 1] >= threshold
    y_pred[neutral_mask] = 1

    # Calculate accuracy
    accuracy = accuracy_score(y_validation, y_pred)

    # Calculate precision, recall and F1 for Neutral
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_validation,
            y_pred,
            labels=[1],
            zero_division=0
        )
    )

    print(f"\nThreshold: {threshold}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Neutral Precision: {precision[0]:.3f}")
    print(f"Neutral Recall:    {recall[0]:.3f}")
    print(f"Neutral F1-score:  {f1[0]:.3f}")