
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.features.combined_vectorize import reviews_to_combined_matrix
from src.model.softmax_model import forward_pass


# Project folder
project_folder = Path(__file__).resolve().parents[2]


# Step 1: Load blind audit
audit_path = (
    project_folder / "data" / "processed" / "blind_audit_50.csv"
)

audit_df = pd.read_csv(audit_path)

print("Blind audit loaded:", len(audit_df), "reviews")


# Step 2: Load vocabularies
with open(
    project_folder / "models" / "vocabulary.json",
    "r",
    encoding="utf-8"
) as file:
    unigram_vocabulary = json.load(file)

with open(
    project_folder / "models" / "bigram_vocabulary.json",
    "r",
    encoding="utf-8"
) as file:
    bigram_vocabulary = json.load(file)


# Step 3: Load trained model
model_path = (
    project_folder / "models" / "sentiment_bigram_model_best.npz"
)

model = np.load(model_path)

W = model["W"]
b = model["b"]

print("Best model loaded successfully!")


# Step 4: Convert reviews into features
reviews = audit_df["review"].fillna("").tolist()

X = reviews_to_combined_matrix(
    reviews,
    unigram_vocabulary,
    bigram_vocabulary
)


# Step 5: Generate predictions
_, probabilities = forward_pass(X, W, b)

predicted_classes = np.argmax(probabilities, axis=1)

label_names = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

audit_df["model_prediction"] = [
    label_names[int(value)]
    for value in predicted_classes
]


# Step 6: Check unknown vocabulary
audit_df["known_features"] = [
    X.getrow(i).nnz
    for i in range(X.shape[0])
]


# Step 7: Remove uncertain human labels
usable = audit_df[
    audit_df["verified_human_label"].isin(
        ["negative", "neutral", "positive"]
    )
].copy()


# Step 8: Compare human labels and model
y_true = usable["verified_human_label"]
y_pred = usable["model_prediction"]

label_order = ["negative", "neutral", "positive"]

accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 65)
print("BLIND AUDIT COMPARISON")
print("=" * 65)

print("Total reviews:", len(audit_df))
print("Usable reviews:", len(usable))
print("Uncertain reviews:", len(audit_df) - len(usable))

print(f"\nMODEL ACCURACY: {accuracy * 100:.2f}%")


# Step 9: Classification report
print("\nCLASSIFICATION REPORT")
print("-" * 65)

print(
    classification_report(
        y_true,
        y_pred,
        labels=label_order,
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0
    )
)


# Step 10: Confusion matrix
matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=label_order
)

print("\nCONFUSION MATRIX")
print("Rows = Human labels")
print("Columns = Model predictions")
print(matrix)


# Step 11: Prediction distribution
print("\nMODEL PREDICTION DISTRIBUTION")
print(usable["model_prediction"].value_counts())


# Step 12: Save detailed comparison
output_path = (
    project_folder
    / "data"
    / "processed"
    / "blind_audit_50_comparison.csv"
)

usable.to_csv(output_path, index=False)

print("\nComparison saved to:")
print(output_path)
