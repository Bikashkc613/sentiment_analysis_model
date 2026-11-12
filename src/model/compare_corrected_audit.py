
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load corrected audit file
file_path = "data/processed/manual_error_audit_100_corrected.csv"
df = pd.read_csv(file_path)

print("=" * 65)
print("CORRECTED MANUAL AUDIT COMPARISON")
print("=" * 65)

# Normalize labels
for column in ["actual_sentiment", "predicted_sentiment", "human_sentiment"]:
    df[column] = df[column].astype(str).str.strip().str.lower()

# Exclude uncertain human labels
valid = df[
    df["human_sentiment"].isin(["negative", "neutral", "positive"])
].copy()

uncertain_count = len(df) - len(valid)

print(f"\nTotal audit samples: {len(df)}")
print(f"Usable labels: {len(valid)}")
print(f"Uncertain labels excluded: {uncertain_count}")

labels = ["negative", "neutral", "positive"]

# Compare original dataset labels with audit labels
print("\n" + "=" * 65)
print("1. ORIGINAL DATASET vs AUDIT LABELS")
print("=" * 65)

original_accuracy = accuracy_score(
    valid["actual_sentiment"],
    valid["human_sentiment"]
)

print(f"Agreement: {original_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    valid["human_sentiment"],
    valid["actual_sentiment"],
    labels=labels,
    zero_division=0
))

# Compare model predictions with audit labels
print("\n" + "=" * 65)
print("2. MODEL PREDICTIONS vs AUDIT LABELS")
print("=" * 65)

model_accuracy = accuracy_score(
    valid["human_sentiment"],
    valid["predicted_sentiment"]
)

print(f"Agreement: {model_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    valid["human_sentiment"],
    valid["predicted_sentiment"],
    labels=labels,
    zero_division=0
))

print("\nConfusion Matrix:")
print("Rows = Audit labels | Columns = Model predictions")
print(labels)

print(confusion_matrix(
    valid["human_sentiment"],
    valid["predicted_sentiment"],
    labels=labels
))

# Save rows where model disagrees with audit labels
disagreements = valid[
    valid["human_sentiment"] != valid["predicted_sentiment"]
].copy()

output_path = "data/processed/corrected_audit_disagreements.csv"
disagreements.to_csv(output_path, index=False)

print("\n" + "=" * 65)
print("AUDIT SUMMARY")
print("=" * 65)

print(f"Model agreement: {model_accuracy * 100:.2f}%")
print(f"Model disagreements: {len(disagreements)}")
print(f"Disagreement file saved: {output_path}")

print("\nIMPORTANT:")
print("These audit labels are AI-assisted preliminary labels.")
print("They need human confirmation before being treated as ground truth.")
