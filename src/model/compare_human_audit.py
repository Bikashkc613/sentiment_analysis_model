import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load manually reviewed file
df = pd.read_csv(
    "data/processed/manual_error_audit_100_reviewed.csv"
)

# Remove rows without human labels
df = df.dropna(subset=["human_sentiment"])
df = df[df["human_sentiment"].str.strip() != ""]

# Normalize labels
for col in ["actual_sentiment", "predicted_sentiment", "human_sentiment"]:
    df[col] = df[col].str.strip().str.title()

labels = ["Negative", "Neutral", "Positive"]

# Compare original labels with human labels
original_accuracy = accuracy_score(
    df["human_sentiment"],
    df["actual_sentiment"]
)

# Compare model predictions with human labels
model_accuracy = accuracy_score(
    df["human_sentiment"],
    df["predicted_sentiment"]
)

print("=" * 60)
print("HUMAN AUDIT RESULTS")
print("=" * 60)

print("Total reviewed:", len(df))

print("\nOriginal dataset vs Human:")
print(f"Agreement: {original_accuracy * 100:.2f}%")

print("\nModel prediction vs Human:")
print(f"Agreement: {model_accuracy * 100:.2f}%")

print("\nModel classification report:")
print(classification_report(
    df["human_sentiment"],
    df["predicted_sentiment"],
    labels=labels,
    zero_division=0
))

# Save disagreements
df["original_matches_human"] = (
    df["actual_sentiment"] == df["human_sentiment"]
)

df["model_matches_human"] = (
    df["predicted_sentiment"] == df["human_sentiment"]
)

df.to_csv(
    "data/processed/human_audit_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
print("File: data/processed/human_audit_comparison.csv")