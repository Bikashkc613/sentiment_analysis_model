
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD BOTH FILES
# ==========================================

verified = pd.read_csv(
    "data/processed/verified_audit_combined.csv"
)

original = pd.read_csv(
    "data/processed/manual_error_audit_100_corrected.csv"
)

# ==========================================
# 2. SELECT REQUIRED COLUMNS
# ==========================================

verified = verified[
    ["sample_id", "verified_human_label"]
].copy()

original = original[
    ["sample_id", "actual_sentiment", "predicted_sentiment"]
].copy()

# ==========================================
# 3. MERGE USING SAMPLE ID
# ==========================================

df = pd.merge(
    verified,
    original,
    on="sample_id",
    how="inner"
)

print("=" * 60)
print("VERIFIED AUDIT COMPARISON")
print("=" * 60)

print("\nMatched reviews:", len(df))

# ==========================================
# 4. CLEAN LABELS
# ==========================================

columns = [
    "verified_human_label",
    "actual_sentiment",
    "predicted_sentiment"
]

for col in columns:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .str.lower()
    )

# ==========================================
# 5. REMOVE UNCERTAIN AND INVALID LABELS
# ==========================================

valid_labels = ["negative", "neutral", "positive"]

df = df[
    df["verified_human_label"].isin(valid_labels)
    & df["actual_sentiment"].isin(valid_labels)
    & df["predicted_sentiment"].isin(valid_labels)
].copy()

print("Usable verified reviews:", len(df))

# ==========================================
# 6. ORIGINAL DATASET VS VERIFIED LABELS
# ==========================================

original_accuracy = accuracy_score(
    df["verified_human_label"],
    df["actual_sentiment"]
)

print("\n" + "=" * 60)
print("ORIGINAL DATASET vs VERIFIED LABELS")
print("=" * 60)

print(f"Agreement: {original_accuracy * 100:.2f}%")

# ==========================================
# 7. MODEL VS VERIFIED LABELS
# ==========================================

model_accuracy = accuracy_score(
    df["verified_human_label"],
    df["predicted_sentiment"]
)

print("\n" + "=" * 60)
print("MODEL vs VERIFIED LABELS")
print("=" * 60)

print(f"Accuracy: {model_accuracy * 100:.2f}%")

# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        df["verified_human_label"],
        df["predicted_sentiment"],
        labels=valid_labels,
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0
    )
)

# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

print("\nCONFUSION MATRIX")
print("Rows = Verified labels")
print("Columns = Model predictions")

matrix = confusion_matrix(
    df["verified_human_label"],
    df["predicted_sentiment"],
    labels=valid_labels
)

print(matrix)

# ==========================================
# 10. SAVE COMPARISON RESULTS
# ==========================================

df.to_csv(
    "data/processed/final_verified_audit_comparison.csv",
    index=False
)

print("\nSaved:")
print("data/processed/final_verified_audit_comparison.csv")

print("\nAUDIT COMPARISON COMPLETED")
