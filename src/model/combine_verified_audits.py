import pandas as pd

# First batch: labels recovered from our conversation
batch_1 = pd.DataFrame({
    "sample_id": [1, 2, 3, 5, 6, 8, 10, 11, 13, 14],
    "verified_human_label": [
        "neutral",
        "neutral",
        "negative",
        "neutral",
        "positive",
        "uncertain",
        "uncertain",
        "neutral",
        "neutral",
        "negative"
    ]
})

# Second batch
batch_2 = pd.read_csv(
    "data/processed/verified_audit_batch_2.csv"
)

# Combine both batches
combined = pd.concat(
    [batch_1, batch_2],
    ignore_index=True
)

combined = combined.sort_values("sample_id")

# Save combined labels
combined.to_csv(
    "data/processed/verified_audit_combined.csv",
    index=False
)

print("AUDIT BATCHES COMBINED")
print("Total reviews:", len(combined))
print("\nLabel counts:")
print(combined["verified_human_label"].value_counts())

print("\nSaved: data/processed/verified_audit_combined.csv")