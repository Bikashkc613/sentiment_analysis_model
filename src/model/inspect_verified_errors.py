
import pandas as pd

df = pd.read_csv(
    "data/processed/final_verified_audit_comparison.csv"
)

errors = df[
    df["verified_human_label"] != df["predicted_sentiment"]
].copy()

print("TOTAL MODEL ERRORS:", len(errors))
print("=" * 70)

for _, row in errors.iterrows():
    print(f"\nSample ID: {row['sample_id']}")
    print("Verified label:", row["verified_human_label"])
    print("Original label:", row["actual_sentiment"])
    print("Model prediction:", row["predicted_sentiment"])
    print("-" * 70)
