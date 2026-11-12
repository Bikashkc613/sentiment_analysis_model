
import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parents[2]

file_path = (
    project_folder
    / "data"
    / "processed"
    / "blind_audit_50_comparison.csv"
)

df = pd.read_csv(file_path)

# Find incorrect predictions
errors = df[
    df["verified_human_label"] != df["model_prediction"]
].copy()

print("=" * 70)
print("BLIND AUDIT - MODEL ERRORS")
print("=" * 70)

print("Total errors:", len(errors))

# Show first 10 errors
for _, row in errors.head(10).iterrows():

    print("\n" + "=" * 70)
    print("Sample ID:", row["sample_id"])
    print("Human label:", row["verified_human_label"])
    print("Model prediction:", row["model_prediction"])
    print("Known features:", row["known_features"])

    print("\nREVIEW:")
    print(row["review"])

print("\n" + "=" * 70)
print("Showing first 10 errors out of", len(errors))
