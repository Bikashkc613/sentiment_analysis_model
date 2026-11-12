
import pandas as pd

file_path = "data/processed/corrected_audit_disagreements.csv"

df = pd.read_csv(file_path)

# Select first 10 disagreements
sample = df.iloc[10:20].copy()

print("=" * 80)
print("MANUAL AUDIT - FIRST 10 REVIEWS")
print("=" * 80)

for _, row in sample.iterrows():
    print(f"\nSample ID: {row['sample_id']}")
    print("-" * 80)
    print("REVIEW:")
    print(row["review"])

    print("\nOriginal label:", row["actual_sentiment"])
    print("Model prediction:", row["predicted_sentiment"])
    print("AI-assisted audit label:", row["human_sentiment"])

    print("\nChoose your label:")
    print("1 = Negative")
    print("2 = Neutral")
    print("3 = Positive")
    print("4 = Uncertain")

    while True:
        choice = input("Your choice (1-4): ").strip()

        labels = {
            "1": "negative",
            "2": "neutral",
            "3": "positive",
            "4": "uncertain"
        }

        if choice in labels:
            break

        print("Invalid choice. Enter 1, 2, 3, or 4.")

    sample.loc[row.name, "verified_human_label"] = labels[choice]

# Save verified results
output_path = "data/processed/verified_audit_first_10.csv"
sample.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print("AUDIT COMPLETED")
print("Saved:", output_path)
print("=" * 80)
