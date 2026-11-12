import pandas as pd
from pathlib import Path

# Load test dataset
test_path = Path("data/interim/test_3class.csv")
df = pd.read_csv(test_path)

# Randomly select 50 reviews
sample = df.sample(n=50, random_state=42).copy()

# Keep original row index for later comparison
sample["test_row_index"] = sample.index

results = []

print("\n" + "=" * 70)
print("BLIND HUMAN AUDIT - 50 RANDOM REVIEWS")
print("=" * 70)

print("""
Choose the sentiment based ONLY on the review:

1 = Negative
2 = Neutral
3 = Positive
4 = Uncertain
""")

for number, (_, row) in enumerate(sample.iterrows(), start=1):

    print("\n" + "=" * 70)
    print(f"REVIEW {number}/50")
    print("=" * 70)

    print("\nREVIEW:")
    print(row["review"])

    while True:
        choice = input(
            "\nYour label (1/2/3/4): "
        ).strip()

        labels = {
            "1": "negative",
            "2": "neutral",
            "3": "positive",
            "4": "uncertain"
        }

        if choice in labels:
            break

        print("Invalid choice. Enter 1, 2, 3, or 4.")

    results.append({
        "sample_id": number,
        "test_row_index": row["test_row_index"],
        "review": row["review"],
        "verified_human_label": labels[choice]
    })

    print(f"Saved label: {labels[choice]}")

# Save audit results
output_path = Path("data/processed/blind_audit_50.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

audit_df = pd.DataFrame(results)
audit_df.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print("BLIND AUDIT COMPLETED")
print("=" * 70)

print(f"Total reviews: {len(audit_df)}")
print(f"Saved to: {output_path}")

print("\nYour label counts:")
print(audit_df["verified_human_label"].value_counts())