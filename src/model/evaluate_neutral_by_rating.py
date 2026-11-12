
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

from src.model.load_model import load_model
from src.model.softmax_model import forward_pass
from src.features.vectorize import reviews_to_matrix


# Project folder
project_folder = Path(__file__).resolve().parents[2]

# Load model
W, b, vocabulary = load_model()

# Load original dataset
data_path = project_folder / "data" / "raw" / "Data.csv"

df = pd.read_csv(data_path)

# Apply the same text cleaning used during training
from src.preprocessing.clean_text import clean_text

df["review"] = df["review"].apply(clean_text)

# Remove empty reviews
df = df[df["review"].str.len() > 0].copy()

# Remove conflicting reviews
rating_counts = df.groupby("review")["rating"].nunique()

conflicting_reviews = rating_counts[
    rating_counts > 1
].index

df = df[
    ~df["review"].isin(conflicting_reviews)
].copy()

# Remove duplicate reviews
df = df.drop_duplicates(subset=["review"])

# Keep only Neutral reviews
neutral_df = df[
    df["rating"].between(4, 7)
].copy()

# Vectorize using the model's existing vocabulary
X_neutral = reviews_to_matrix(
    neutral_df["review"],
    vocabulary
)

# Make predictions
scores, probabilities = forward_pass(
    X_neutral,
    W,
    b
)

predictions = np.argmax(probabilities, axis=1)

label_mapping = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

neutral_df["predicted"] = [
    label_mapping[prediction]
    for prediction in predictions
]

# Evaluate each original rating separately
print("\nNeutral performance by original rating:")

for rating in [4, 5, 6, 7]:

    rating_df = neutral_df[
        neutral_df["rating"] == rating
    ]

    correct = (
        rating_df["predicted"] == "neutral"
    ).sum()

    total = len(rating_df)

    accuracy = correct / total * 100

    print(f"\nRating: {rating}")
    print("Total reviews:", total)
    print("Predicted Neutral:", correct)
    print("Neutral recall:", round(accuracy, 2), "%")

    print("\nPrediction distribution:")
    print(rating_df["predicted"].value_counts())

# Overall report
print("\nOverall Neutral classification report:")

print(
    classification_report(
        ["neutral"] * len(neutral_df),
        neutral_df["predicted"],
        labels=["negative", "neutral", "positive"],
        zero_division=0
    )
)
