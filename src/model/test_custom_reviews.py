
import json
from pathlib import Path

import numpy as np

from src.features.combined_vectorize import reviews_to_combined_matrix
from src.model.softmax_model import forward_pass


# Project folder
project_folder = Path(__file__).resolve().parents[2]


# Step 1: Load vocabularies
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


# Step 2: Load trained model
model_path = (
    project_folder / "models" / "sentiment_bigram_model_best.npz"
)

model = np.load(model_path)

W = model["W"]
b = model["b"]

print("Best model loaded successfully!")


# Step 3: Sentiment labels
sentiment_labels = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# Step 4: Custom reviews
custom_reviews = [
    "This movie was absolutely wonderful and amazing!",
    "The movie was boring, terrible and disappointing.",
    "The movie was okay and nothing special.",
    "The acting was excellent and the story was beautiful.",
    "The plot was weak and the acting was awful.",
    "Some scenes were good but others were boring.",
    "The movie was average, neither good nor bad.",
    "I loved the performances but disliked the ending.",
    "qwerty xyz abc randomwords",
    "good",
    "bad",
    "okay"
]


# Step 5: Test each review
print("\n" + "=" * 70)
print("CUSTOM REVIEW TESTING")
print("=" * 70)


for index, review in enumerate(custom_reviews, start=1):

    # Word count
    word_count = len(review.split())

    # Convert review to numerical features
    X = reviews_to_combined_matrix(
        [review],
        unigram_vocabulary,
        bigram_vocabulary
    )

    # Count known features
    known_features = X.nnz

    print(f"\nTEST {index}")
    print("-" * 70)

    print("Review:", review)
    print("Word count:", word_count)
    print("Known features:", known_features)

    # Skip reviews with no known features
    if known_features == 0:
        print("Prediction: Cannot predict reliably.")
        continue

    # Calculate probabilities
    _, probabilities = forward_pass(X, W, b)

    class_probabilities = probabilities[0]

    # Get prediction
    predicted_class = int(np.argmax(class_probabilities))

    sentiment = sentiment_labels[predicted_class]

    confidence = float(
        class_probabilities[predicted_class] * 100
    )

    # Calculate margin
    sorted_probabilities = np.sort(class_probabilities)[::-1]

    margin = float(
        (sorted_probabilities[0] - sorted_probabilities[1]) * 100
    )

    # Display all probabilities
    print("\nProbabilities:")

    for class_id, class_name in sentiment_labels.items():

        percentage = class_probabilities[class_id] * 100

        print(f"{class_name:<10}: {percentage:.2f}%")

    print("\nPredicted sentiment:", sentiment)
    print(f"Confidence: {confidence:.2f}%")
    print(f"Prediction margin: {margin:.2f}%")

    # Warnings
    if word_count < 3:
        print("Warning: Very short review.")

    if confidence < 50:
        print("Warning: Low confidence.")

    if margin < 10:
        print("Warning: Close prediction.")


print("\n" + "=" * 70)
print("CUSTOM REVIEW TESTING COMPLETED")
print("=" * 70)
