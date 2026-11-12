
from pathlib import Path
import json

import numpy as np

from src.preprocessing.clean_text import clean_text
from src.features.combined_vectorize import reviews_to_combined_matrix
from src.model.softmax_model import forward_pass


# ==========================================
# PROJECT SETTINGS
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MODEL_FOLDER = PROJECT_FOLDER / "models"

MODEL_PATH = MODEL_FOLDER / "sentiment_bigram_neutral2.npz"

UNIGRAM_VOCAB_PATH = MODEL_FOLDER / "vocabulary.json"
BIGRAM_VOCAB_PATH = MODEL_FOLDER / "bigram_vocabulary.json"

# Bigram scaling used during training
BIGRAM_SCALE = 0.5

# Neutral decision threshold
NEUTRAL_THRESHOLD = 0.34

# Class labels
sentiment_labels = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# ==========================================
# LOAD MODEL AND VOCABULARIES
# ==========================================

def load_model():

    # Load model weights
    model = np.load(MODEL_PATH, allow_pickle=True)

    W = model["W"]
    b = model["b"]

    # Load unigram vocabulary
    with open(UNIGRAM_VOCAB_PATH, "r", encoding="utf-8") as file:
        unigram_vocabulary = json.load(file)

    # Load bigram vocabulary
    with open(BIGRAM_VOCAB_PATH, "r", encoding="utf-8") as file:
        bigram_vocabulary = json.load(file)

    model.close()

    print("Sentiment model loaded successfully!")

    return W, b, unigram_vocabulary, bigram_vocabulary


W, b, unigram_vocabulary, bigram_vocabulary = load_model()


# ==========================================
# PREDICT SENTIMENT
# ==========================================

def predict_sentiment(review):

    # Step 1: Clean review
    cleaned_review = clean_text(review)

    if not cleaned_review.strip():
        return None, None, None, None, 0

    # Step 2: Convert review into feature matrix
    X = reviews_to_combined_matrix(
        [cleaned_review],
        unigram_vocabulary,
        bigram_vocabulary
    )

    # Step 3: Count known features
    known_features = X.nnz

    if known_features == 0:
        return None, None, None, None, 0

    # Step 4: Apply bigram scaling
    unigram_size = len(unigram_vocabulary)
    total_features = X.shape[1]

    feature_scales = np.ones(total_features)
    feature_scales[unigram_size:] = BIGRAM_SCALE

    X = X.multiply(feature_scales)

    # Step 5: Calculate probabilities
    _, probabilities = forward_pass(X, W, b)

    class_probabilities = probabilities[0]

    # Step 6: Apply Neutral threshold
    if class_probabilities[1] >= NEUTRAL_THRESHOLD:

        predicted_class = 1

    else:

        # Choose Negative or Positive
        non_neutral_classes = [0, 2]

        predicted_class = non_neutral_classes[
            np.argmax(class_probabilities[non_neutral_classes])
        ]

    # Step 7: Get prediction details
    sentiment = sentiment_labels[predicted_class]

    confidence = float(class_probabilities[predicted_class])

    other_probabilities = np.delete(
        class_probabilities,
        predicted_class
    )

    margin = confidence - float(np.max(other_probabilities))

    return (
        sentiment,
        confidence,
        class_probabilities,
        margin,
        known_features
    )


# ==========================================
# INTERACTIVE CLI
# ==========================================

def run_cli():

    print("\n")
    print("MOVIE SENTIMENT ANALYSIS")
    print("=" * 45)

    print("Neutral threshold:", NEUTRAL_THRESHOLD)

    while True:

        review = input(
            "\nEnter movie review (or 'exit' to quit): "
        )

        if review.lower().strip() == "exit":
            print("\nExiting sentiment analyzer.")
            break

        if not review.strip():
            print("Please enter a review.")
            continue

        (
            sentiment,
            confidence,
            probabilities,
            margin,
            known_features
        ) = predict_sentiment(review)

        # No known features
        if sentiment is None:

            print("\nUnable to predict sentiment.")
            print("No known vocabulary features found.")
            continue

        # Show known features
        print("\nKnown features found:", known_features)

        # Show probabilities
        print("\nSENTIMENT PROBABILITIES")
        print("-" * 40)

        print(f"Negative  : {probabilities[0] * 100:.2f}%")
        print(f"Neutral   : {probabilities[1] * 100:.2f}%")
        print(f"Positive  : {probabilities[2] * 100:.2f}%")

        # Show prediction
        print("\n" + "-" * 40)

        print("Predicted sentiment:", sentiment)
        print(f"Confidence: {confidence * 100:.2f}%")
        print(f"Prediction margin: {margin * 100:.2f}%")

        # Warnings
        if confidence < 0.45:
            print("Warning: Model has low confidence.")

        if margin < 0.05:
            print("Warning: Close prediction. Another class has a similar probability.")

        if known_features < 3:
            print("Warning: Very few known features found.")

        print("-" * 40)


# ==========================================
# RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    run_cli()
