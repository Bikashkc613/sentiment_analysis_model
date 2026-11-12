
from pathlib import Path
import json
import numpy as np

from src.features.combined_vectorize import (
    reviews_to_combined_matrix
)

from src.model.predict_with_strategy import (
    predict_with_strategy
)


# Project folder
project_folder = Path(__file__).resolve().parents[2]


# Load the BIGRAM model (15,000 features)
model_path = (
    project_folder
    / "models"
    / "sentiment_bigram_model.npz"
)

model = np.load(model_path)

W = model["W"]
b = model["b"]


# Load unigram vocabulary
with open(
    project_folder / "models" / "vocabulary.json",
    "r",
    encoding="utf-8"
) as file:
    unigram_vocabulary = json.load(file)


# Load bigram vocabulary
with open(
    project_folder / "models" / "bigram_vocabulary.json",
    "r",
    encoding="utf-8"
) as file:
    bigram_vocabulary = json.load(file)


def predict_sentiment(
    review,
    strategy="argmax",
    margin=0.10
):

    X = reviews_to_combined_matrix(
        [review],
        unigram_vocabulary,
        bigram_vocabulary
    )

    from src.model.softmax_model import forward_pass

    _, probabilities = forward_pass(X, W, b)

    prediction = predict_with_strategy(
    X,
    W,
    b,
    strategy=strategy,
    margin=margin
)

    print("\nClass probabilities:")

    print(f"Negative: {probabilities[0, 0]:.4f}")
    print(f"Neutral:  {probabilities[0, 1]:.4f}")
    print(f"Positive: {probabilities[0, 2]:.4f}")

    labels = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    return labels[int(prediction[0])]


if __name__ == "__main__":

    print("Bigram model loaded!")
    print("Weight shape:", W.shape)
    print("Bias shape:", b.shape)

    review = input("\nEnter your movie review: ")

    print("\nNormal prediction:")
    print(
        predict_sentiment(
            review,
            strategy="argmax"
        )
    )

    print("\nNeutral-aware prediction:")
    print(
        predict_sentiment(
            review,
            strategy="neutral_margin",
            margin=0.12
        )
    )
