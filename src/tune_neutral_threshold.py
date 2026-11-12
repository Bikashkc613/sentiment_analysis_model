from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from src.features.combined_vectorize import reviews_to_combined_matrix
from src.model.softmax_model import forward_pass
from src.preprocessing.clean_text import clean_text


# Project paths
PROJECT_FOLDER = Path(__file__).resolve().parents[1]

VALIDATION_PATH = (
    PROJECT_FOLDER / "data" / "interim" / "validation_3class.csv"
)

MODEL_PATH = (
    PROJECT_FOLDER / "models" / "sentiment_bigram_neutral2.npz"
)

UNIGRAM_VOCAB_PATH = PROJECT_FOLDER / "models" / "vocabulary.json"
BIGRAM_VOCAB_PATH = PROJECT_FOLDER / "models" / "bigram_vocabulary.json"

BIGRAM_SCALE = 0.5


def load_validation_data():
    df = pd.read_csv(VALIDATION_PATH)

    # Automatically find review and label columns
    review_column = next(
        (col for col in ["review", "text"] if col in df.columns),
        None
    )

    label_column = next(
        (col for col in ["label", "sentiment", "target"] if col in df.columns),
        None
    )

    if review_column is None or label_column is None:
        raise ValueError(
            f"Could not find review/label columns. Found: {list(df.columns)}"
        )

    reviews = df[review_column].fillna("").astype(str).tolist()
    labels = df[label_column].to_numpy()

    # Convert text labels to numeric labels if needed
    if labels.dtype.kind in {"U", "S", "O"}:
        label_mapping = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }

        labels = np.array([
            label_mapping[str(label).strip().lower()]
            for label in labels
        ])

    labels = labels.astype(int)

    return reviews, labels


def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0
    )

    macro_f1 = np.mean(f1)

    return {
        "accuracy": accuracy,
        "neutral_precision": precision[1],
        "neutral_recall": recall[1],
        "neutral_f1": f1[1],
        "macro_f1": macro_f1,
    }


def make_predictions(probabilities, neutral_threshold):
    predictions = []

    for row in probabilities:
        negative_probability = row[0]
        neutral_probability = row[1]
        positive_probability = row[2]

        if neutral_probability >= neutral_threshold:
            predictions.append(1)
        elif negative_probability >= positive_probability:
            predictions.append(0)
        else:
            predictions.append(2)

    return np.array(predictions)


def print_result(name, result):
    print(f"\n{name}")
    print("-" * 55)
    print(f"Accuracy:          {result['accuracy'] * 100:.2f}%")
    print(f"Neutral precision: {result['neutral_precision'] * 100:.2f}%")
    print(f"Neutral recall:    {result['neutral_recall'] * 100:.2f}%")
    print(f"Neutral F1:        {result['neutral_f1'] * 100:.2f}%")
    print(f"Macro F1:          {result['macro_f1'] * 100:.2f}%")


def main():
    print("Loading validation data...")

    reviews, y_validation = load_validation_data()

    print(f"Validation reviews: {len(reviews)}")

    # Load vocabularies
    with open(UNIGRAM_VOCAB_PATH, "r", encoding="utf-8") as file:
        unigram_vocabulary = json.load(file)

    with open(BIGRAM_VOCAB_PATH, "r", encoding="utf-8") as file:
        bigram_vocabulary = json.load(file)

    # Clean reviews
    cleaned_reviews = [clean_text(review) for review in reviews]

    # Convert reviews into model features
    X_validation = reviews_to_combined_matrix(
        cleaned_reviews,
        unigram_vocabulary,
        bigram_vocabulary
    )

    # Apply the same bigram scaling used during training
    unigram_count = len(unigram_vocabulary)

    X_validation = X_validation.tocsr()
    X_validation[:, unigram_count:] *= BIGRAM_SCALE

    # Load model
    model = np.load(MODEL_PATH)

    W = model["W"]
    b = model["b"]

    print(f"Model features: {W.shape[0]}")
    print(f"Validation features: {X_validation.shape[1]}")

    if X_validation.shape[1] != W.shape[0]:
        raise ValueError("Feature count does not match model dimensions!")

    # Get probabilities
    _, probabilities = forward_pass(X_validation, W, b)

    # Normal argmax baseline
    baseline_predictions = np.argmax(probabilities, axis=1)
    baseline_result = evaluate_predictions(
        y_validation,
        baseline_predictions
    )

    print_result("BASELINE (normal argmax)", baseline_result)

    # Test different Neutral thresholds
    results = []

    for threshold in np.arange(0.20, 0.61, 0.01):
        predictions = make_predictions(probabilities, threshold)
        result = evaluate_predictions(y_validation, predictions)

        results.append((threshold, result))

    # Best threshold by Neutral F1
    best_neutral = max(
        results,
        key=lambda item: (
            item[1]["neutral_f1"],
            item[1]["macro_f1"]
        )
    )

    # Best threshold by Macro F1
    best_macro = max(
        results,
        key=lambda item: item[1]["macro_f1"]
    )

    print("\n\nTHRESHOLD RESULTS")
    print("=" * 85)
    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Neutral P':<12}"
        f"{'Neutral R':<12}"
        f"{'Neutral F1':<12}"
        f"{'Macro F1':<12}"
    )
    print("-" * 85)

    for threshold, result in results:
        print(
            f"{threshold:<12.2f}"
            f"{result['accuracy'] * 100:<12.2f}"
            f"{result['neutral_precision'] * 100:<12.2f}"
            f"{result['neutral_recall'] * 100:<12.2f}"
            f"{result['neutral_f1'] * 100:<12.2f}"
            f"{result['macro_f1'] * 100:<12.2f}"
        )

    print_result(
        f"BEST NEUTRAL F1 THRESHOLD: {best_neutral[0]:.2f}",
        best_neutral[1]
    )

    print_result(
        f"BEST MACRO F1 THRESHOLD: {best_macro[0]:.2f}",
        best_macro[1]
    )

    print("\nThreshold tuning complete.")
    print("No model files were changed.")


if __name__ == "__main__":
    main()