
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.features.prepare_3class_data_with_bigrams import (
    prepare_training_data,
    prepare_validation_data
)

from src.model.softmax_model import forward_pass


# ==========================================
# SETTINGS
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

MODELS = {
    "Previous model (Neutral weight 1.5)": (
        "sentiment_bigram_model_scaled.npz"
    ),
    "New model (Neutral weight 2.0)": (
        "sentiment_bigram_neutral2.npz"
    )
}

BIGRAM_SCALE = 0.5

LABELS = ["Negative", "Neutral", "Positive"]


# ==========================================
# SCALE BIGRAM FEATURES
# ==========================================

def scale_bigram_features(X, unigram_count):

    X_unigram = X[:, :unigram_count]
    X_bigram = X[:, unigram_count:]

    X_bigram = X_bigram * BIGRAM_SCALE

    from scipy.sparse import hstack

    return hstack(
        [X_unigram, X_bigram],
        format="csr"
    )


# ==========================================
# EVALUATE ONE MODEL
# ==========================================

def evaluate_model(model_name, model_filename, X_validation, y_validation):

    model_path = PROJECT_FOLDER / "models" / model_filename

    if not model_path.exists():
        print(f"\nModel not found: {model_path}")
        return

    saved_model = np.load(model_path)

    W = saved_model["W"]
    b = saved_model["b"]

    # Check saved bigram scale
    saved_scale = float(saved_model["bigram_scale"])

    if saved_scale != BIGRAM_SCALE:
        raise ValueError(
            f"Bigram scale mismatch in {model_filename}"
        )

    # Get probabilities
    _, probabilities = forward_pass(
        X_validation,
        W,
        b
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    print("\n" + "=" * 65)
    print(model_name)
    print("=" * 65)

    print(f"Validation accuracy: {accuracy * 100:.2f}%")

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_validation,
            predictions,
            labels=[0, 1, 2],
            target_names=LABELS,
            digits=4,
            zero_division=0
        )
    )

    print("CONFUSION MATRIX")
    print(
        confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1, 2]
        )
    )


# ==========================================
# MAIN
# ==========================================

def main():

    print("Preparing validation data...")

    (
        X_train,
        y_train,
        unigram_vocabulary,
        bigram_vocabulary
    ) = prepare_training_data()

    X_validation, y_validation = prepare_validation_data(
        unigram_vocabulary,
        bigram_vocabulary
    )

    unigram_count = len(unigram_vocabulary)

    X_validation = scale_bigram_features(
        X_validation,
        unigram_count
    )

    print("\nValidation samples:", X_validation.shape[0])

    # Evaluate both models
    for model_name, model_filename in MODELS.items():

        evaluate_model(
            model_name,
            model_filename,
            X_validation,
            y_validation
        )


if __name__ == "__main__":
    main()
