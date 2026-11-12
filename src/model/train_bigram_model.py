
from pathlib import Path

import numpy as np
from scipy.sparse import hstack

from src.features.prepare_3class_data_with_bigrams import (
    prepare_training_data,
    prepare_validation_data
)

from src.model.softmax_model import (
    initialize_parameters,
    forward_pass,
    cross_entropy_loss,
    train_one_step
)


# ==========================================
# PROJECT SETTINGS
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parents[2]

BIGRAM_SCALE = 0.5

EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.018

# Class order:
# 0 = Negative
# 1 = Neutral
# 2 = Positive

CLASS_WEIGHTS = np.array([1.0, 2.0, 1.0])

# Save as a separate model
MODEL_PATH = (
    PROJECT_FOLDER
    / "models"
    / "sentiment_bigram_neutral2.npz"
)


# ==========================================
# SCALE BIGRAM FEATURES
# ==========================================

def scale_bigram_features(X, unigram_count):

    X_unigram = X[:, :unigram_count]
    X_bigram = X[:, unigram_count:]

    X_bigram = X_bigram * BIGRAM_SCALE

    return hstack(
        [X_unigram, X_bigram],
        format="csr"
    )


# ==========================================
# TRAIN MODEL
# ==========================================

def train_bigram_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE
):

    number_of_features = X_train.shape[1]
    number_of_classes = 3

    rng = np.random.default_rng(42)

    W, b = initialize_parameters(
        number_of_features,
        number_of_classes
    )

    best_loss = float("inf")
    best_W = None
    best_b = None
    best_epoch = 0

    print("\nStarting Neutral weight experiment...")
    print("Neutral class weight:", CLASS_WEIGHTS[1])
    print("Bigram scale:", BIGRAM_SCALE)
    print("Learning rate:", learning_rate)

    for epoch in range(epochs):

        # Shuffle training data
        indices = rng.permutation(X_train.shape[0])

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Mini-batch training
        for start in range(0, X_train.shape[0], batch_size):

            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # train_one_step returns:
            # W, b, loss

            W, b, _ = train_one_step(
                X_batch,
                y_batch,
                W,
                b,
                learning_rate,
                CLASS_WEIGHTS
            )

        # ==================================
        # VALIDATION
        # ==================================

        # forward_pass returns:
        # scores, probabilities

        _, validation_probabilities = forward_pass(
            X_validation,
            W,
            b
        )

        # Calculate validation loss
        validation_loss = cross_entropy_loss(
            validation_probabilities,
            y_validation
        )

        # Predictions
        predictions = np.argmax(
            validation_probabilities,
            axis=1
        )

        accuracy = np.mean(
            predictions == y_validation
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {validation_loss:.6f} "
            f"| Accuracy: {accuracy * 100:.2f}%"
        )

        # Save best model
        if validation_loss < best_loss:

            best_loss = validation_loss
            best_W = W.copy()
            best_b = b.copy()
            best_epoch = epoch + 1

    print("\nTraining completed!")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_loss)

    return best_W, best_b, best_epoch, best_loss


# ==========================================
# MAIN
# ==========================================

def main():

    print("Preparing training data...")

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

    print("Training samples:", X_train.shape[0])
    print("Validation samples:", X_validation.shape[0])

    print("Unigram features:", len(unigram_vocabulary))
    print("Bigram features:", len(bigram_vocabulary))

    unigram_count = len(unigram_vocabulary)

    # Scale bigram features
    X_train = scale_bigram_features(
        X_train,
        unigram_count
    )

    X_validation = scale_bigram_features(
        X_validation,
        unigram_count
    )

    print("Total features:", X_train.shape[1])

    # Train model
    best_W, best_b, best_epoch, best_loss = train_bigram_model(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    # Save new model
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        MODEL_PATH,
        W=best_W,
        b=best_b,
        bigram_scale=BIGRAM_SCALE,
        best_epoch=best_epoch,
        best_validation_loss=best_loss
    )

    print("\nNew model saved successfully!")
    print("Model path:", MODEL_PATH)


if __name__ == "__main__":
    main()
