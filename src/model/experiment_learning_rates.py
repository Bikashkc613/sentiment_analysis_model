
from pathlib import Path

import numpy as np

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


def train_experiment(
    X_train,
    y_train,
    X_validation,
    y_validation,
    learning_rate,
    epochs=50,
    batch_size=256
):

    number_of_features = X_train.shape[1]

    W, b = initialize_parameters(
        number_of_features,
        3
    )

    class_weights = np.array([1.0, 1.5, 1.0])

    best_accuracy = -1
    best_loss = float("inf")
    best_epoch = 0

    best_W = None
    best_b = None

    rng = np.random.default_rng(42)

    for epoch in range(epochs):

        indices = rng.permutation(len(y_train))

        for start in range(0, len(y_train), batch_size):

            batch_indices = indices[start:start + batch_size]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            W, b, _ = train_one_step(
                X_batch,
                y_batch,
                W,
                b,
                learning_rate,
                class_weights
            )

        # Validation evaluation
        _, probabilities = forward_pass(
            X_validation,
            W,
            b
        )

        validation_loss = cross_entropy_loss(
            probabilities,
            y_validation
        )

        predictions = np.argmax(probabilities, axis=1)

        validation_accuracy = np.mean(
            predictions == y_validation
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Accuracy: {validation_accuracy * 100:.2f}% | "
            f"Loss: {validation_loss:.4f}"
        )

        # Save the best model parameters in memory
        if (
            validation_accuracy > best_accuracy
            or (
                validation_accuracy == best_accuracy
                and validation_loss < best_loss
            )
        ):

            best_accuracy = validation_accuracy
            best_loss = validation_loss
            best_epoch = epoch + 1

            best_W = W.copy()
            best_b = b.copy()

    return (
        best_W,
        best_b,
        best_accuracy,
        best_loss,
        best_epoch
    )


if __name__ == "__main__":

    # Load training and validation data
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

    # Selected learning rate
    learning_rate = 0.018

    print("\n" + "=" * 60)
    print(f"TRAINING FINAL BIGRAM MODEL | LR: {learning_rate}")
    print("=" * 60)

    (
        best_W,
        best_b,
        best_accuracy,
        best_loss,
        best_epoch
    ) = train_experiment(
        X_train,
        y_train,
        X_validation,
        y_validation,
        learning_rate=learning_rate,
        epochs=50,
        batch_size=256
    )

    # Project root directory
    project_folder = Path(__file__).resolve().parents[2]

    # Save the best model separately
    model_path = (
        project_folder
        / "models"
        / "sentiment_bigram_model_best.npz"
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        model_path,
        W=best_W,
        b=best_b,
        learning_rate=learning_rate,
        best_epoch=best_epoch,
        validation_accuracy=best_accuracy,
        validation_loss=best_loss
    )

    print("\n" + "=" * 60)
    print("BEST MODEL SAVED SUCCESSFULLY!")
    print("=" * 60)

    print(f"Model path: {model_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Learning rate: {learning_rate}")
    print(f"Validation accuracy: {best_accuracy * 100:.2f}%")
    print(f"Validation loss: {best_loss:.4f}")
    print(f"W shape: {best_W.shape}")
    print(f"b shape: {best_b.shape}")
