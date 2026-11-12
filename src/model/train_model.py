
import numpy as np

from src.model.save_model import save_model

from src.features.prepare_3class_data import (
    prepare_training_data,
    prepare_validation_data
)

from src.model.softmax_model import (
    initialize_parameters,
    forward_pass,
    cross_entropy_loss,
    train_one_step
)


def train_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    epochs=40,
    batch_size=256,
    learning_rate=0.001
):

    # Initialize model parameters
    number_of_features = X_train.shape[1]
    number_of_classes = 3

    W, b = initialize_parameters(
        number_of_features,
        number_of_classes
    )

    # Give more importance to Neutral examples
    class_weights = np.array([
        1.0,  # Negative
        1.5,  # Neutral
        1.0   # Positive
    ])

    # Calculate initial training loss
    _, probabilities = forward_pass(X_train, W, b)

    initial_loss = cross_entropy_loss(
        probabilities,
        y_train
    )

    print("Initial training loss:", initial_loss)

    # Calculate initial validation loss
    _, validation_probabilities = forward_pass(
        X_validation, W, b
    )

    initial_validation_loss = cross_entropy_loss(
        validation_probabilities,
        y_validation
    )

    print("Initial validation loss:", initial_validation_loss)

    # Track the best model
    best_validation_loss = float("inf")
    best_W = None
    best_b = None

    # Random number generator
    rng = np.random.default_rng(42)

    # Training loop
    for epoch in range(epochs):

        # Shuffle training examples
        indices = rng.permutation(len(y_train))

        # Process mini-batches
        for start in range(0, len(y_train), batch_size):

            end = start + batch_size

            batch_indices = indices[start:end]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Update model weights
            W, b, _ = train_one_step(
                X_batch,
                y_batch,
                W,
                b,
                learning_rate,
                class_weights
            )

        # Calculate training loss
        _, probabilities = forward_pass(X_train, W, b)

        training_loss = cross_entropy_loss(
            probabilities,
            y_train
        )

        # Calculate validation loss
        _, validation_probabilities = forward_pass(
            X_validation,
            W,
            b
        )

        validation_loss = cross_entropy_loss(
            validation_probabilities,
            y_validation
        )

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("Training loss:", training_loss)
        print("Validation loss:", validation_loss)

        # Save parameters when validation loss improves
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            best_W = W.copy()
            best_b = b.copy()

            print("New best model found!")

    print("\nBest validation loss:", best_validation_loss)

    return best_W, best_b


if __name__ == "__main__":

    # Load training data
    X_train, y_train, vocabulary = prepare_training_data()

    # Load validation data using training vocabulary
    X_validation, y_validation = prepare_validation_data(
        vocabulary
    )

    print("\nTraining data loaded!")
    print("Training reviews:", X_train.shape[0])

    print("\nValidation data loaded!")
    print("Validation reviews:", X_validation.shape[0])

    # Train model
    W, b = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        epochs=40,
        batch_size=256,
        learning_rate=0.001
    )

    print("\nTraining completed!")

    print("Best weight shape:", W.shape)
    print("Best bias shape:", b.shape)

    # Save the best model
    save_model(W, b, vocabulary)

    print("\nBest model saved successfully!")
