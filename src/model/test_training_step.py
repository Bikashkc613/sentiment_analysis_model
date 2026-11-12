import numpy as np

from src.features.prepare_3class_data import (
    prepare_training_data
)

from src.model.softmax_model import (
    initialize_parameters,
    forward_pass,
    cross_entropy_loss,
    train_one_step
)


# Load training data
X_train, y_train, vocabulary = prepare_training_data()

# Use a small batch of 256 reviews
X_batch = X_train[:256]
y_batch = y_train[:256]

# Initialize parameters
W, b = initialize_parameters(
    X_train.shape[1],
    3
)

# Calculate loss before learning
scores, probabilities = forward_pass(
    X_batch, W, b
)

loss_before = cross_entropy_loss(
    probabilities, y_batch
)

# Perform one learning step
W, b, training_loss = train_one_step(
    X_batch,
    y_batch,
    W,
    b,
    learning_rate=0.00001
)

# Calculate loss after learning
scores, probabilities = forward_pass(
    X_batch, W, b
)

loss_after = cross_entropy_loss(
    probabilities, y_batch
)

print("First training step completed!")

print("\nLoss before learning:")
print(loss_before)

print("\nLoss returned by training step:")
print(training_loss)

print("\nLoss after learning:")
print(loss_after)

print("\nWeights shape:", W.shape)
print("Bias shape:", b.shape)

print("\nDid loss decrease?")
print(loss_after < loss_before)