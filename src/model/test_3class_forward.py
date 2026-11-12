import numpy as np

from src.features.prepare_3class_data import (
    prepare_training_data
)

from src.model.softmax_model import (
    initialize_parameters,
    forward_pass,
    predict
)


# Load actual three-class training data
X_train, y_train, vocabulary = prepare_training_data()

# Initialize model parameters
number_of_features = X_train.shape[1]
number_of_classes = 3

W, b = initialize_parameters(
    number_of_features,
    number_of_classes
)

# Run forward pass
scores, probabilities = forward_pass(
    X_train,
    W,
    b
)

# Generate predictions
predictions = predict(X_train, W, b)


print("Three-class forward pass successful!")

print("\nScores shape:")
print(scores.shape)

print("\nProbabilities shape:")
print(probabilities.shape)

print("\nPredictions shape:")
print(predictions.shape)

print("\nFirst 5 probability rows:")
print(probabilities[:5])

print("\nFirst 5 predictions:")
print(predictions[:5])

print("\nFirst 5 actual labels:")
print(y_train[:5])

print("\nProbability row sums (first 5):")
print(np.sum(probabilities[:5], axis=1))