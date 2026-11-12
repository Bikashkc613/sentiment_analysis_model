
import numpy as np



def softmax(scores):
    """
    Convert raw scores into probabilities.

    Works with:
    - One review: shape (3,)
    - Multiple reviews: shape (number_of_reviews, 3)
    """

    # Find the maximum score in each row
    shifted_scores = scores - np.max(
        scores,
        axis=1,
        keepdims=True
    ) if scores.ndim == 2 else scores - np.max(scores)

    # Calculate exponentials
    exp_scores = np.exp(shifted_scores)

    # Calculate probabilities
    probabilities = exp_scores / np.sum(
        exp_scores,
        axis=1,
        keepdims=True
    ) if scores.ndim == 2 else exp_scores / np.sum(exp_scores)

    return probabilities


def forward_pass(X, W, b):
    """
    Calculate raw scores and probabilities.

    X = input features
    W = weights
    b = bias
    """

    # Calculate raw scores
    scores = X @ W + b

    # Convert scores into probabilities
    probabilities = softmax(scores)

    return scores, probabilities

def predict(X, W, b):
    """
    Predict the sentiment class for each review.
    """

    scores, probabilities = forward_pass(X, W, b)

    predictions = np.argmax(probabilities, axis=1)

    return predictions

def initialize_parameters(number_of_features, number_of_classes):
    """
    Create initial weights and biases.
    """

    W = np.zeros((number_of_features, number_of_classes))
    b = np.zeros(number_of_classes)

    return W, b

def cross_entropy_loss(probabilities, y_true):
    """
    Calculate average cross-entropy loss.

    probabilities: predicted probabilities
    y_true: actual class labels
    """

    number_of_samples = len(y_true)

    correct_class_probabilities = probabilities[
        np.arange(number_of_samples),
        y_true
    ]

    loss = -np.mean(
        np.log(correct_class_probabilities + 1e-15)
    )

    return loss


def compute_gradients(
    X,
    probabilities,
    y_true,
    class_weights=None
):
    number_of_samples = X.shape[0]

    # Calculate prediction errors
    d_scores = probabilities.copy()

    d_scores[
        np.arange(number_of_samples),
        y_true
    ] -= 1

    # Apply class weights
    if class_weights is not None:

        sample_weights = class_weights[y_true]

        d_scores *= sample_weights[:, np.newaxis]

        # Normalize by the sum of sample weights
        d_scores /= np.sum(sample_weights)

    else:
        d_scores /= number_of_samples

    # Calculate gradients
    dW = X.T @ d_scores

    db = np.sum(d_scores, axis=0)

    return dW, db



def update_parameters(W, b, dW, db, learning_rate):
    """
    Update weights and bias using gradients.
    """

    W = W - learning_rate * dW
    b = b - learning_rate * db

    return W, b


def train_one_step(
    X,
    y_true,
    W,
    b,
    learning_rate,
    class_weights=None
):

    scores, probabilities = forward_pass(X, W, b)

    loss = cross_entropy_loss(
        probabilities,
        y_true
    )

    dW, db = compute_gradients(
        X,
        probabilities,
        y_true,
        class_weights
    )

    W, b = update_parameters(
        W,
        b,
        dW,
        db,
        learning_rate
    )

    return W, b, loss
