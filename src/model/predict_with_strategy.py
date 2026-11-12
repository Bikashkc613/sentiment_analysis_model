import numpy as np

from src.model.softmax_model import forward_pass


def predict_with_strategy(
    X,
    W,
    b,
    strategy="argmax",
    margin=0.10
):
    """
    Predict sentiment using different decision strategies.

    Classes:
    0 = Negative
    1 = Neutral
    2 = Positive
    """

    _, probabilities = forward_pass(X, W, b)

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    if strategy == "argmax":
        return predictions

    elif strategy == "neutral_margin":

        strongest_other = np.maximum(
            probabilities[:, 0],
            probabilities[:, 2]
        )

        neutral_mask = (
            probabilities[:, 1]
            >= strongest_other - margin
        )

        predictions[neutral_mask] = 1

        return predictions

    else:
        raise ValueError(
            "Unknown strategy. Use 'argmax' or 'neutral_margin'."
        )