
from pathlib import Path
import json
import numpy as np


def load_model():

    # Find project root
    project_folder = Path(__file__).resolve().parents[2]

    models_folder = project_folder / "models"

    # Load weights and bias
    parameters_path = models_folder / "sentiment_model.npz"

    parameters = np.load(parameters_path)

    W = parameters["W"]
    b = parameters["b"]

    # Load vocabulary
    vocabulary_path = models_folder / "vocabulary.json"

    with open(vocabulary_path, "r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    print("Model loaded successfully!")
    print("Weight shape:", W.shape)
    print("Bias shape:", b.shape)
    print("Vocabulary size:", len(vocabulary))

    return W, b, vocabulary


if __name__ == "__main__":

    W, b, vocabulary = load_model()
