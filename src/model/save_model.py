
from pathlib import Path
import json
import numpy as np


def save_model(W, b, vocabulary):

    # Find the project root folder
    project_folder = Path(__file__).resolve().parents[2]

    # Create models folder if it doesn't exist
    models_folder = project_folder / "models"
    models_folder.mkdir(exist_ok=True)

    # Save weights and bias
    parameters_path = models_folder / "sentiment_model.npz"

    np.savez_compressed(
        parameters_path,
        W=W,
        b=b
    )

    # Save vocabulary as JSON
    vocabulary_path = models_folder / "vocabulary.json"

    with open(vocabulary_path, "w", encoding="utf-8") as file:
        json.dump(vocabulary, file)

    print("\nModel saved successfully!")
    print("Parameters:", parameters_path)
    print("Vocabulary:", vocabulary_path)
