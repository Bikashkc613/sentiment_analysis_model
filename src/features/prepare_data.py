
from pathlib import Path
import pandas as pd
import numpy as np

from src.features.vocabulary import build_vocabulary
from src.features.vectorize import review_to_vector


def prepare_training_data():

    # Find project folder
    project_folder = Path(__file__).resolve().parents[2]

    # Load training dataset
    train_path = project_folder / "data" / "interim" / "train.csv"
    train_df = pd.read_csv(train_path)

    # Build vocabulary using training reviews
    vocabulary = build_vocabulary(
        train_df["review"],
        max_words=5000
    )

    # Convert reviews into numerical vectors
    X_train = np.array(
        [
            review_to_vector(review, vocabulary)
            for review in train_df["review"]
        ],
        dtype=np.float32
    )

    # Convert sentiment labels into numbers
    label_mapping = {
        "negative": 0,
        "positive": 1
    }

    y_train = train_df["sentiment"].map(
        label_mapping
    ).to_numpy(dtype=np.int64)

    return X_train, y_train, vocabulary


def prepare_testing_data(vocabulary):

    project_folder = Path(__file__).resolve().parents[2]

    # Load testing dataset
    test_path = project_folder / "data" / "interim" / "test.csv"
    test_df = pd.read_csv(test_path)

    # Convert testing reviews using training vocabulary
    X_test = np.array(
        [
            review_to_vector(review, vocabulary)
            for review in test_df["review"]
        ],
        dtype=np.float32
    )

    # Convert testing labels into numbers
    label_mapping = {
        "negative": 0,
        "positive": 1
    }

    y_test = test_df["sentiment"].map(
        label_mapping
    ).to_numpy(dtype=np.int64)

    return X_test, y_test



if __name__ == "__main__":

    # Prepare training data
    X_train, y_train, vocabulary = prepare_training_data()

    # Prepare testing data using the SAME vocabulary
    X_test, y_test = prepare_testing_data(vocabulary)

    print("Training and testing data prepared successfully!")

    print("\nTraining data:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("\nTesting data:")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nVocabulary size:", len(vocabulary))
