
from pathlib import Path
import pandas as pd
import numpy as np
import json

from src.features.vocabulary import build_vocabulary
from src.features.vectorize import reviews_to_matrix


def prepare_training_data():

    project_folder = Path(__file__).resolve().parents[2]

    train_path = (
        project_folder
        / "data"
        / "interim"
        / "train_3class.csv"
    )

    train_df = pd.read_csv(train_path)

    # Build vocabulary using cleaned training reviews
    vocabulary = build_vocabulary(
        train_df["review"],
        max_words=5000
    )

    print("\nVocabulary information:")
    print("Total vocabulary words:", len(vocabulary))

    # Verify vocabulary
    uppercase_words = [
        word for word in vocabulary
        if word != word.lower()
    ]

    print("Words containing uppercase letters:", len(uppercase_words))
    print("First 20 vocabulary words:", list(vocabulary.keys())[:20])

    # Save the newly created vocabulary
    vocabulary_path = (
        project_folder
        / "models"
        / "vocabulary.json"
    )

    vocabulary_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(vocabulary_path, "w", encoding="utf-8") as file:
        json.dump(vocabulary, file, ensure_ascii=False, indent=4)

    print("Vocabulary saved to:", vocabulary_path)

    print("\nTraining sentiment distribution:")
    print(train_df["sentiment"].value_counts())

    # Check reviews with no known vocabulary words
    empty_reviews = sum(
        not any(
            word in vocabulary
            for word in review.lower().split()
        )
        for review in train_df["review"]
    )

    print("\nReviews with no known vocabulary words:", empty_reviews)

    # Convert reviews into sparse matrix
    X_train = reviews_to_matrix(
        train_df["review"],
        vocabulary
    )

    label_mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    y_train = train_df["sentiment"].map(
        label_mapping
    ).to_numpy(dtype=np.int64)

    return X_train, y_train, vocabulary


def prepare_validation_data(vocabulary):

    project_folder = Path(__file__).resolve().parents[2]

    validation_path = (
        project_folder
        / "data"
        / "interim"
        / "validation_3class.csv"
    )

    validation_df = pd.read_csv(validation_path)

    X_validation = reviews_to_matrix(
        validation_df["review"],
        vocabulary
    )

    label_mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    y_validation = validation_df["sentiment"].map(
        label_mapping
    ).to_numpy(dtype=np.int64)

    return X_validation, y_validation


def prepare_testing_data(vocabulary):

    project_folder = Path(__file__).resolve().parents[2]

    test_path = (
        project_folder
        / "data"
        / "interim"
        / "test_3class.csv"
    )

    test_df = pd.read_csv(test_path)

    X_test = reviews_to_matrix(
        test_df["review"],
        vocabulary
    )

    label_mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    y_test = test_df["sentiment"].map(
        label_mapping
    ).to_numpy(dtype=np.int64)

    return X_test, y_test


if __name__ == "__main__":

    X_train, y_train, vocabulary = prepare_training_data()

    X_validation, y_validation = prepare_validation_data(vocabulary)

    X_test, y_test = prepare_testing_data(vocabulary)

    print("\nThree-class data prepared successfully!")

    print("\nTraining data:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("\nValidation data:")
    print("X_validation shape:", X_validation.shape)
    print("y_validation shape:", y_validation.shape)

    print("\nTesting data:")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nVocabulary size:", len(vocabulary))

    print("\nTraining label counts:")
    print(np.bincount(y_train))

    print("\nValidation label counts:")
    print(np.bincount(y_validation))

    print("\nTesting label counts:")
    print(np.bincount(y_test))

    print("\nMatrix type:", type(X_train))
    print("Training non-zero values:", X_train.nnz)
