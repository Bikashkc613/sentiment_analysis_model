
from pathlib import Path
import pandas as pd
import numpy as np
import json

from src.features.vocabulary import build_vocabulary
from src.features.bigram_vocabulary import build_bigram_vocabulary
from src.features.combined_vectorize import reviews_to_combined_matrix


def prepare_training_data():

    project_folder = Path(__file__).resolve().parents[2]

    train_path = (
        project_folder
        / "data"
        / "interim"
        / "train_3class.csv"
    )

    train_df = pd.read_csv(train_path)

    # Build unigram vocabulary
    unigram_vocabulary = build_vocabulary(
        train_df["review"],
        max_words=5000
    )

    # Build bigram vocabulary using training reviews only
    bigram_vocabulary = build_bigram_vocabulary(
        train_df["review"],
        min_frequency=100,
        max_bigrams=10000
    )

    print("\nVocabulary information:")
    print("Unigram vocabulary:", len(unigram_vocabulary))
    print("Bigram vocabulary:", len(bigram_vocabulary))
    print(
        "Total features:",
        len(unigram_vocabulary) + len(bigram_vocabulary)
    )

    # Save bigram vocabulary
    bigram_path = (
        project_folder
        / "models"
        / "bigram_vocabulary.json"
    )

    bigram_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(bigram_path, "w", encoding="utf-8") as file:
        json.dump(
            bigram_vocabulary,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("Bigram vocabulary saved to:", bigram_path)

    # Create training matrix
    X_train = reviews_to_combined_matrix(
        train_df["review"],
        unigram_vocabulary,
        bigram_vocabulary
    )

    label_mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

    y_train = train_df["sentiment"].map(
        label_mapping
    ).to_numpy(dtype=np.int64)

    return (
        X_train,
        y_train,
        unigram_vocabulary,
        bigram_vocabulary
    )


def prepare_validation_data(
    unigram_vocabulary,
    bigram_vocabulary
):

    project_folder = Path(__file__).resolve().parents[2]

    validation_path = (
        project_folder
        / "data"
        / "interim"
        / "validation_3class.csv"
    )

    validation_df = pd.read_csv(validation_path)

    X_validation = reviews_to_combined_matrix(
        validation_df["review"],
        unigram_vocabulary,
        bigram_vocabulary
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


def prepare_testing_data(
    unigram_vocabulary,
    bigram_vocabulary
):

    project_folder = Path(__file__).resolve().parents[2]

    test_path = (
        project_folder
        / "data"
        / "interim"
        / "test_3class.csv"
    )

    test_df = pd.read_csv(test_path)

    X_test = reviews_to_combined_matrix(
        test_df["review"],
        unigram_vocabulary,
        bigram_vocabulary
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

    X_test, y_test = prepare_testing_data(
        unigram_vocabulary,
        bigram_vocabulary
    )

    print("\nThree-class combined data prepared!")

    print("\nTraining data:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("\nValidation data:")
    print("X_validation shape:", X_validation.shape)
    print("y_validation shape:", y_validation.shape)

    print("\nTesting data:")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nTotal features:", X_train.shape[1])

    print("\nTraining label counts:")
    print(np.bincount(y_train))

    print("\nValidation label counts:")
    print(np.bincount(y_validation))

    print("\nTesting label counts:")
    print(np.bincount(y_test))

    print("\nMatrix type:", type(X_train))
    print("Training non-zero values:", X_train.nnz)
