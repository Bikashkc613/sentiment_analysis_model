import json
import numpy as np
import pandas as pd

from src.features.prepare_3class_data_with_bigrams import (
    prepare_validation_data
)
from src.model.softmax_model import forward_pass


# --------------------------------------------------
# 1. LOAD SAVED VOCABULARIES
# --------------------------------------------------

with open("models/vocabulary.json", "r", encoding="utf-8") as f:
    unigram_vocabulary = json.load(f)

with open("models/bigram_vocabulary.json", "r", encoding="utf-8") as f:
    bigram_vocabulary = json.load(f)


# --------------------------------------------------
# 2. LOAD BEST MODEL
# --------------------------------------------------

model = np.load("models/sentiment_bigram_model_best.npz")

W = model["W"]
b = model["b"]

print("Best model loaded successfully!")


# --------------------------------------------------
# 3. LOAD VALIDATION DATA
# --------------------------------------------------

X_validation, y_validation = prepare_validation_data(
    unigram_vocabulary,
    bigram_vocabulary
)

validation_df = pd.read_csv(
    "data/interim/validation_3class.csv"
)

reviews = validation_df["review"].tolist()

print("Validation reviews:", len(reviews))


# --------------------------------------------------
# 4. MAKE PREDICTIONS
# --------------------------------------------------

scores, probabilities = forward_pass(
    X_validation,
    W,
    b
)

predictions = np.argmax(probabilities, axis=1)

sentiment_names = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# --------------------------------------------------
# 5. FIND INCORRECT PREDICTIONS
# --------------------------------------------------

errors = []

for i in range(len(y_validation)):

    actual = int(y_validation[i])
    predicted = int(predictions[i])

    if actual != predicted:

        confidence = float(probabilities[i, predicted])

        errors.append({
            "review": reviews[i],
            "actual_sentiment": sentiment_names[actual],
            "predicted_sentiment": sentiment_names[predicted],
            "confidence": round(confidence, 4),
            "error_type": (
                f"{sentiment_names[actual]} -> "
                f"{sentiment_names[predicted]}"
            )
        })


# --------------------------------------------------
# 6. CREATE ERROR DATAFRAME
# --------------------------------------------------

errors_df = pd.DataFrame(errors)

output_path = "data/processed/prediction_errors.csv"

errors_df.to_csv(output_path, index=False)


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

total = len(y_validation)
incorrect = len(errors_df)

print("\n" + "=" * 60)
print("PREDICTION ERROR ANALYSIS")
print("=" * 60)

print("Total validation reviews:", total)
print("Incorrect predictions:", incorrect)
print("Error percentage:", round(incorrect / total * 100, 2), "%")

print("\nError types:")
print(errors_df["error_type"].value_counts())

print("\n20 lowest-confidence wrong predictions:")

if not errors_df.empty:
    print(
        errors_df.sort_values("confidence")
        .head(20)
        .to_string(index=False)
    )

print("\nSaved to:", output_path)