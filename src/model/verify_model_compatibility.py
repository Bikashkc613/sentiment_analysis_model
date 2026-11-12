
import json
from pathlib import Path
import numpy as np

from src.features.prepare_3class_data_with_bigrams import (
    prepare_training_data
)

project_folder = Path(__file__).resolve().parents[2]

# Rebuild vocabulary from training data
(
    X_train,
    y_train,
    unigram_vocabulary,
    bigram_vocabulary
) = prepare_training_data()

# Load saved vocabularies
with open(project_folder / "models" / "vocabulary.json", encoding="utf-8") as f:
    saved_unigram = json.load(f)

with open(project_folder / "models" / "bigram_vocabulary.json", encoding="utf-8") as f:
    saved_bigram = json.load(f)

# Load best model
model = np.load(
    project_folder / "models" / "sentiment_bigram_model_best.npz"
)

W = model["W"]
b = model["b"]

print("\nUNIGRAM VOCABULARY MATCH:",
      unigram_vocabulary == saved_unigram)

print("BIGRAM VOCABULARY MATCH:",
      bigram_vocabulary == saved_bigram)

print("\nTRAINING FEATURE COUNT:", X_train.shape[1])
print("MODEL WEIGHT SHAPE:", W.shape)
print("MODEL BIAS SHAPE:", b.shape)

print("\nFINAL CHECK:")

if (
    unigram_vocabulary == saved_unigram
    and bigram_vocabulary == saved_bigram
    and X_train.shape[1] == W.shape[0]
    and W.shape[1] == 3
    and b.shape == (3,)
):
    print("PASS: Vocabulary and model dimensions match.")
else:
    print("FAIL: Compatibility problem detected.")
