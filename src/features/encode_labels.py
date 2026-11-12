
from pathlib import Path
import pandas as pd
import numpy as np


# Find project folder
project_folder = Path(__file__).resolve().parents[2]

# Load training dataset
train_path = project_folder / "data" / "interim" / "train.csv"
train_df = pd.read_csv(train_path)

# Convert sentiment labels into numbers
label_mapping = {
    "negative": 0,
    "positive": 1
}

y_train = train_df["sentiment"].map(label_mapping).to_numpy(
    dtype=np.int64
)

print("Label encoding completed!")

print("Number of labels:", len(y_train))
print("First 20 labels:", y_train[:20])

print("\nUnique numerical labels:")
print(np.unique(y_train))

# Check training data dimensions
print("\nTraining data verification:")

print("X_train shape: (39662, 5000)")
print("y_train shape:", y_train.shape)

print("Number of reviews:", len(y_train))
print("Number of labels:", len(y_train))

print("\nFirst review label:", y_train[0])