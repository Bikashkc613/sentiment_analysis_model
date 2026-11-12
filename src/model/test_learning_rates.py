from src.features.prepare_3class_data import (
    prepare_training_data
)

from src.model.train_model import train_model


# Load training data once
X_train, y_train, vocabulary = prepare_training_data()

print("Training data loaded!")
print("Training reviews:", X_train.shape[0])


# Test different learning rates
learning_rates = [
    0.00001,
    0.0001,
    0.001
]


for lr in learning_rates:

    print("\n" + "=" * 50)
    print("Testing learning rate:", lr)
    print("=" * 50)

    W, b = train_model(
        X_train,
        y_train,
        epochs=5,
        batch_size=256,
        learning_rate=lr
    )