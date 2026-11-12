# Sentiment Model Experiment Log

## Bigram Model

- Architecture: Softmax Logistic Regression
- Features: 5,000 Unigrams + 10,000 Bigrams
- Classes: Negative, Neutral, Positive
- Training Accuracy: 75.42%
- Validation Accuracy: 74.29%
- Training Loss: 0.6868
- Validation Loss: 0.6978
- Batch Size: 256
- Learning Rate: 0.001
- Epochs: 40
- Class Weights: [1.0, 1.5, 1.0]

## Observation

Training and validation performance are close.
No significant overfitting is observed.