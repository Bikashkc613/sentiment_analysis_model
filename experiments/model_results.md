# Model Experiment Results

## Best Bigram Model

- Model: Softmax Logistic Regression
- Feature extraction: Unigram + Bigram
- Unigram features: 5,000
- Bigram features: 10,000
- Total features: 15,000
- Classes: Negative, Neutral, Positive
- Learning rate: 0.018
- Best epoch: 43
- Test accuracy: 79.29%

## Test Classification Results

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 82.22% | 84.64% | 83.41% |
| Neutral | 61.69% | 49.79% | 55.10% |
| Positive | 82.98% | 88.25% | 85.53% |

## Model File

`models/sentiment_bigram_model_best.npz`

## Observation

The model performs well on Negative and Positive reviews. Neutral sentiment remains the most challenging class.