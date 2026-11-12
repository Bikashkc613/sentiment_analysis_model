# Movie Sentiment Analysis — 3-Class Classifier

A machine learning project that classifies movie reviews into three sentiment categories:

- **Negative**
- **Neutral**
- **Positive**

The model is implemented using Python and NumPy, with a custom softmax regression implementation.

## Project Overview

This project builds a three-class sentiment classifier using movie reviews.

Instead of using a pre-trained sentiment model or a ready-made machine learning classifier, the core softmax regression model is implemented from scratch using NumPy.

The model uses:
- Unigram Bag-of-Words features
- Bigram Bag-of-Words features
- Sparse feature matrices
- Softmax regression
- Mini-batch gradient descent

## Dataset

The project uses a movie review dataset containing 102,283 processed reviews.

| Sentiment | Number of reviews |
|---|---:|
| Negative | 38,200 |
| Neutral | 20,246 |
| Positive | 43,837 |
| **Total** | **102,283** |

### Rating-to-sentiment mapping

- Ratings 1–3: Negative
- Ratings 4–7: Neutral
- Ratings 8–10: Positive

Dataset source: [Nuanced Sentiment Classification — GitHub](https://github.com/abrish2049/nuanced-sentiment-classification)

The source dataset is licensed under CC BY-NC-SA 4.0. Check its license terms before redistributing the data.

## Model Architecture

The model uses a combined vocabulary of 15,000 features:

| Feature type | Vocabulary size |
|---|---:|
| Unigrams | 5,000 |
| Bigrams | 10,000 |
| **Total** | **15,000** |

The model calculates class probabilities using softmax:

\[
P(y=k\mid x)=
\frac{e^{z_k}}{\sum_j e^{z_j}}
\]

The class with the highest probability becomes the predicted sentiment.

## Training Results

The best saved model was trained with:

- Learning rate: 0.018
- Best epoch: 43
- Feature count: 15,000
- Model type: Softmax regression

Previously recorded evaluation results:

| Evaluation | Accuracy |
|---|---:|
| Validation | Approximately 79.25% |
| Test | Approximately 79.29% |

Note: The test set was used during experimentation, so this test result should not be presented as a completely untouched final benchmark.

A separate 50-review blind audit produced 52% agreement with the user's labels. This was a small manual evaluation and should be reported separately from the dataset-based evaluation.

## Project Structure

```text
sentiment_analysis_model/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   ├── vocabulary.json
│   ├── bigram_vocabulary.json
│   └── sentiment_bigram_model_best.npz
│
├── src/
│   ├── features/
│   ├── model/
│   └── predict.py
│
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd sentiment_analysis_model
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Predictor

From the project root:

```bash
python -m src.predict
```

Enter a movie review when prompted.

The predictor displays:
- Negative probability
- Neutral probability
- Positive probability
- Predicted sentiment
- Prediction confidence
- Prediction margin

Type `quit` to stop.

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn (evaluation metrics)
- Git and GitHub

## Learning Objectives

This project demonstrates practical understanding of:

1. Text preprocessing
2. Vocabulary creation
3. Unigram and bigram feature extraction
4. Sparse matrix representation
5. Softmax regression
6. Mini-batch gradient descent
7. Model evaluation
8. Confusion matrix and classification metrics
9. Error analysis
10. Custom text prediction

## Limitations

- The model uses Bag-of-Words features and does not fully understand word order or context.
- Sarcasm and mixed sentiment can be difficult to classify.
- Neutral sentiment is particularly challenging.
- Predictions depend on the vocabulary learned during training.

## Future Improvements

- Improve neutral sentiment classification.
- Experiment with TF-IDF features.
- Explore more advanced text representations.
- Build a web-based prediction interface.
- Evaluate on a new, untouched test dataset.

---

**Author:** Your Name

**Project:** Three-Class Movie Sentiment Analysis