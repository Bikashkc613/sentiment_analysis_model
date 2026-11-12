
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_sentiment


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Movie Sentiment Analysis API",
    description="An API that predicts movie review sentiment.",
    version="1.0.0"
)


# ==========================================
# REQUEST MODEL
# ==========================================

class ReviewRequest(BaseModel):

    review: str = Field(
        min_length=1,
        max_length=10000,
        description="Movie review text"
    )


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Movie Sentiment Analysis API is running!",
        "classes": ["Negative", "Neutral", "Positive"]
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "sentiment_bigram_neutral2",
        "neutral_threshold": 0.34
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(request: ReviewRequest):

    review = request.review.strip()

    if not review:
        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty."
        )

    (
        sentiment,
        confidence,
        probabilities,
        margin,
        known_features
    ) = predict_sentiment(review)

    if sentiment is None:

        raise HTTPException(
            status_code=422,
            detail="Unable to predict. No known features found."
        )

    return {
        "review": review,
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "prediction_margin": round(margin, 4),
        "probabilities": {
            "negative": round(float(probabilities[0]), 4),
            "neutral": round(float(probabilities[1]), 4),
            "positive": round(float(probabilities[2]), 4)
        },
        "known_features": known_features
    }
