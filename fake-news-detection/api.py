"""
FastAPI REST API for Fake News Detection
Provides endpoints for single and batch prediction with confidence thresholds.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from models import EnsembleStackingClassifier

# ── App setup ──
app = FastAPI(
    title="Fake News Detection API",
    description="ML-powered fake news classifier with batch prediction",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
loaded_model = None
loaded_vectorizer = None
loaded_preprocessor = None
model_info = {}


# ── Request/Response models ──
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="News text to classify")
    title: Optional[str] = Field(None, description="Optional title")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)

class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=1000)
    titles: Optional[list[str]] = None
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)

class PredictionResult(BaseModel):
    label: str
    confidence: float
    real_probability: float
    fake_probability: float
    is_confident: bool

class BatchPredictionResult(BaseModel):
    predictions: list[PredictionResult]
    total: int
    confident_count: int
    uncertain_count: int

class ModelInfo(BaseModel):
    model_type: str
    accuracy: float
    features: int
    training_samples: int


# ── Model persistence ──
def save_model(model, vectorizer, preprocessor, info, path=None):
    """Save trained model, vectorizer, and preprocessor to disk."""
    if path is None:
        path = MODEL_DIR
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(path, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(path, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(path, "info.pkl"), "wb") as f:
        pickle.dump(info, f)

    print(f"Model saved to {path}")


def load_saved_model(path=None):
    """Load saved model from disk."""
    global loaded_model, loaded_vectorizer, loaded_preprocessor, model_info

    if path is None:
        path = MODEL_DIR

    model_file = os.path.join(path, "model.pkl")
    if not os.path.exists(model_file):
        return False

    with open(model_file, "rb") as f:
        loaded_model = pickle.load(f)
    with open(os.path.join(path, "vectorizer.pkl"), "rb") as f:
        loaded_vectorizer = pickle.load(f)
    with open(os.path.join(path, "preprocessor.pkl"), "rb") as f:
        loaded_preprocessor = pickle.load(f)
    with open(os.path.join(path, "info.pkl"), "rb") as f:
        model_info = pickle.load(f)

    print(f"Model loaded from {path}")
    return True


def _predict_single(text, title=None, confidence_threshold=0.5):
    """Internal prediction function."""
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and save a model first.")

    # Combine title and text
    full_text = f"{title} {text}" if title else text
    cleaned = loaded_preprocessor.clean_text(full_text)
    features = loaded_vectorizer.transform([cleaned])

    # Get prediction
    prediction = loaded_model.predict(features)[0]

    # Get probabilities
    if hasattr(loaded_model, 'predict_proba'):
        proba = loaded_model.predict_proba(features)[0]
        real_prob = float(proba[0])
        fake_prob = float(proba[1])
        confidence = max(real_prob, fake_prob)
    else:
        real_prob = 1.0 if prediction == 0 else 0.0
        fake_prob = 1.0 if prediction == 1 else 0.0
        confidence = 1.0

    label = "REAL" if prediction == 0 else "FAKE"
    is_confident = confidence >= confidence_threshold

    return PredictionResult(
        label=label,
        confidence=round(confidence, 4),
        real_probability=round(real_prob, 4),
        fake_probability=round(fake_prob, 4),
        is_confident=is_confident
    )


# ── Endpoints ──
@app.get("/")
def root():
    return {
        "message": "Fake News Detection API",
        "endpoints": {
            "predict": "POST /predict - Classify a single news article",
            "batch": "POST /predict/batch - Classify multiple articles",
            "info": "GET /model/info - Get model information",
            "health": "GET /health - Health check"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": loaded_model is not None
    }


@app.get("/model/info")
def get_model_info():
    if not model_info:
        raise HTTPException(status_code=404, detail="No model info available.")
    return model_info


@app.post("/predict", response_model=PredictionResult)
def predict(req: PredictRequest):
    return _predict_single(req.text, req.title, req.confidence_threshold)


@app.post("/predict/batch", response_model=BatchPredictionResult)
def predict_batch(req: BatchPredictRequest):
    predictions = []
    for i, text in enumerate(req.texts):
        title = req.titles[i] if req.titles and i < len(req.titles) else None
        result = _predict_single(text, title, req.confidence_threshold)
        predictions.append(result)

    confident = sum(1 for p in predictions if p.is_confident)

    return BatchPredictionResult(
        predictions=predictions,
        total=len(predictions),
        confident_count=confident,
        uncertain_count=len(predictions) - confident
    )


@app.post("/model/train")
def train_and_save_model():
    """Train model on the built-in dataset and save to disk."""
    global loaded_model, loaded_vectorizer, loaded_preprocessor, model_info

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "fake_news_dataset.csv")
    df = pd.read_csv(dataset_path)

    # Preprocess
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    df = preprocessor.preprocess_dataframe(df, text_column='text', title_column='title')

    # Feature extraction
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, max_df=0.95
    )
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label'].values

    # Train
    model = MultinomialNB(alpha=0.1)
    model.fit(X, y)

    accuracy = float((model.predict(X) == y).mean())

    # Save
    info = {
        "model_type": "MultinomialNB",
        "accuracy": accuracy,
        "features": X.shape[1],
        "training_samples": len(df),
        "classes": ["REAL", "FAKE"]
    }

    save_model(model, vectorizer, preprocessor, info)
    loaded_model = model
    loaded_vectorizer = vectorizer
    loaded_preprocessor = preprocessor
    model_info = info

    return {"message": "Model trained and saved.", "accuracy": accuracy}


# ── Startup ──
@app.on_event("startup")
def startup():
    if not load_saved_model():
        print("No saved model found. Training new model...")
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import TfidfVectorizer
        from preprocessing import TextPreprocessor

        dataset_path = os.path.join(os.path.dirname(__file__), "fake_news_dataset.csv")
        df = pd.read_csv(dataset_path)
        preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
        df = preprocessor.preprocess_dataframe(df, text_column='text', title_column='title')

        vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2, max_df=0.95
        )
        X = vectorizer.fit_transform(df['processed_text'])
        y = df['label'].values

        model = MultinomialNB(alpha=0.1)
        model.fit(X, y)
        accuracy = float((model.predict(X) == y).mean())

        info = {
            "model_type": "MultinomialNB",
            "accuracy": accuracy,
            "features": X.shape[1],
            "training_samples": len(df),
            "classes": ["REAL", "FAKE"]
        }
        save_model(model, vectorizer, preprocessor, info)
        loaded_model = model
        loaded_vectorizer = vectorizer
        loaded_preprocessor = preprocessor
        model_info = info
        print(f"Model trained with {accuracy:.2%} accuracy.")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
