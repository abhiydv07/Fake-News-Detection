"""
Model Training Module
Implements multiple ML algorithms for fake news detection including
traditional ML, deep learning (LSTM), and ensemble stacking.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier


# Define all available models
MODELS = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs', random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=20, min_samples_split=5, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    ),
    'SVM': CalibratedClassifierCV(
        SVC(kernel='linear', C=1.0, random_state=42), ensemble=False
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5, weights='distance', n_jobs=-1
    ),
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'XGBoost': XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        eval_metric='logloss', random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    ),
}


class EnsembleStackingClassifier(BaseEstimator, ClassifierMixin):
    """Stacking ensemble that combines multiple base models with a meta-learner."""

    def __init__(self, base_models=None, meta_learner=None, confidence_threshold=0.5):
        if base_models is None:
            self.base_models = {
                'Logistic Regression': LogisticRegression(
                    max_iter=1000, random_state=42
                ),
                'SVM': CalibratedClassifierCV(
                    SVC(kernel='linear', C=1.0, random_state=42), ensemble=False
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=50, max_depth=10, random_state=42
                ),
            }
        else:
            self.base_models = base_models

        if meta_learner is None:
            self.meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self.meta_learner = meta_learner

        self.confidence_threshold = confidence_threshold
        self.trained_base_models = {}
        self.is_fitted = False
        self.classes_ = None

    def fit(self, X, y):
        """Train base models and meta-learner."""
        from sklearn.model_selection import cross_val_predict

        self.classes_ = np.unique(y)
        meta_features_train = np.zeros((X.shape[0], len(self.base_models)))

        for i, (name, model) in enumerate(self.base_models.items()):
            # Get cross-validated predictions for meta-features
            cv_preds = cross_val_predict(model, X, y, cv=3, method='predict_proba')
            if cv_preds.shape[1] == 2:
                meta_features_train[:, i] = cv_preds[:, 1]
            else:
                meta_features_train[:, i] = cv_preds[:, 0]

            # Fit the model on full training data
            model.fit(X, y)
            self.trained_base_models[name] = model

        # Train meta-learner on stacked features
        self.meta_learner.fit(meta_features_train, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Predict using stacking ensemble."""
        meta_features = self._get_meta_features(X)
        return self.meta_learner.predict(meta_features)

    def predict_proba(self, X):
        """Get probability predictions."""
        meta_features = self._get_meta_features(X)
        return self.meta_learner.predict_proba(meta_features)

    def _get_meta_features(self, X):
        """Generate meta-features from base model predictions."""
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, (name, model) in enumerate(self.trained_base_models.items()):
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                meta_features[:, i] = proba[:, 1]
            else:
                meta_features[:, i] = proba[:, 0]
        return meta_features

    def predict_with_confidence(self, X):
        """
        Predict with confidence threshold.
        Returns (predictions, confidences, is_confident).
        Predictions below threshold are marked as uncertain.
        """
        proba = self.predict_proba(X)
        confidences = np.max(proba, axis=1)
        predictions = self.meta_learner.predict(self._get_meta_features(X))
        is_confident = confidences >= self.confidence_threshold
        return predictions, confidences, is_confident


class ModelTrainer:
    """Trains and evaluates multiple ML models."""

    def __init__(self, model_names=None):
        if model_names is None:
            self.model_names = list(MODELS.keys())
        else:
            self.model_names = [n for n in model_names if n in MODELS]

        self.trained_models = {}
        self.results = {}

    def train_all(self, X_train, y_train):
        """Train all selected models on the training data."""
        for name in self.model_names:
            print(f"  Training {name}...")
            model = MODELS[name]
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        return self.trained_models

    def predict(self, X, model_name):
        """Get predictions from a specific model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained.")
        return self.trained_models[model_name].predict(X)

    def predict_proba(self, X, model_name):
        """Get probability predictions from a specific model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained.")
        model = self.trained_models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            raise ValueError(f"Model '{model_name}' does not support probability predictions.")

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation on all models."""
        cv_results = {}
        for name in self.model_names:
            model = MODELS[name]
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            cv_results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std()
            }
            print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        self.results['cross_validation'] = cv_results
        return cv_results

    def get_best_model(self):
        """Return the name of the best model based on cross-validation."""
        if 'cross_validation' not in self.results:
            return None
        cv = self.results['cross_validation']
        return max(cv, key=lambda k: cv[k]['mean_score'])


def build_lstm_model(input_dim, max_length=200):
    """
    Build an LSTM model for text classification.

    Args:
        input_dim: Vocabulary size.
        max_length: Maximum sequence length.

    Returns:
        Compiled Keras model.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, LSTM, Dense, Dropout, Bidirectional,
            GlobalMaxPooling1D, Conv1D, BatchNormalization
        )
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            Embedding(input_dim=min(input_dim, 20000), output_dim=128,
                     input_length=max_length),
            Conv1D(64, 5, activation='relu'),
            BatchNormalization(),
            Bidirectional(LSTM(64, return_sequences=True)),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    except ImportError:
        print("  TensorFlow not available. LSTM model will be skipped.")
        return None


def train_lstm(X_train_text, y_train, X_test_text, y_test,
               max_words=20000, max_length=200, epochs=10, batch_size=32):
    """
    Train an LSTM model on text data.

    Args:
        X_train_text: Training texts (list or Series).
        y_train: Training labels.
        X_test_text: Test texts.
        y_test: Test labels.
        max_words: Maximum vocabulary size.
        max_length: Maximum sequence length.
        epochs: Number of training epochs.
        batch_size: Training batch size.

    Returns:
        Dict with trained model and results, or None if TF unavailable.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except ImportError:
        print("  TensorFlow not available. Skipping LSTM training.")
        return None

    print("  Tokenizing texts...")
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=max_length)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=max_length)

    print("  Building LSTM model...")
    model = build_lstm_model(max_words, max_length)
    if model is None:
        return None

    print(f"  Training LSTM for {epochs} epochs...")
    model.fit(
        X_train_seq, y_train,
        validation_data=(X_test_seq, y_test),
        epochs=epochs, batch_size=batch_size,
        verbose=1
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test_seq, y_test, verbose=0)
    print(f"  LSTM Test Accuracy: {accuracy:.4f}")

    return {
        'model': model,
        'tokenizer': tokenizer,
        'accuracy': accuracy,
        'loss': loss,
        'max_length': max_length,
        'max_words': max_words
    }


class LSTMClassifier:
    """Wrapper to make LSTM compatible with sklearn-style API."""

    def __init__(self, lstm_result):
        self.model = lstm_result['model']
        self.tokenizer = lstm_result['tokenizer']
        self.max_length = lstm_result['max_length']

    def predict(self, X):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = pad_sequences(self.tokenizer.texts_to_sequences(X), maxlen=self.max_length)
        preds = self.model.predict(seq, verbose=0)
        return np.argmax(preds, axis=1)

    def predict_proba(self, X):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = pad_sequences(self.tokenizer.texts_to_sequences(X), maxlen=self.max_length)
        return self.model.predict(seq, verbose=0)
