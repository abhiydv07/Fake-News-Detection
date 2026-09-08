# 🛡️ Fake News Detection Using Machine Learning

A complete machine learning pipeline for detecting fake news articles, featuring a trained ensemble model, REST API, and a modern web interface with offline fallback.

## 📸 Features

### ML Pipeline
- **Multiple ML Models**: Decision Tree, Naive Bayes, KNN, Random Forest, SVM, Logistic Regression, XGBoost
- **Ensemble Stacking**: Combines top models with a meta-learner for improved accuracy
- **TF-IDF Feature Extraction**: Bigram support with configurable max features
- **Text Preprocessing**: Stemming, lemmatization, stopword removal, special character cleaning
- **Adversarial Testing**: Robustness evaluation against text perturbations

### Web Interface
- 🌙 **Dark/Light Theme** — Toggle with persistence
- 📊 **Single & Batch Prediction** — Analyze one or many articles at once
- ⚖️ **Compare Mode** — Side-by-side ML model vs keyword classifier comparison
- 📁 **File Upload** — Drag & drop `.txt` files for batch analysis
- 📋 **Copy/Share Results** — One-click copy formatted prediction to clipboard
- 📥 **CSV Export** — Download batch results as spreadsheet
- 📜 **Prediction History** — Auto-saved with timestamps (localStorage)
- 🎚️ **Confidence Threshold Slider** — Adjustable in real-time
- ❓ **Help Tooltips** — Confidence level explanations for non-technical users
- 📱 **Mobile Responsive** — Works on phones and tablets
- ♿ **Accessible** — ARIA labels, keyboard navigation, focus indicators

### Auto-Connectivity
- 🟢 **Online Mode**: Uses the trained ML model via FastAPI
- 🟡 **Offline Mode**: Built-in keyword classifier with weighted scoring and regex patterns
- 🔄 **Auto-detection**: Seamlessly switches based on API/server availability
- 🔁 **Auto-reconnect**: Exponential backoff polling when connection is restored

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
cd fake-news-detection
pip install -r requirements.txt
```

### Running the Application

**Web Interface (recommended):**
```bash
python api.py
```
Then open `web_interface.html` in your browser. The interface auto-detects the API server.

**GUI Application:**
```bash
python main.py
```

**CLI Mode:**
```bash
python main.py --cli
```

**Retrain Models:**
```bash
python main.py --cli    # Trains and evaluates all models
```

## 📂 Project Structure

```
fake-news-detection/
├── api.py                 # FastAPI REST server
├── main.py                # Entry point (GUI/CLI modes)
├── web_interface.html     # Modern web UI (self-contained)
├── gui.py                 # Tkinter desktop GUI
├── models.py              # ML model training & ensemble stacking
├── preprocessing.py       # Text preprocessing pipeline
├── feature_extraction.py  # TF-IDF & feature extraction
├── evaluation.py          # Model evaluation metrics
├── adversarial.py         # Adversarial robustness testing
├── test_all.py            # Full test suite
├── test_pipeline.py       # Pipeline integration tests
├── fake_news_dataset.csv  # Training dataset
├── saved_models/          # Serialized model artifacts
│   ├── model.pkl
│   ├── vectorizer.pkl
│   ├── preprocessor.pkl
│   └── info.pkl
├── requirements.txt       # Python dependencies
└── render.yaml            # Render.com deployment config
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Models | scikit-learn, XGBoost |
| Preprocessing | NLTK (stemming, lemmatization, stopwords) |
| API Server | FastAPI + Uvicorn |
| Web Interface | Vanilla HTML/CSS/JavaScript (no framework) |
| Desktop GUI | Tkinter |
| Dataset | Fake News Dataset (CSV) |

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check & model status |
| `POST` | `/predict` | Classify a single article |
| `POST` | `/predict/batch` | Classify multiple articles |

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launches new spacecraft into orbit", "confidence_threshold": 0.6}'
```

### Example Response

```json
{
  "label": "REAL",
  "confidence": 0.89,
  "real_probability": 0.89,
  "fake_probability": 0.11,
  "is_confident": true
}
```

## 🧪 Running Tests

```bash
python -m pytest test_all.py -v
python -m pytest test_pipeline.py -v
```

## 🌐 Deployment

The project includes a `render.yaml` for easy deployment to [Render.com](https://render.com):

1. Push to GitHub
2. Connect your repo to Render
3. Render auto-detects the config and deploys the API

For the web interface, you can host `web_interface.html` on any static file server (GitHub Pages, Netlify, etc.).

## 📄 License

This project is for educational purposes.

---

**Built with ❤️ using Python, scikit-learn, and FastAPI**
