"""
Test script: Predict on all samples, show per-sample results, then tune for better accuracy.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import TextPreprocessor
from feature_extraction import FeatureExtractor
from models import ModelTrainer, MODELS
from evaluation import ModelEvaluator

# ── Load & preprocess ──
df = pd.read_csv('fake_news_dataset.csv')
print(f"Dataset: {len(df)} articles ({(df['label']==0).sum()} real, {(df['label']==1).sum()} fake)\n")

preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
df = preprocessor.preprocess_dataframe(df, text_column='text', title_column='title')

extractor = FeatureExtractor(method='tfidf', max_features=5000, ngram_range=(1, 2))
X = extractor.fit_transform(df['processed_text'])
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Phase 1: Baseline test on all samples ──
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("PHASE 1: BASELINE -- Test on ALL samples (train+test)")
print("=" * 70)

trainer = ModelTrainer()
trainer.train_all(X_train, y_train)

# Test on ALL data
y_all_pred = trainer.predict(X, 'Naive Bayes')
all_acc = accuracy_score(y, y_all_pred)
print(f"\nNaive Bayes on full dataset: {all_acc:.2%} accuracy\n")

# Show per-sample results
print(f"{'#':<4} {'Actual':<8} {'Predicted':<10} {'Status':<8} Title")
print("-" * 80)

wrong = 0
for i, row in df.iterrows():
    actual = y[i]
    pred = y_all_pred[i]
    actual_label = "REAL" if actual == 0 else "FAKE"
    pred_label = "REAL" if pred == 0 else "FAKE"
    status = "OK" if actual == pred else "WRONG"
    if actual != pred:
        wrong += 1
    # Get title from original df
    title = str(df.iloc[i].get('title', 'N/A'))[:55]
    print(f"{i+1:<4} {actual_label:<8} {pred_label:<10} {status:<8} {title}")

print(f"\nResults: {len(df)-wrong} correct, {wrong} wrong out of {len(df)}")
print(f"Accuracy: {all_acc:.2%}")

# ── Phase 2: Detailed evaluation on test set ──
print("\n" + "=" * 70)
print("PHASE 2: DETAILED TEST SET EVALUATION (20% holdout)")
print("=" * 70)

evaluator = ModelEvaluator()
for name in trainer.model_names:
    y_pred = trainer.predict(X_test, name)
    try:
        y_prob = trainer.predict_proba(X_test, name)
    except:
        y_prob = None
    results = evaluator.evaluate(y_test, y_pred, y_prob, model_name=name)
    print(f"\n--- {name} ---")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1_score']:.4f}")

# ── Phase 3: Tune Naive Bayes for better accuracy ──
print("\n" + "=" * 70)
print("PHASE 3: HYPERPARAMETER TUNING (Naive Bayes + GridSearchCV)")
print("=" * 70)

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# Tune alpha
alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
best_alpha = 0.1
best_score = 0

print("\nTuning MultinomialNB alpha parameter:")
for alpha in alphas:
    nb = MultinomialNB(alpha=alpha)
    scores = cross_val_score(nb, X, y, cv=5, scoring='accuracy')
    mean = scores.mean()
    print(f"  alpha={alpha:<5}  CV accuracy: {mean:.4f} (+/- {scores.std():.4f})")
    if mean > best_score:
        best_score = mean
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha} (CV accuracy: {best_score:.4f})")

# Train with best alpha and test
nb_best = MultinomialNB(alpha=best_alpha)
nb_best.fit(X_train, y_train)
y_pred_best = nb_best.predict(X_test)
tuned_acc = accuracy_score(y_test, y_pred_best)
print(f"Tuned NB on test set: {tuned_acc:.4f}")

# Also tune TF-IDF parameters
print("\nTuning TF-IDF + NB pipeline:")
from sklearn.pipeline import Pipeline

best_pipeline_score = 0
best_config = {}

for max_feat in [3000, 5000, 8000, 10000]:
    for ngram in [(1,1), (1,2), (1,3)]:
        for alpha in [0.05, 0.1, 0.2]:
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=max_feat, ngram_range=ngram,
                                          sublinear_tf=True, min_df=2, max_df=0.95)),
                ('nb', MultinomialNB(alpha=alpha))
            ])
            scores = cross_val_score(pipe, df['processed_text'], y, cv=5, scoring='accuracy')
            mean = scores.mean()
            if mean > best_pipeline_score:
                best_pipeline_score = mean
                best_config = {'max_feat': max_feat, 'ngram': ngram, 'alpha': alpha}

print(f"Best config: {best_config}")
print(f"Best CV accuracy: {best_pipeline_score:.4f}")

# Final test with best config
from sklearn.feature_extraction.text import TfidfVectorizer

final_tfidf = TfidfVectorizer(
    max_features=best_config['max_feat'],
    ngram_range=best_config['ngram'],
    sublinear_tf=True, min_df=2, max_df=0.95
)
X_final = final_tfidf.fit_transform(df['processed_text'])
X_tr, X_te, y_tr, y_te = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

final_nb = MultinomialNB(alpha=best_config['alpha'])
final_nb.fit(X_tr, y_tr)
y_final_pred = final_nb.predict(X_te)
final_acc = accuracy_score(y_te, y_final_pred)
print(f"Final tuned model on test set: {final_acc:.4f}")

print("\n" + classification_report(y_te, y_final_pred, target_names=['Real', 'Fake']))
print(f"\nFinal Accuracy: {final_acc:.2%}")

# ── Phase 4: Test 30 new unseen examples ──
print("\n" + "=" * 70)
print("PHASE 4: TEST ON 30 UNSEEN CUSTOM EXAMPLES")
print("=" * 70)

test_cases = [
    # Real news (label 0)
    ("Federal Reserve raises interest rates by 25 basis points", 0),
    ("WHO approves new malaria vaccine for children under 5", 0),
    ("SpaceX successfully launches Starship into orbit", 0),
    ("Tokyo Olympics committee releases official budget report", 0),
    ("CERN discovers new subatomic particle consistent with Standard Model", 0),
    ("Apple reports record quarterly revenue of 120 billion dollars", 0),
    ("New York City opens largest urban solar farm in the country", 0),
    ("European Union reaches agreement on digital markets regulation", 0),
    ("NASA's James Webb telescope captures images of early universe galaxies", 0),
    ("World Health Organization declares end to global health emergency", 0),
    ("India successfully tests hypersonic missile in Rajasthan desert", 0),
    ("University of Oxford publishes peer-reviewed study on gene therapy", 0),
    # Fake news (label 1)
    ("Government using birds to spy on citizens claims conspiracy theorist", 1),
    ("Drinking mountain dew cures autism says anonymous internet post", 1),
    ("Secret society controls world economy from underground bunker", 1),
    ("Chemtrails turning frogs gay according to fake scientist", 1),
    ("Elon Musk reveals earth is actually a simulation run by aliens", 1),
    ("Eating dirt daily makes you immortal says fake doctor on YouTube", 1),
    ("NASA covering up evidence that moon is actually a space station", 1),
    ("5G towers causing mass bird deaths worldwide conspiracy exposed", 1),
    ("Government hiding free energy device that powers homes forever", 1),
    ("Eating only bananas for 30 days cures all mental illness", 1),
    ("Ancient aliens built the pyramids and will return next year", 1),
    ("Secret government experiment turned citizens into mindless zombies", 1),
    # Tricky / ambiguous cases
    ("Local hospital reports decrease in emergency room visits", 0),
    ("Scientists warn about potential risks of new AI technology", 0),
    ("Viral social media post claims sunlight kills viruses instantly", 1),
    ("People claim WiFi signals cause headaches and dizziness", 1),
    ("New study finds coffee reduces risk of heart disease", 0),
    ("Unverified report says vaccines cause more harm than good", 1),
]

print(f"{'#':<4} {'Actual':<8} {'Predicted':<10} {'Confidence':<12} Title")
print("-" * 85)

correct = 0
for i, (text, label) in enumerate(test_cases):
    cleaned = preprocessor.clean_text(text)
    features = final_tfidf.transform([cleaned])
    pred = final_nb.predict(features)[0]
    proba = final_nb.predict_proba(features)[0]
    confidence = max(proba)
    
    actual_label = "REAL" if label == 0 else "FAKE"
    pred_label = "REAL" if pred == 0 else "FAKE"
    match = "OK" if pred == label else "WRONG"
    if pred == label:
        correct += 1
    
    print(f"{i+1:<4} {actual_label:<8} {pred_label:<10} {confidence:.1%}       {text[:55]}")

print(f"\nUnseen test results: {correct}/{len(test_cases)} correct ({correct/len(test_cases):.1%})")
