"""
Comprehensive test script: Tests all improvements including metadata features,
sentiment analysis, ensemble stacking, adversarial testing, and confidence threshold.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import TextPreprocessor
from feature_extraction import CombinedFeatureExtractor
from models import EnsembleStackingClassifier
from adversarial import adversarial_test


def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def main():
    # Load data
    df = pd.read_csv('fake_news_dataset.csv')
    print(f"Dataset: {len(df)} articles ({(df['label']==0).sum()} real, {(df['label']==1).sum()} fake)")

    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    df = preprocessor.preprocess_dataframe(df, text_column='text', title_column='title')

    # Store original text for metadata extraction
    raw_texts = df['combined_text'].tolist()
    processed_texts = df['processed_text'].tolist()
    labels = df['label'].values

    X_train_text, X_test_text, X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        processed_texts, raw_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # ═══════════════════════════════════════════════════════
    # TEST 1: Baseline TF-IDF
    # ═══════════════════════════════════════════════════════
    print_header("TEST 1: Baseline TF-IDF + Naive Bayes")

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                           sublinear_tf=True, min_df=2, max_df=0.95)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_tfidf, y_train)
    baseline_acc = accuracy_score(y_test, nb.predict(X_test_tfidf))
    print(f"  Baseline accuracy: {baseline_acc:.4f}")

    # ═══════════════════════════════════════════════════════
    # TEST 2: Combined Features (TF-IDF + Metadata + Sentiment)
    # ═══════════════════════════════════════════════════════
    print_header("TEST 2: Combined Features (TF-IDF + Metadata + Sentiment)")

    extractor = CombinedFeatureExtractor(
        max_features=5000, ngram_range=(1, 2),
        use_metadata=True, use_sentiment=True, use_word2vec=False
    )
    X_train_combined = extractor.fit_transform(X_train_text, X_train_raw)
    X_test_combined = extractor.transform(X_test_text, X_test_raw)

    print(f"  Feature matrix shape: {X_train_combined.shape}")

    lr_combined = LogisticRegression(max_iter=1000, random_state=42)
    lr_combined.fit(X_train_combined, y_train)
    combined_acc = accuracy_score(y_test, lr_combined.predict(X_test_combined))
    print(f"  Combined features accuracy: {combined_acc:.4f}")
    print(f"  Improvement over baseline: {(combined_acc - baseline_acc)*100:+.2f}%")

    # ═══════════════════════════════════════════════════════
    # TEST 3: Ensemble Stacking
    # ═══════════════════════════════════════════════════════
    print_header("TEST 3: Ensemble Stacking (LR + SVM + RF)")

    ensemble = EnsembleStackingClassifier(confidence_threshold=0.6)
    ensemble.fit(X_train_combined, y_train)
    ensemble_preds = ensemble.predict(X_test_combined)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    print(f"  Ensemble accuracy: {ensemble_acc:.4f}")
    print(f"  Improvement over baseline: {(ensemble_acc - baseline_acc)*100:+.2f}%")

    # ═══════════════════════════════════════════════════════
    # TEST 4: Confidence Threshold
    # ═══════════════════════════════════════════════════════
    print_header("TEST 4: Confidence Threshold (Reject uncertain predictions)")

    predictions, confidences, is_confident = ensemble.predict_with_confidence(X_test_combined)

    total = len(y_test)
    confident_mask = is_confident
    confident_total = confident_mask.sum()
    confident_correct = (predictions[confident_mask] == y_test[confident_mask]).sum() if confident_total > 0 else 0
    confident_acc = confident_correct / confident_total if confident_total > 0 else 0

    print(f"  Total test samples: {total}")
    print(f"  Confident predictions (threshold=0.6): {confident_total}/{total} ({confident_total/total:.1%})")
    print(f"  Uncertain predictions (rejected): {total - confident_total}/{total}")
    print(f"  Accuracy on confident predictions: {confident_acc:.4f}")
    print(f"  Accuracy when uncertain is rejected: {confident_acc:.4f}")

    # Show per-sample confidence
    print(f"\n  Sample predictions with confidence:")
    print(f"  {'#':<4} {'Actual':<8} {'Predicted':<10} {'Conf':<8} {'Confident':<10} Title")
    print(f"  {'-'*85}")
    for i in range(min(20, total)):
        actual = "REAL" if y_test[i] == 0 else "FAKE"
        pred = "REAL" if predictions[i] == 0 else "FAKE"
        conf = confidences[i]
        conf_str = "YES" if is_confident[i] else "NO"
        match = "(correct)" if predictions[i] == y_test[i] else "(WRONG)"
        title = X_test_text[i][:45] if isinstance(X_test_text[i], str) else str(X_test_text[i])[:45]
        print(f"  {i+1:<4} {actual:<8} {pred:<10} {conf:.1%}   {conf_str:<10} {match}")

    # ═══════════════════════════════════════════════════════
    # TEST 5: Unseen 30 examples
    # ═══════════════════════════════════════════════════════
    print_header("TEST 5: 30 Unseen Examples (Ensemble Stacking)")

    test_cases = [
        ("Federal Reserve raises interest rates by 25 basis points", 0),
        ("WHO approves new malaria vaccine for children under 5", 0),
        ("SpaceX successfully launches Starship into orbit", 0),
        ("Tokyo Olympics committee releases official budget report", 0),
        ("CERN discovers new subatomic particle consistent with Standard Model", 0),
        ("Apple reports record quarterly revenue of 120 billion dollars", 0),
        ("New York City opens largest urban solar farm in the country", 0),
        ("European Union reaches agreement on digital markets regulation", 0),
        ("NASAs James Webb telescope captures images of early universe galaxies", 0),
        ("World Health Organization declares end to global health emergency", 0),
        ("India successfully tests hypersonic missile in Rajasthan desert", 0),
        ("University of Oxford publishes peer-reviewed study on gene therapy", 0),
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
        ("Local hospital reports decrease in emergency room visits", 0),
        ("Scientists warn about potential risks of new AI technology", 0),
        ("Viral social media post claims sunlight kills viruses instantly", 1),
        ("People claim WiFi signals cause headaches and dizziness", 1),
        ("New study finds coffee reduces risk of heart disease", 0),
        ("Unverified report says vaccines cause more harm than good", 1),
    ]

    # Preprocess test cases
    test_cleaned = [preprocessor.clean_text(t) for t, _ in test_cases]
    test_raw = [t for t, _ in test_cases]

    # Extract features
    test_features = extractor.transform(test_cleaned, test_raw)

    # Predict with ensemble
    test_preds, test_confs, test_confident = ensemble.predict_with_confidence(test_features)

    correct = 0
    confident_correct = 0
    confident_total_t = 0

    print(f"  {'#':<4} {'Actual':<8} {'Predicted':<10} {'Conf':<8} {'Conf':<8} Title")
    print(f"  {'-'*85}")

    for i, (text, label) in enumerate(test_cases):
        actual = "REAL" if label == 0 else "FAKE"
        pred = "REAL" if test_preds[i] == 0 else "FAKE"
        conf = test_confs[i]
        conf_str = "YES" if test_confident[i] else "NO"
        match = "OK" if test_preds[i] == label else "WRONG"

        if test_preds[i] == label:
            correct += 1
        if test_confident[i]:
            confident_total_t += 1
            if test_preds[i] == label:
                confident_correct += 1

        print(f"  {i+1:<4} {actual:<8} {pred:<10} {conf:.1%}   {conf_str:<8} {text[:55]}")

    print(f"\n  Overall: {correct}/{len(test_cases)} ({correct/len(test_cases):.1%})")
    if confident_total_t > 0:
        print(f"  Confident: {confident_correct}/{confident_total_t} ({confident_correct/confident_total_t:.1%})")
    print(f"  Rejected: {len(test_cases)-confident_total_t}/{len(test_cases)}")

    # ═══════════════════════════════════════════════════════
    # TEST 6: Adversarial Robustness
    # ═══════════════════════════════════════════════════════
    print_header("TEST 6: Adversarial Robustness Test")

    # Use the TF-IDF vectorizer from ensemble for adversarial test
    # We need the raw text for adversarial generation
    adv_test_cases = [(text, label) for text, label in test_cases]

    adv_results = adversarial_test(
        model=nb,
        vectorizer=tfidf,
        preprocessor=preprocessor,
        test_cases=adv_test_cases,
        n_variants=3
    )

    print(f"  Original accuracy: {adv_results['original_accuracy']:.4f}")
    print(f"  Adversarial accuracy: {adv_results['adversarial_accuracy']:.4f}")
    print(f"  Robustness gap: {adv_results['robustness_gap']:.4f}")
    print(f"  Total adversarial tests: {adv_results['total_adversarial']}")

    if adv_results['failures']:
        print(f"\n  Adversarial failures ({len(adv_results['failures'])}):")
        for f in adv_results['failures'][:5]:
            print(f"    Type: {f['variant_type']}")
            print(f"    True: {f['true_label']} | Predicted: {f['predicted']}")
            print(f"    Variant: {f['variant'][:70]}...")
            print()

    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    print_header("SUMMARY OF ALL IMPROVEMENTS")

    print(f"""
  1. Baseline (TF-IDF + NB):           {baseline_acc:.2%}
  2. Combined Features (+meta+sent):    {combined_acc:.2%} ({(combined_acc-baseline_acc)*100:+.2f}%)
  3. Ensemble Stacking:                 {ensemble_acc:.2%} ({(ensemble_acc-baseline_acc)*100:+.2f}%)
  4. Confident-only accuracy:           {confident_acc:.2%} (threshold=0.6)
  5. Unseen 30 test:                    {correct/len(test_cases):.2%}
  6. Adversarial robustness:            {adv_results['adversarial_accuracy']:.2%}
  7. Robustness gap:                    {adv_results['robustness_gap']:.2%}

  Models implemented:
    - Naive Bayes, SVM, Logistic Regression, Decision Tree
    - Random Forest, XGBoost, Gradient Boosting, KNN
    - Ensemble Stacking (NB + SVM + LR)
    - LSTM (requires TensorFlow)

  Features implemented:
    - TF-IDF vectorization (unigrams + bigrams)
    - Metadata (length, punctuation, caps, readability)
    - Sentiment analysis (positive/negative ratios)
    - Word2Vec embeddings (optional)

  Deployment:
    - FastAPI REST API with batch prediction
    - Model persistence (pickle/joblib)
    - Confidence threshold with reject option
    - Adversarial testing framework
""")


if __name__ == '__main__':
    main()
