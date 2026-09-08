"""
Fake News Detection Using Machine Learning
===========================================
Main entry point for the application.

Technologies: Python, scikit-learn, Tkinter, NLTK, XGBoost
Algorithms: Decision Tree, Naive Bayes, KNN, Random Forest, SVM,
            Logistic Regression, XGBoost

Usage:
    python main.py              # Launch GUI application
    python main.py --cli        # Run in command-line mode
    python main.py --generate   # Generate a sample dataset
"""
import sys
import os

# Ensure the script directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_gui():
    """Launch the Tkinter GUI application."""
    from gui import launch_gui
    launch_gui()


def run_cli():
    """Run the pipeline in command-line mode."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from preprocessing import TextPreprocessor
    from feature_extraction import FeatureExtractor
    from models import ModelTrainer
    from evaluation import ModelEvaluator

    print("=" * 60)
    print("  Fake News Detection Using Machine Learning")
    print("  CLI Mode")
    print("=" * 60)

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), 'fake_news_dataset.csv')
    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")

    # Preprocessing
    print("[1/5] Preprocessing text...")
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    df = preprocessor.preprocess_dataframe(df, text_column='text', title_column='title')
    print(f"  Clean articles: {len(df)}")

    # Feature extraction
    print("[2/5] Extracting TF-IDF features...")
    extractor = FeatureExtractor(method='tfidf', max_features=5000, ngram_range=(1, 2))
    X = extractor.fit_transform(df['processed_text'])
    y = df['label'].values
    print(f"  Feature matrix: {X.shape}")

    # Split
    print("[3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Train
    print("[4/5] Training models...\n")
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train)

    # Evaluate
    print("\n[5/5] Evaluating models...\n")
    evaluator = ModelEvaluator()
    all_results = {}

    for name in trainer.model_names:
        y_pred = trainer.predict(X_test, name)
        try:
            y_prob = trainer.predict_proba(X_test, name)
        except Exception:
            y_prob = None

        results = evaluator.evaluate(y_test, y_pred, y_prob, model_name=name)
        all_results[name] = results

        print(f"--- {name} ---")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1 Score:  {results['f1_score']:.4f}")
        if results.get('roc_auc'):
            print(f"  AUC:       {results['roc_auc']:.4f}")
        print(f"  Confusion Matrix:\n    {results['confusion_matrix']}\n")

    # Best model
    best_name = max(all_results, key=lambda k: all_results[k]['accuracy'])
    best_acc = all_results[best_name]['accuracy']
    print(f"\n{'=' * 60}")
    print(f"  Best Model: {best_name}")
    print(f"  Accuracy: {best_acc:.4f}")
    print(f"{'=' * 60}")

    # Summary table
    print("\nSummary Table:")
    summary = evaluator.get_summary_table(all_results)
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'AUC':<10}")
    print("-" * 75)
    for row in summary:
        print(f"{row['Model']:<25} {row['Accuracy']:<10} {row['Precision']:<10} "
              f"{row['Recall']:<10} {row['F1 Score']:<10} {row['AUC']:<10}")

    # Sample prediction
    print("\n--- Sample Prediction ---")
    sample_text = "Government announces new economic policy to boost employment"
    cleaned = preprocessor.clean_text(sample_text)
    features = extractor.transform([cleaned])
    prediction = trainer.predict(features, best_name)[0]
    label = "REAL" if prediction == 0 else "FAKE"
    print(f"Text: '{sample_text}'")
    print(f"Prediction: {label}")

    print("\nDone! Run with --gui flag for the graphical interface.")


if __name__ == '__main__':
    if '--gui' in sys.argv:
        run_gui()
    elif '--cli' in sys.argv:
        run_cli()
    else:
        # Default: launch GUI
        run_gui()
