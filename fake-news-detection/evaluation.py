"""
Model Evaluation Module
Handles calculating metrics, confusion matrix, and ROC curves.
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)


class ModelEvaluator:
    """Evaluates and visualizes ML model performance."""

    def __init__(self):
        self.evaluation_results = {}

    def evaluate(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """
        Calculate all evaluation metrics for a model.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities (for ROC curve).
            model_name: Name of the model.

        Returns:
            Dict with all metrics.
        """
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true, y_pred, target_names=['Real', 'Fake']
            )
        }

        if y_prob is not None and len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            # Binary: use probability of positive class
            fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
        elif y_prob is not None and len(y_prob.shape) == 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        else:
            fpr, tpr, thresholds = None, None, None

        if fpr is not None:
            results['roc_auc'] = auc(fpr, tpr)
            results['roc_fpr'] = fpr
            results['roc_tpr'] = tpr
        else:
            results['roc_auc'] = None

        self.evaluation_results[model_name] = results
        return results

    def plot_confusion_matrix(self, cm, model_name="Model", ax=None):
        """
        Plot a confusion matrix as a heatmap.

        Args:
            cm: Confusion matrix array.
            model_name: Name of the model.
            ax: Matplotlib axes to plot on.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        classes = ['Real', 'Fake']
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=classes, yticklabels=classes,
            xlabel='Predicted Label', ylabel='True Label',
            title=f'Confusion Matrix - {model_name}'
        )

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        return ax

    def plot_roc_curve(self, results_dict, ax=None):
        """
        Plot ROC curves for one or more models.

        Args:
            results_dict: Dict of {model_name: evaluation_results}.
            ax: Matplotlib axes to plot on.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))

        for (name, results), color in zip(results_dict.items(), colors):
            if 'roc_fpr' in results and results['roc_fpr'] is not None:
                ax.plot(results['roc_fpr'], results['roc_tpr'],
                        color=color, lw=2,
                        label=f'{name} (AUC = {results["roc_auc"]:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        return ax

    def plot_accuracy_comparison(self, results_dict, ax=None):
        """
        Plot a bar chart comparing accuracy across models.

        Args:
            results_dict: Dict of {model_name: evaluation_results}.
            ax: Matplotlib axes to plot on.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        names = list(results_dict.keys())
        accuracies = [results_dict[n]['accuracy'] for n in names]
        precisions = [results_dict[n]['precision'] for n in names]
        recalls = [results_dict[n]['recall'] for n in names]
        f1s = [results_dict[n]['f1_score'] for n in names]

        x = np.arange(len(names))
        width = 0.2

        ax.bar(x - 1.5 * width, accuracies, width, label='Accuracy', color='steelblue')
        ax.bar(x - 0.5 * width, precisions, width, label='Precision', color='forestgreen')
        ax.bar(x + 0.5 * width, recalls, width, label='Recall', color='darkorange')
        ax.bar(x + 1.5 * width, f1s, width, label='F1 Score', color='crimson')

        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return ax

    def get_summary_table(self, results_dict):
        """
        Generate a summary table of all model results.

        Args:
            results_dict: Dict of {model_name: evaluation_results}.

        Returns:
            List of dicts with metrics for each model.
        """
        rows = []
        for name, results in results_dict.items():
            rows.append({
                'Model': name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1 Score': f"{results['f1_score']:.4f}",
                'AUC': f"{results['roc_auc']:.4f}" if results.get('roc_auc') else 'N/A'
            })
        return rows
