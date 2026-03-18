"""Model evaluation module."""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and visualization."""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names
        self.metrics = {}
        
    def compute_metrics(self, y_true: np.ndarray, 
                        y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        self.metrics = metrics
        return metrics
    
    def print_classification_report(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> str:
        """Generate detailed classification report."""
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            zero_division=0
        )
        logger.info('\n' + report)
        return report
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              save_path: str = None, figsize: Tuple[int, int] = (8, 6)):
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f'Confusion matrix saved to {save_path}')
        plt.close()
        
        return fig, ax, cm
    
    def analyze_misclassifications(self, y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> Dict:
        """Analyze misclassified samples."""
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        analysis = {
            'total_misclassified': len(misclassified_indices),
            'misclassification_rate': len(misclassified_indices) / len(y_true),
            'indices': misclassified_indices,
            'true_labels': y_true[misclassified_indices],
            'predicted_labels': y_pred[misclassified_indices]
        }
        
        cm = confusion_matrix(y_true, y_pred)
        if self.class_names:
            confusion_pairs = []
            for i, true_class in enumerate(self.class_names):
                for j, pred_class in enumerate(self.class_names):
                    if i != j and cm[i, j] > 0:
                        confusion_pairs.append({
                            'true': true_class,
                            'predicted': pred_class,
                            'count': cm[i, j]
                        })
            analysis['confusion_patterns'] = sorted(
                confusion_pairs, key=lambda x: x['count'], reverse=True
            )
        
        logger.info(f'Misclassification rate: {analysis["misclassification_rate"]:.2%}')
        return analysis
    
    def plot_metrics_comparison(self, models_metrics: Dict[str, Dict[str, float]],
                                save_path: str = None):
        """Plot comparison of metrics across multiple models."""
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        models = list(models_metrics.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_names):
            values = [models_metrics[model][metric] for model in models]
            bars = axes[idx].bar(models, values, color=sns.color_palette('viridis', len(models)))
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(f'{metric.capitalize()} Comparison')
            axes[idx].set_ylim(0, 1.05)
            axes[idx].tick_params(axis='x', rotation=45)
            
            for bar, val in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                              f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig, axes
