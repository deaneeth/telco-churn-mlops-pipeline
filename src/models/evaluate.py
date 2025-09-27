"""
Model evaluation module for telco churn prediction.

This module contains functions for evaluating machine learning models
and generating comprehensive evaluation reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import logging

# Optional plotly imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None
    make_subplots = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for evaluating machine learning models."""
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.evaluation_results = {}
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str = "model") -> dict:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model instance
            X_test (pd.DataFrame): Testing features
            y_test (pd.Series): Testing target
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics and predictions
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate comprehensive metrics using the standalone function
        metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        
        # Create evaluation result in expected format
        evaluation = {
            'model_name': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'], 
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Store results
        self.evaluation_results[model_name] = evaluation
        
        logger.info(f"Model {model_name} evaluated successfully")
        return evaluation
    
    def plot_confusion_matrix(self, model_name: str, figsize: tuple = (8, 6)):
        """
        Plot confusion matrix for a specific model.
        
        Args:
            model_name (str): Name of the model
            figsize (tuple): Figure size
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} has not been evaluated")
        
        cm = self.evaluation_results[model_name]['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def plot_roc_curve(self, model_name: str = None, figsize: tuple = (10, 8)):
        """
        Plot ROC curve for one or all models.
        
        Args:
            model_name (str): Name of specific model, None for all models
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        models_to_plot = [model_name] if model_name else list(self.evaluation_results.keys())
        
        for name in models_to_plot:
            if name not in self.evaluation_results:
                continue
            
            evaluation = self.evaluation_results[name]
            if evaluation['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(evaluation['y_true'], evaluation['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curve(self, model_name: str = None, figsize: tuple = (10, 8)):
        """
        Plot Precision-Recall curve for one or all models.
        
        Args:
            model_name (str): Name of specific model, None for all models
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        models_to_plot = [model_name] if model_name else list(self.evaluation_results.keys())
        
        for name in models_to_plot:
            if name not in self.evaluation_results:
                continue
            
            evaluation = self.evaluation_results[name]
            if evaluation['y_pred_proba'] is not None:
                precision, recall, _ = precision_recall_curve(
                    evaluation['y_true'], evaluation['y_pred_proba']
                )
                avg_precision = average_precision_score(
                    evaluation['y_true'], evaluation['y_pred_proba']
                )
                
                plt.plot(recall, precision, linewidth=2,
                        label=f'{name} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def create_interactive_comparison(self):
        """
        Create interactive comparison dashboard using Plotly.
        
        Returns:
            go.Figure: Interactive Plotly figure or None if plotly not available
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create interactive comparison.")
            return None
            
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        # Prepare data for comparison
        models = []
        metrics = []
        
        for model_name, evaluation in self.evaluation_results.items():
            report = evaluation['classification_report']
            models.append(model_name)
            metrics.append({
                'model': model_name,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'accuracy': report['accuracy']
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add bar charts for each metric
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, (row, col) in zip(metrics_to_plot, positions):
            fig.add_trace(
                go.Bar(
                    x=metrics_df['model'],
                    y=metrics_df[metric],
                    name=metric.title(),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=600,
            title_text="Model Performance Comparison",
            showlegend=False
        )
        
        return fig
    
    def generate_report(self, model_name: str) -> str:
        """
        Generate a comprehensive text report for a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            str: Formatted evaluation report
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} has not been evaluated")
        
        evaluation = self.evaluation_results[model_name]
        report = evaluation['classification_report']
        cm = evaluation['confusion_matrix']
        
        report_text = f"""
=== Model Evaluation Report: {model_name} ===

Classification Report:
{classification_report(evaluation['y_true'], evaluation['y_pred'])}

Confusion Matrix:
True Negatives:  {cm[0, 0]}    False Positives: {cm[0, 1]}
False Negatives: {cm[1, 0]}    True Positives:  {cm[1, 1]}

Key Metrics:
- Accuracy:  {report['accuracy']:.4f}
- Precision: {report['weighted avg']['precision']:.4f}
- Recall:    {report['weighted avg']['recall']:.4f}
- F1-Score:  {report['weighted avg']['f1-score']:.4f}

Churn Class Performance:
- Precision: {report['1']['precision']:.4f}
- Recall:    {report['1']['recall']:.4f}
- F1-Score:  {report['1']['f1-score']:.4f}
"""
        
        return report_text
    
    def compare_models(self) -> pd.DataFrame:
        """
        Create a comparison DataFrame of all evaluated models.
        
        Returns:
            pd.DataFrame: Model comparison results
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        comparisons = []
        
        for model_name, evaluation in self.evaluation_results.items():
            report = evaluation['classification_report']
            
            comparison = {
                'Model': model_name,
                'Accuracy': report['accuracy'],
                'Precision': report['weighted avg']['precision'],
                'Recall': report['weighted avg']['recall'],
                'F1-Score': report['weighted avg']['f1-score'],
                'Churn_Precision': report['1']['precision'],
                'Churn_Recall': report['1']['recall'],
                'Churn_F1': report['1']['f1-score']
            }
            
            comparisons.append(comparison)
        
        comparison_df = pd.DataFrame(comparisons)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        return comparison_df
    
    def save_evaluation_results(self, file_path: str):
        """
        Save evaluation results to a file.
        
        Args:
            file_path (str): Path to save the results
        """
        import pickle
        
        with open(file_path, 'wb') as f:
            pickle.dump(self.evaluation_results, f)
        
        logger.info(f"Evaluation results saved to {file_path}")


def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              avg_customer_value: float = 1000) -> dict:
    """
    Calculate business-relevant metrics for churn prediction.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        avg_customer_value (float): Average customer lifetime value
        
    Returns:
        dict: Business metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Business calculations
    retention_cost = 100  # Cost to retain a customer
    acquisition_cost = 200  # Cost to acquire a new customer
    
    # Prevented churn (true positives)
    prevented_churn_value = tp * avg_customer_value
    
    # Cost of retention efforts
    retention_costs = (tp + fp) * retention_cost
    
    # Lost customers (false negatives)
    lost_customer_value = fn * avg_customer_value
    
    # Net benefit
    net_benefit = prevented_churn_value - retention_costs - lost_customer_value
    
    business_metrics = {
        'prevented_churn_customers': tp,
        'prevented_churn_value': prevented_churn_value,
        'retention_costs': retention_costs,
        'lost_customers': fn,
        'lost_customer_value': lost_customer_value,
        'net_benefit': net_benefit,
        'roi': (net_benefit / retention_costs) * 100 if retention_costs > 0 else 0
    }
    
    return business_metrics


# Standalone functions expected by test modules
def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        dict: Classification metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )
    
    # Calculate basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Calculate confusion matrix components for additional metrics
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        # Specificity (True Negative Rate)  
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        # Negative Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    else:
        # Multi-class case - set defaults
        metrics['specificity'] = 0
        metrics['npv'] = 0
    
    # Add probability-based metrics if available
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['precision_recall_auc'] = average_precision_score(y_true, y_pred_proba)
    else:
        metrics['roc_auc'] = 0
        metrics['precision_recall_auc'] = 0
    
    return metrics


def generate_confusion_matrix_plot(y_true, y_pred, title="Confusion Matrix", figsize=(8, 6)):
    """
    Generate confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The confusion matrix plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['No Churn', 'Churn'],
               yticklabels=['No Churn', 'Churn'], ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    return fig


def generate_roc_curve_plot(y_true, y_pred_proba, title="ROC Curve", figsize=(8, 6)):
    """
    Generate ROC curve plot.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The ROC curve plot
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    return fig


def generate_feature_importance_plot(feature_importances, feature_names, title="Feature Importance", figsize=(10, 8)):
    """
    Generate feature importance plot.
    
    Args:
        feature_importances: Feature importance values
        feature_names: Feature names
        title: Plot title  
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The feature importance plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Sort features by importance
    indices = np.argsort(feature_importances)[::-1][:20]  # Top 20
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(indices)), feature_importances[indices])
    ax.set_title(title)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    
    return fig


def compare_models(models, X_test, y_test, metric='accuracy'):
    """
    Compare multiple models based on a specific metric.
    
    Args:
        models: Dictionary of model instances
        X_test: Test features 
        y_test: Test labels
        metric: Metric to compare on
        
    Returns:
        dict: Comparison results with model evaluations
    """
    results = {}
    
    for model_name, model in models.items():
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            
            results[model_name] = {
                'metrics': metrics,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
        except Exception as e:
            logger.warning(f"Failed to evaluate model {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def main():
    """Main function for standalone execution."""
    print("Model evaluation module - standalone functions available:")
    functions = [
        'calculate_classification_metrics',
        'generate_confusion_matrix_plot', 
        'generate_roc_curve_plot',
        'generate_feature_importance_plot',
        'compare_models'
    ]
    for func in functions:
        print(f"  - {func}")


if __name__ == "__main__":
    main()