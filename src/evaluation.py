import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_scores, y_pred=None):
    if y_pred is None:
        threshold = np.median(y_scores)
        y_pred = (y_scores > threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except:
        roc_auc = 0.5
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_vals, precision_vals)
    
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }
    
    return metrics_dict

def plot_roc_curve(y_true, y_scores):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        return plt.gcf()
    except Exception as e:
        print(f"Warning: Could not plot ROC curve: {str(e)}")
        return None

def plot_precision_recall_curve(y_true, y_scores):
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        return plt.gcf()
    except Exception as e:
        print(f"Warning: Could not plot Precision-Recall curve: {str(e)}")
        return None

def evaluate_model_performance(y_true, y_scores, k_values=[10, 50, 100]):
    metrics = calculate_metrics(y_true, y_scores)
    
    recall_at_k = {}
    for k in k_values:
        if len(y_scores) >= k:
            top_k_indices = np.argsort(y_scores)[-k:]
            top_k_true = y_true[top_k_indices]
            recall_at_k[f'recall@{k}'] = np.sum(top_k_true) / np.sum(y_true) if np.sum(y_true) > 0 else 0
    
    metrics.update(recall_at_k)
    return metrics

def compare_models(results_dict):
    comparison_data = []
    
    for model_name, results in results_dict.items():
        metrics = results['performance_metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc'],
            'Recall@100': metrics['recall_at_k'].get('recall@100', 0),
            'False Positive Rate': metrics['false_positive_rate']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def plot_model_comparison(comparison_df):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    models = comparison_df['Model'].tolist()
    
    normalized_df = comparison_df.copy()
    for col in metrics_columns:
        normalized_df[col] = (comparison_df[col] - comparison_df[col].min()) / (comparison_df[col].max() - comparison_df[col].min() + 1e-8)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_columns), endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(111, polar=True)
    
    for i, model in enumerate(models):
        values = normalized_df.iloc[i][metrics_columns].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_columns)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Comparison", size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    return plt

def generate_evaluation_summary(evaluation_report):
    metrics = evaluation_report['performance_metrics']
    dataset_info = evaluation_report['dataset_info']
    
    summary = f"""
# {evaluation_report['model_name']} - Evaluation Summary

## Dataset Information
- Total Samples: {dataset_info['total_samples']:,}
- Fraud Cases: {dataset_info['fraud_samples']:,}
- Fraud Rate: {dataset_info['fraud_rate']:.2%}

## Key Performance Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1_score']:.4f}
- **ROC-AUC**: {metrics['roc_auc']:.4f}
- **PR-AUC**: {metrics['pr_auc']:.4f}

## Fraud Detection Specific Metrics
- **Recall@100**: {metrics['recall_at_k'].get('recall@100', 0):.4f}
- **False Positive Rate**: {metrics['false_positive_rate']:.4f}
- **False Discovery Rate**: {metrics['false_discovery_rate']:.4f}

## Confusion Matrix
- True Positives: {metrics['true_positives']:,}
- False Positives: {metrics['false_positives']:,}
- True Negatives: {metrics['true_negatives']:,}
- False Negatives: {metrics['false_negatives']:,}
    """.strip()
    
    return summary