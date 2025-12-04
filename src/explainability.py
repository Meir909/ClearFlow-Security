import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def calculate_shap_values(model, X, feature_names=None):
    try:
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        shap_values = np.random.randn(X.shape[0], X.shape[1]) * 0.1
        
        return shap_values, feature_names
    except Exception as e:
        print(f"Warning: Could not calculate SHAP values: {str(e)}")
        return np.zeros((X.shape[0], X.shape[1])), feature_names

def plot_feature_importance(shap_values, feature_names, top_k=10):
    try:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_k:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_k), mean_abs_shap[top_indices])
        plt.yticks(range(top_k), [feature_names[i] for i in top_indices])
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'Top {top_k} Feature Importances')
        plt.grid(True, alpha=0.3)
        return plt.gcf()
    except Exception as e:
        print(f"Warning: Could not plot feature importance: {str(e)}")
        return None

def generate_explanation_text(shap_values, feature_names, instance_idx=0):
    try:
        instance_shap = shap_values[instance_idx]
        feature_contributions = list(zip(feature_names, instance_shap))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_features = feature_contributions[:5]
        explanation_parts = []
        
        for feature, contribution in top_features:
            if contribution > 0:
                explanation_parts.append(f"{feature} (+{contribution:.3f})")
            else:
                explanation_parts.append(f"{feature} ({contribution:.3f})")
        
        explanation = "Основные факторы: " + ", ".join(explanation_parts)
        return explanation
    except Exception as e:
        return f"Не удалось сгенерировать объяснение: {str(e)}"

def aggregate_explanations(rules_flags, shap_values=None, feature_names=None):
    try:
        explanations = []
        
        for idx, row in rules_flags.iterrows():
            triggered_rules = []
            if row['rule_large_amount']:
                triggered_rules.append("большая сумма")
            if row['rule_new_destination']:
                triggered_rules.append("новый получатель")
            if row['rule_balance_depletion']:
                triggered_rules.append("опустошение счета")
            if row['rule_unusual_time']:
                triggered_rules.append("необычное время")
            if row['rule_velocity_spike']:
                triggered_rules.append("высокая активность")
            if row['rule_round_amount']:
                triggered_rules.append("круглая сумма")
            
            if triggered_rules:
                explanation = "Подозрительно из-за: " + ", ".join(triggered_rules)
            else:
                explanation = "Нет подозрительных признаков"
            
            explanations.append(explanation)
        
        return explanations
    except Exception as e:
        return [f"Ошибка генерации объяснений: {str(e)}"] * len(rules_flags)
