import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Some visualizations will be disabled.")

class EnhancedVisualizations:
    
    def __init__(self):
        pass
    
    def create_model_performance_dashboard(self, model_details, y_true=None):
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Skipping model performance dashboard.")
            return None
            
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Распределение скоров по моделям', 
                              'Вклад моделей', 
                              'Количество обнаруженных аномалий',
                              'Сравнение средних скоров'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#11998e']
            
            model_names = list(model_details.keys())
            scores_list = [detail['scores'] for detail in model_details.values()]
            anomalies_list = [detail['anomalies'] for detail in model_details.values()]
            weights = [detail['weight'] for detail in model_details.values()]
            mean_scores = [np.mean(scores) for scores in scores_list]
            anomaly_counts = [np.sum(anomalies) for anomalies in anomalies_list]
            
            for i, (model_name, scores) in enumerate(zip(model_names, scores_list)):
                fig.add_trace(
                    go.Histogram(x=scores, name=model_name, marker_color=colors[i % len(colors)],
                                opacity=0.7, showlegend=True),
                    row=1, col=1
                )
            
            fig.add_trace(
                go.Bar(x=model_names, y=weights, marker_color=colors[:len(model_names)],
                      name='Вклад моделей'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(x=model_names, y=anomaly_counts, marker_color=colors[:len(model_names)],
                      name='Обнаруженные аномалии'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=model_names, y=mean_scores, mode='markers+lines',
                          marker=dict(size=12, color=colors[:len(model_names)]),
                          name='Средние скоры'),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="Анализ производительности моделей",
                height=800,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Fraud Score", row=1, col=1)
            fig.update_yaxes(title_text="Частота", row=1, col=1)
            
            fig.update_xaxes(title_text="Модель", row=1, col=2)
            fig.update_yaxes(title_text="Вес", row=1, col=2)
            
            fig.update_xaxes(title_text="Модель", row=2, col=1)
            fig.update_yaxes(title_text="Количество аномалий", row=2, col=1)
            
            fig.update_xaxes(title_text="Модель", row=2, col=2)
            fig.update_yaxes(title_text="Средний Fraud Score", row=2, col=2)
            
            return fig
            
        except Exception as e:
            print(f"Error creating performance dashboard: {str(e)}")
            return None
    
    def create_feature_importance_analysis(self, df, model_details, top_n=15):
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Skipping feature importance analysis.")
            return None
            
        try:
            exclude_cols = ['step', 'type', 'nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if len(feature_cols) == 0:
                return None
            
            feature_importance = {}
            
            for col in feature_cols:
                values = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                correlations = []
                for model_name, detail in model_details.items():
                    if 'scores' in detail:
                        scores = detail['scores'][:len(values)]
                        if len(scores) > 1 and len(values) > 1:
                            corr = abs(np.corrcoef(values[:len(scores)], scores)[0, 1])
                            correlations.append(corr if not np.isnan(corr) else 0)
                
                avg_corr = np.mean(correlations) if correlations else 0
                feature_importance[col] = avg_corr
            
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, importances = zip(*sorted_features)
            
            fig = go.Figure(data=[
                go.Bar(x=list(features), y=list(importances), 
                      marker_color='viridis')
            ])
            
            fig.update_layout(
                title="Анализ важности признаков",
                xaxis_title="Признаки",
                yaxis_title="Важность (средняя корреляция со скорами)",
                height=500
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating feature importance analysis: {str(e)}")
            return None
    
    def create_anomaly_cluster_visualization(self, df, combined_scores, is_suspicious, n_components=2):
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Skipping anomaly cluster visualization.")
            return None
            
        try:
            exclude_cols = ['step', 'type', 'nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if len(feature_cols) == 0:
                return None
            
            X_df = df[feature_cols].copy()
            for col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
            
            X_df = X_df.fillna(0)
            X = X_df.values
            
            if X.shape[0] < 2:
                return None
            
            if X.shape[1] > n_components:
                pca = PCA(n_components=min(n_components, X.shape[1]))
                X_reduced = pca.fit_transform(X)
            else:
                X_reduced = X[:, :n_components]
            
            fig = go.Figure()
            
            normal_points = X_reduced[~is_suspicious.astype(bool)]
            suspicious_points = X_reduced[is_suspicious.astype(bool)]
            
            if len(normal_points) > 0:
                fig.add_trace(go.Scatter(
                    x=normal_points[:, 0],
                    y=normal_points[:, 1] if normal_points.shape[1] > 1 else np.zeros(len(normal_points)),
                    mode='markers',
                    marker=dict(color='blue', size=6, opacity=0.6),
                    name='Нормальные транзакции'
                ))
            
            if len(suspicious_points) > 0:
                fig.add_trace(go.Scatter(
                    x=suspicious_points[:, 0],
                    y=suspicious_points[:, 1] if suspicious_points.shape[1] > 1 else np.zeros(len(suspicious_points)),
                    mode='markers',
                    marker=dict(color='red', size=8, opacity=0.8),
                    name='Подозрительные транзакции'
                ))
            
            fig.update_layout(
                title="Визуализация кластеров аномалий (PCA)",
                xaxis_title="PC1",
                yaxis_title="PC2",
                height=600
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating anomaly cluster visualization: {str(e)}")
            return None

enhanced_viz = EnhancedVisualizations()