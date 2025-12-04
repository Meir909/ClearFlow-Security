import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import json
import os
from datetime import datetime, timedelta

class SelfLearningFraudDetector:
    
    def __init__(self, model_storage_path="self_learning_models"):
        self.model_storage_path = model_storage_path
        self.pattern_history = []
        self.anomaly_clusters = {}
        self.performance_metrics = {
            'detection_rates': [],
            'false_positive_rates': [],
            'adaptation_events': []
        }
        
        if not os.path.exists(model_storage_path):
            os.makedirs(model_storage_path)
            
        self.load_previous_learning()
    
    def detect_new_patterns(self, df, fraud_scores, is_suspicious):
        suspicious_df = df[is_suspicious.astype(bool)].copy()
        
        if len(suspicious_df) == 0:
            return []
        
        pattern_features = self.extract_pattern_features(suspicious_df)
        
        if len(pattern_features) == 0:
            return []
        
        best_patterns = []
        
        try:
            clustering = DBSCAN(eps=0.5, min_samples=3)
            clusters = clustering.fit_predict(pattern_features)
            
            unique_clusters, counts = np.unique(clusters, return_counts=True)
            dbscan_patterns = []
            
            for cluster_id, count in zip(unique_clusters, counts):
                if cluster_id != -1 and count >= 3:
                    cluster_mask = clusters == cluster_id
                    cluster_data = pattern_features[cluster_mask]
                    
                    if len(cluster_data) > 1:
                        try:
                            sil_score = silhouette_score(pattern_features, clusters)
                        except:
                            sil_score = 0.0
                    else:
                        sil_score = 0.0
                    
                    pattern = {
                        'pattern_id': f"dbscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cluster_id}",
                        'timestamp': datetime.now().isoformat(),
                        'frequency': int(count),
                        'characteristics': self.describe_pattern(cluster_data, suspicious_df[cluster_mask]),
                        'confidence': float(count / len(suspicious_df)),
                        'cluster_center': cluster_data.mean(axis=0).tolist(),
                        'detection_method': 'DBSCAN',
                        'quality_score': sil_score,
                        'cluster_size': len(cluster_data)
                    }
                    
                    dbscan_patterns.append(pattern)
                    self.pattern_history.append(pattern)
            
            best_patterns.extend(dbscan_patterns)
        except Exception as e:
            print(f"Warning: DBSCAN clustering failed: {str(e)}")
        
        try:
            statistical_patterns = self.detect_statistical_patterns(suspicious_df, fraud_scores[is_suspicious.astype(bool)])
            best_patterns.extend(statistical_patterns)
        except Exception as e:
            print(f"Warning: Statistical pattern detection failed: {str(e)}")
        
        return best_patterns
    
    def detect_statistical_patterns(self, suspicious_df, suspicious_scores):
        patterns = []
        
        if len(suspicious_scores) > 10:
            q75, q25 = np.percentile(suspicious_scores, [75 ,25])
            iqr = q75 - q25
            upper_bound = q75 + (1.5 * iqr)
            
            very_high_confidence = suspicious_scores[suspicious_scores > upper_bound]
            if len(very_high_confidence) >= 3:
                pattern = {
                    'pattern_id': f"statistical_{datetime.now().strftime('%Y%m%d_%H%M%S')}_high_confidence",
                    'timestamp': datetime.now().isoformat(),
                    'frequency': int(len(very_high_confidence)),
                    'characteristics': f"Very high confidence fraud scores (>{upper_bound:.4f})",
                    'confidence': float(len(very_high_confidence) / len(suspicious_scores)),
                    'detection_method': 'Statistical Outliers',
                    'quality_score': 0.8,
                    'cluster_size': len(very_high_confidence)
                }
                patterns.append(pattern)
                self.pattern_history.append(pattern)
        
        if 'amount' in suspicious_df.columns and len(suspicious_df) > 5:
            amounts = suspicious_df['amount'].values
            mean_amount = np.mean(amounts)
            std_amount = np.std(amounts)
            
            large_threshold = mean_amount + 2 * std_amount
            large_transactions = amounts[amounts > large_threshold]
            
            if len(large_transactions) >= 3:
                pattern = {
                    'pattern_id': f"statistical_{datetime.now().strftime('%Y%m%d_%H%M%S')}_large_amounts",
                    'timestamp': datetime.now().isoformat(),
                    'frequency': int(len(large_transactions)),
                    'characteristics': f"Large transaction amounts (>{large_threshold:.2f})",
                    'confidence': float(len(large_transactions) / len(amounts)),
                    'detection_method': 'Amount Analysis',
                    'quality_score': 0.7,
                    'cluster_size': len(large_transactions)
                }
                patterns.append(pattern)
                self.pattern_history.append(pattern)
        
        return patterns

    def get_pattern_insights(self):
        if not self.pattern_history:
            return "Нет обнаруженных паттернов"
        
        method_counts = {}
        for pattern in self.pattern_history:
            method = pattern.get('detection_method', 'Unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        qualities = [p.get('quality_score', 0) for p in self.pattern_history]
        avg_quality = np.mean(qualities) if qualities else 0
        
        total_patterns = len(self.pattern_history)
        recent_patterns = [p for p in self.pattern_history 
                          if datetime.fromisoformat(p['timestamp']) > datetime.now() - timedelta(days=7)]
        
        insights = f"""
        ### Анализ самообучения
        
        - **Всего паттернов обнаружено**: {total_patterns}
        - **Паттернов за последние 7 дней**: {len(recent_patterns)}
        - **Среднее качество паттернов**: {avg_quality:.2f}
        
        **Распределение по методам обнаружения:**
        """
        
        for method, count in method_counts.items():
            insights += f"\n- {method}: {count}"
        
        return insights.strip()

    def extract_pattern_features(self, df):
        try:
            features = []
            
            required_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0
            
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            feature_matrix = df[required_cols].values
            
            feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)
            
            return feature_matrix
        except Exception as e:
            print(f"Warning: Could not extract pattern features: {str(e)}")
            return np.array([]).reshape(0, 5)
    
    def describe_pattern(self, cluster_data, cluster_df):
        if len(cluster_data) == 0:
            return "Unknown pattern"
        
        avg_amount = cluster_df['amount'].mean()
        avg_old_balance = cluster_df['oldbalanceOrg'].mean()
        depletion_rate = ((cluster_df['oldbalanceOrg'] - cluster_df['newbalanceOrig']) / 
                         (cluster_df['oldbalanceOrg'] + 1e-8)).mean()
        
        if avg_amount > 10000:
            amount_desc = "very large transactions"
        elif avg_amount > 1000:
            amount_desc = "large transactions"
        else:
            amount_desc = "moderate transactions"
        
        if depletion_rate > 0.9:
            balance_desc = "near complete account depletion"
        elif depletion_rate > 0.5:
            balance_desc = "significant account depletion"
        else:
            balance_desc = "partial account usage"
        
        return f"{amount_desc} with {balance_desc}"
    
    def adapt_rules_based_on_patterns(self, new_patterns):
        adapted_rules = []
        
        for pattern in new_patterns:
            rule = {
                'rule_id': f"adaptive_{pattern['pattern_id']}",
                'description': f"Adaptive detection for {pattern['characteristics']}",
                'conditions': self.generate_adaptive_conditions(pattern),
                'confidence': pattern['confidence'],
                'active': True
            }
            
            adapted_rules.append(rule)
            
            self.performance_metrics['adaptation_events'].append({
                'timestamp': datetime.now().isoformat(),
                'pattern_id': pattern['pattern_id'],
                'rule_id': rule['rule_id']
            })
        
        return adapted_rules
    
    def generate_adaptive_conditions(self, pattern):
        return {
            'min_confidence': pattern['confidence'],
            'pattern_match_required': True
        }
    
    def update_performance_metrics(self, df, is_suspicious, ground_truth=None):
        metrics = {}
        
        if ground_truth is not None and 'isFraud' in df.columns:
            true_fraud = df['isFraud'].astype(bool)
            detected_fraud = is_suspicious.astype(bool)
            
            if true_fraud.sum() > 0:
                detection_rate = (detected_fraud & true_fraud).sum() / true_fraud.sum()
                metrics['detection_rate'] = float(detection_rate)
                self.performance_metrics['detection_rates'].append(detection_rate)
            
            if non_fraud.sum() > 0:
                false_positive_rate = (detected_fraud & non_fraud).sum() / non_fraud.sum()
                metrics['false_positive_rate'] = float(false_positive_rate)
                self.performance_metrics['false_positive_rates'].append(false_positive_rate)
        
        return metrics
    
    def save_learning(self):
        try:
            learning_state = {
                'pattern_history': self.pattern_history,
                'performance_metrics': self.performance_metrics,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(os.path.join(self.model_storage_path, 'learning_state.json'), 'w') as f:
                json.dump(learning_state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save learning state: {str(e)}")
    
    def load_previous_learning(self):
        try:
            state_file = os.path.join(self.model_storage_path, 'learning_state.json')
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    learning_state = json.load(f)
                
                self.pattern_history = learning_state.get('pattern_history', [])
                self.performance_metrics = learning_state.get('performance_metrics', {
                    'detection_rates': [],
                    'false_positive_rates': [],
                    'adaptation_events': []
                })
        except Exception as e:
            print(f"Warning: Could not load previous learning state: {str(e)}")
    
    def get_system_status(self):
        return {
            'patterns_learned': len(self.pattern_history),
            'recent_detection_rate': np.mean(self.performance_metrics['detection_rates'][-10:]) if self.performance_metrics['detection_rates'] else 0,
            'recent_false_positive_rate': np.mean(self.performance_metrics['false_positive_rates'][-10:]) if self.performance_metrics['false_positive_rates'] else 0,
            'adaptations_made': len(self.performance_metrics['adaptation_events']),
            'last_adaptation': self.performance_metrics['adaptation_events'][-1]['timestamp'] if self.performance_metrics['adaptation_events'] else 'Never'
        }

self_learning_detector = SelfLearningFraudDetector()

def integrate_self_learning(df, fraud_scores, is_suspicious):
    try:
        if df.empty or len(fraud_scores) == 0 or len(is_suspicious) == 0:
            print("Self-learning: Empty data provided, skipping learning phase")
            return {
                'new_patterns': [],
                'metrics': {},
                'system_status': {}
            }
        
        if len(df) != len(fraud_scores) or len(df) != len(is_suspicious):
            raise ValueError("Несоответствие размеров данных для самообучения")
        
        new_patterns = self_learning_detector.detect_new_patterns(df, fraud_scores, is_suspicious)
        
        if new_patterns:
            adapted_rules = self_learning_detector.adapt_rules_based_on_patterns(new_patterns)
            print(f"Self-learning: Discovered {len(new_patterns)} new fraud patterns")
            print(f"Self-learning: Adapted {len(adapted_rules)} detection rules")
        
        metrics = self_learning_detector.update_performance_metrics(df, is_suspicious)
        
        self_learning_detector.save_learning()
        
        return {
            'new_patterns': new_patterns,
            'metrics': metrics,
            'system_status': self_learning_detector.get_system_status()
        }
    except Exception as e:
        print(f"Warning: Self-learning integration failed: {str(e)}")
        return {
            'new_patterns': [],
            'metrics': {},
            'system_status': {}
        }