import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import hashlib

class AdvancedVisualizations:
    
    @staticmethod
    def create_fraud_heatmap(df, is_suspicious, time_column='step', amount_column='amount'):
        try:
            df_viz = df.copy()
            df_viz['is_suspicious'] = is_suspicious
            df_viz['hour'] = df_viz[time_column] % 24
            df_viz['day_of_week'] = (df_viz[time_column] // 24) % 7
            
            heatmap_data = df_viz[df_viz['is_suspicious'] == 1].groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
            
            for day in range(7):
                if day not in heatmap_data.index:
                    heatmap_data.loc[day] = 0
            for hour in range(24):
                if hour not in heatmap_data.columns:
                    heatmap_data[hour] = 0
                    
            heatmap_data = heatmap_data.sort_index().sort_index(axis=1)
            
            plt.figure(figsize=(12, 6))
            sns.heatmap(heatmap_data, 
                       cmap='RdYlBu_r',
                       annot=True, 
                       fmt='d',
                       cbar_kws={'label': 'Количество подозрительных транзакций'})
            plt.title('Тепловая карта мошенничества по дням недели и часам')
            plt.xlabel('Час дня')
            plt.ylabel('День недели (0=Понедельник)')
            
            return plt.gcf()
        except Exception as e:
            print(f"Ошибка при создании тепловой карты: {str(e)}")
            return None
    
    @staticmethod
    def create_amount_distribution_chart(df, fraud_scores, is_suspicious):
        try:
            plt.figure(figsize=(12, 6))
            
            normal_amounts = df[~is_suspicious.astype(bool)]['amount']
            fraud_amounts = df[is_suspicious.astype(bool)]['amount']
            
            plt.hist(normal_amounts, bins=50, alpha=0.7, label='Нормальные транзакции', color='blue')
            plt.hist(fraud_amounts, bins=50, alpha=0.7, label='Подозрительные транзакции', color='red')
            
            plt.xlabel('Сумма транзакции')
            plt.ylabel('Частота')
            plt.title('Распределение сумм транзакций')
            plt.legend()
            plt.yscale('log')
            
            return plt.gcf()
        except Exception as e:
            print(f"Ошибка при создании распределения сумм: {str(e)}")
            return None
    
    @staticmethod
    def create_risk_timeline(df, fraud_scores, is_suspicious):
        try:
            plt.figure(figsize=(12, 6))
            
            df_sorted = df.copy()
            df_sorted['fraud_score'] = fraud_scores
            df_sorted['is_suspicious'] = is_suspicious
            df_sorted = df_sorted.sort_values('step')
            
            plt.plot(df_sorted['step'], df_sorted['fraud_score'], 
                    alpha=0.7, linewidth=1, color='purple')
            plt.scatter(df_sorted[df_sorted['is_suspicious'] == 1]['step'], 
                       df_sorted[df_sorted['is_suspicious'] == 1]['fraud_score'],
                       color='red', alpha=0.7, s=20, label='Подозрительные')
            plt.scatter(df_sorted[df_sorted['is_suspicious'] == 0]['step'], 
                       df_sorted[df_sorted['is_suspicious'] == 0]['fraud_score'],
                       color='blue', alpha=0.3, s=10, label='Нормальные')
            
            plt.xlabel('Временной шаг')
            plt.ylabel('Fraud Score')
            plt.title('Временная шкала рисков')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            return plt.gcf()
        except Exception as e:
            print(f"Ошибка при создании временной шкалы: {str(e)}")
            return None

advanced_viz = AdvancedVisualizations()