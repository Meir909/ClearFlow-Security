import pandas as pd
import numpy as np

def rule_engine(df):
    try:
        if df.empty:
            raise ValueError("Пустой набор данных для применения правил.")
        

        required_columns = ['amount', 'oldbalanceDest', 'oldbalanceOrg', 'newbalanceOrig', 'step', 'nameOrig']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные столбцы для правил: {missing_columns}")
        
        
        if len(df) > 10000:
            sample_df = df.sample(n=min(10000, len(df)), random_state=42)
            large_amount_threshold = sample_df['amount'].quantile(0.95)
        else:
            large_amount_threshold = df['amount'].quantile(0.95)
        
        rules_flags = pd.DataFrame(index=df.index)
        rules_flags['rule_large_amount'] = (df['amount'] > large_amount_threshold).astype(np.int8)
        rules_flags['rule_new_destination'] = (df['oldbalanceDest'] == 0).astype(np.int8)
        rules_flags['rule_balance_depletion'] = ((df['oldbalanceOrg'] - df['newbalanceOrig']) / (df['oldbalanceOrg'] + 1e-6) > 0.9).astype(np.int8)
        rules_flags['rule_unusual_time'] = ((df['step'] % 24 >= 22) | (df['step'] % 24 <= 6)).astype(np.int8)
      
        if len(df) > 5000:
            
            mean_amount = df['amount'].mean()
            rules_flags['rule_velocity_spike'] = (df['amount'] > mean_amount * 3).astype(np.int8)
        else:
           
            rules_flags['rule_velocity_spike'] = (df['amount'] > df.groupby('nameOrig')['amount'].transform('mean') * 3).astype(np.int8)
        
        rules_flags['rule_round_amount'] = (df['amount'] % 1000 == 0).astype(np.int8)
        
      
        rules_combined = rules_flags.values.mean(axis=1)
        
        return rules_combined, rules_flags
    except Exception as e:
        print(f"Warning: Rule engine failed: {str(e)}")
        rules_combined = pd.Series(np.zeros(len(df)), index=df.index)
        rules_flags = pd.DataFrame(np.zeros((len(df), 6), dtype=np.int8), index=df.index, 
                                  columns=['rule_large_amount', 'rule_new_destination', 'rule_balance_depletion',
                                          'rule_unusual_time', 'rule_velocity_spike', 'rule_round_amount'])
        return rules_combined, rules_flags

def get_rule_explanations(rules_flags):
    
    rule_names = ["очень большая сумма", "перевод на новый счет", "опустошение счета", 
                  "необычное время", "резкий скачок активности", "круглая сумма"]
    
    explanations = []
    rule_columns = ['rule_large_amount', 'rule_new_destination', 'rule_balance_depletion',
                   'rule_unusual_time', 'rule_velocity_spike', 'rule_round_amount']
    

    for idx, row in rules_flags.iterrows():
        triggered_indices = [i for i, col in enumerate(rule_columns) if row[col]]
        if triggered_indices:
            triggered_rules = [rule_names[i] for i in triggered_indices]
            explanation = "Подозрительно из-за: " + ", ".join(triggered_rules)
        else:
            explanation = "Нет подозрительных признаков"
        explanations.append(explanation)
    
    return explanations