import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, low_memory=False)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Неподдерживаемый формат файла. Используйте CSV или Excel файлы.")
        
        required_columns = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                          'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing_columns}")
        
     
        if df.empty:
            raise ValueError("Файл пустой. Проверьте содержимое файла.")
        
      
        if len(df) < 2:
            raise ValueError("Недостаточно данных для анализа. Файл должен содержать минимум 2 строки.")
        
        return df
    except FileNotFoundError:
        raise Exception(f"Файл не найден: {file_path}")
    except pd.errors.EmptyDataError:
        raise Exception("Файл пустой или поврежден.")
    except pd.errors.ParserError:
        raise Exception("Ошибка парсинга файла. Проверьте формат данных.")
    except Exception as e:
        raise Exception(f"Ошибка загрузки данных: {str(e)}")

def preprocess(df):
    try:
        if df.empty:
            raise ValueError("Пустой набор данных для предобработки.")
        
      
        if len(df) > 50000:
       
            sample_size = min(50000, len(df))
            df = df.sample(n=sample_size, random_state=42)
        
        df_processed = df.copy()
        
        
        required_columns = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                          'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
        missing_columns = [col for col in required_columns if col not in df_processed.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing_columns}")
        
     
        numeric_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        for col in numeric_columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        

        if df_processed[numeric_columns].isnull().all().all():
            raise ValueError("Все числовые столбцы содержат некорректные данные.")
        
 
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(0)
        
       
        df_processed[numeric_columns] = df_processed[numeric_columns].abs()
        

        df_processed['errorBalanceOrig'] = df_processed['oldbalanceOrg'] - df_processed['newbalanceOrig'] - df_processed['amount']
        df_processed['errorBalanceDest'] = df_processed['newbalanceDest'] - df_processed['oldbalanceDest'] - df_processed['amount']
        
    
        df_processed['hour'] = df_processed['step'] % 24
        df_processed['day_of_week'] = (df_processed['step'] // 24) % 7
        

        if 'isFraud' in df_processed.columns:
            df_processed['isFraud'] = pd.to_numeric(df_processed['isFraud'], errors='coerce').fillna(0)
        
        if 'isFlaggedFraud' in df_processed.columns:
            df_processed['isFlaggedFraud'] = pd.to_numeric(df_processed['isFlaggedFraud'], errors='coerce').fillna(0)
        

        if 'type' in df_processed.columns:
            type_encoder = LabelEncoder()
            df_processed['type_encoded'] = type_encoder.fit_transform(df_processed['type'])
        

        df_processed = df_processed.fillna(0)
        
 
        if df_processed.empty:
            raise ValueError("Предобработка привела к пустому набору данных.")
        
        return df_processed
    except ValueError as ve:
        raise ValueError(f"Ошибка предобработки данных: {str(ve)}")
    except Exception as e:
        raise Exception(f"Ошибка предобработки: {str(e)}")