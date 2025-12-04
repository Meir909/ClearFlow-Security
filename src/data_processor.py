import pandas as pd
import numpy as np
import os
import hashlib
import json
from datetime import datetime, timedelta
import threading
import time

class DataProcessor:
    
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        self.processing_lock = threading.Lock()
        self.cache_metadata = {}
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self.load_cache_metadata()
    
    def get_file_hash(self, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def load_cache_metadata(self):
        metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.cache_metadata = json.load(f)
        except Exception as e:
            print(f"Ошибка при загрузке метаданных кэша: {str(e)}")
            self.cache_metadata = {}
    
    def save_cache_metadata(self):
        metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            print(f"Ошибка при сохранении метаданных кэша: {str(e)}")
    
    def is_cached(self, file_path):
        file_hash = self.get_file_hash(file_path)
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.pkl")
        
        if os.path.exists(cache_file):
            cache_time = os.path.getmtime(cache_file)
            if datetime.now() - datetime.fromtimestamp(cache_time) < timedelta(hours=24):
                return True
        return False
    
    def load_from_cache(self, file_path):
        try:
            file_hash = self.get_file_hash(file_path)
            cache_file = os.path.join(self.cache_dir, f"{file_hash}.pkl")
            
            if os.path.exists(cache_file):
                return None
        except Exception as e:
            print(f"Ошибка при загрузке из кэша: {str(e)}")
        return None
    
    def save_to_cache(self, file_path, processed_data):
        try:
            file_hash = self.get_file_hash(file_path)
            cache_file = os.path.join(self.cache_dir, f"{file_hash}.pkl")
            
            self.cache_metadata[file_hash] = {
                "original_file": file_path,
                "cached_at": datetime.now().isoformat(),
                "size": os.path.getsize(file_path)
            }
            
            self.save_cache_metadata()
            return True
        except Exception as e:
            print(f"Ошибка при сохранении в кэш: {str(e)}")
            return False
    
    def process_large_file(self, file_path, chunk_size=10000):
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                chunks = []
                chunk_count = 0
                
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    chunks.append(chunk)
                    chunk_count += 1
                    
                    if chunk_count > 100:
                        break
                
                if chunks:
                    return pd.concat(chunks, ignore_index=True)
                    
            elif file_extension == '.xlsx':
                xl_file = pd.ExcelFile(file_path)
                sheets = []
                
                for sheet_name in xl_file.sheet_names[:5]:
                    df = xl_file.parse(sheet_name)
                    sheets.append(df)
                
                if sheets:
                    return pd.concat(sheets, ignore_index=True)
            
            return pd.read_csv(file_path) if file_extension == '.csv' else pd.read_excel(file_path)
            
        except Exception as e:
            print(f"Ошибка при обработке большого файла: {str(e)}")
            return None
    
    def get_file_preview(self, file_path, preview_rows=100):
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                return pd.read_csv(file_path, nrows=preview_rows)
            elif file_extension == '.xlsx':
                xl_file = pd.ExcelFile(file_path)
                first_sheet = xl_file.sheet_names[0]
                return xl_file.parse(first_sheet, nrows=preview_rows)
        except Exception as e:
            print(f"Ошибка при получении предпросмотра: {str(e)}")
            return None
    
    def monitor_memory_usage(self):
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss / 1024 / 1024,
                "vms": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except:
            return {"rss": 0, "vms": 0, "percent": 0}

data_processor = DataProcessor()