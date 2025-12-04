import json
import os
import hashlib
from datetime import datetime

class UserPreferences:
    
    def __init__(self, prefs_dir="user_preferences"):
        self.prefs_dir = prefs_dir
        self.current_prefs = {}
        
        if not os.path.exists(prefs_dir):
            os.makedirs(prefs_dir)
    
    def get_user_id(self):
        system_info = f"{os.name}_{os.getcwd()}"
        return hashlib.md5(system_info.encode()).hexdigest()[:16]
    
    def load_preferences(self):
        user_id = self.get_user_id()
        prefs_file = os.path.join(self.prefs_dir, f"prefs_{user_id}.json")
        
        try:
            if os.path.exists(prefs_file):
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    self.current_prefs = json.load(f)
            else:
                self.current_prefs = self.get_default_preferences()
        except Exception as e:
            print(f"Ошибка при загрузке настроек: {str(e)}")
            self.current_prefs = self.get_default_preferences()
        
        return self.current_prefs
    
    def save_preferences(self, prefs=None):
        if prefs:
            self.current_prefs = prefs
        
        user_id = self.get_user_id()
        prefs_file = os.path.join(self.prefs_dir, f"prefs_{user_id}.json")
        
        try:
            with open(prefs_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_prefs, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Ошибка при сохранении настроек: {str(e)}")
            return False
    
    def get_default_preferences(self):
        return {
            "theme": "dark",
            "language": "ru",
            "default_models": ["isolation_forest", "autoencoder"],
            "contamination_level": 0.05,
            "favorite_views": [],
            "alert_thresholds": {
                "high_risk": 0.8,
                "medium_risk": 0.5,
                "low_risk": 0.2
            },
            "export_formats": ["csv", "json"],
            "dashboard_layout": "default",
            "last_updated": datetime.now().isoformat(),
            "theme_transition_speed": 0.5,
            "enable_animations": True
        }
    
    def update_preference(self, key, value):
        self.current_prefs[key] = value
        self.current_prefs["last_updated"] = datetime.now().isoformat()
        return self.save_preferences()
    
    def get_preference(self, key, default=None):
        return self.current_prefs.get(key, default)
    
    def add_favorite_view(self, view_name):
        favorites = self.current_prefs.get("favorite_views", [])
        if view_name not in favorites:
            favorites.append(view_name)
            self.current_prefs["favorite_views"] = favorites
            self.current_prefs["last_updated"] = datetime.now().isoformat()
            return self.save_preferences()
        return True
    
    def remove_favorite_view(self, view_name):
        favorites = self.current_prefs.get("favorite_views", [])
        if view_name in favorites:
            favorites.remove(view_name)
            self.current_prefs["favorite_views"] = favorites
            self.current_prefs["last_updated"] = datetime.now().isoformat()
            return self.save_preferences()
        return True

user_prefs = UserPreferences()