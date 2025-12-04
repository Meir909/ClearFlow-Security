import sqlite3
import hashlib
import pandas as pd
import os
from datetime import datetime

class UserDatabase:
    def __init__(self, db_path="user_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the user database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                position TEXT,
                age INTEGER,
                company TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
       
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_analysis_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                file_name TEXT,
                file_path TEXT,
                location_permission BOOLEAN DEFAULT FALSE,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Hash a password for storing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username, password, full_name, position, age, company):
        """Register a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (username, password_hash, full_name, position, age, company)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, password_hash, full_name, position, age, company))
            
            conn.commit()
            conn.close()
            return True, "User registered successfully"
        except sqlite3.IntegrityError:
            return False, "Username already exists"
        except Exception as e:
            return False, f"Registration error: {str(e)}"
    
    def authenticate_user(self, username, password):
        """Authenticate a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                SELECT id, full_name, position, age, company FROM users 
                WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            
            user = cursor.fetchone()
            conn.close()
            
            if user:
               
                self.update_last_login(username)
                return True, {
                    'id': user[0],
                    'full_name': user[1],
                    'position': user[2],
                    'age': user[3],
                    'company': user[4]
                }
            else:
                return False, "Invalid username or password"
        except Exception as e:
            return False, f"Authentication error: {str(e)}"
    
    def update_last_login(self, username):
        """Update user's last login timestamp"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP 
                WHERE username = ?
            ''', (username,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error updating last login: {str(e)}")
    
    def store_user_analysis_data(self, user_id, file_name, file_path, location_permission=False):
        """Store user analysis data with location permission info"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_analysis_data (user_id, file_name, file_path, location_permission)
                VALUES (?, ?, ?, ?)
            ''', (user_id, file_name, file_path, location_permission))
            
            conn.commit()
            conn.close()
            return True, "Analysis data stored successfully"
        except Exception as e:
            return False, f"Error storing analysis data: {str(e)}"
    
    def get_user_analysis_history(self, user_id):
        """Get user's analysis history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_name, file_path, location_permission, analysis_timestamp 
                FROM user_analysis_data 
                WHERE user_id = ? 
                ORDER BY analysis_timestamp DESC
            ''', (user_id,))
            
            history = cursor.fetchall()
            conn.close()
            
            return history
        except Exception as e:
            print(f"Error retrieving analysis history: {str(e)}")
            return []
    
    def update_location_permission(self, user_id, file_name, permission):
        """Update location permission for a specific analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_analysis_data 
                SET location_permission = ? 
                WHERE user_id = ? AND file_name = ?
            ''', (permission, user_id, file_name))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating location permission: {str(e)}")
            return False


user_db = UserDatabase()