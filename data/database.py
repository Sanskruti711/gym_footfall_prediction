"""
📊 DATABASE OPERATIONS CLASS
Handles all SQL operations for the gym system
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta

class GymDatabase:
    def __init__(self, db_path='data/gym.db'):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            return True
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return False
    
    def get_training_data(self, limit=None):
        """Get data for model training"""
        query = """
        SELECT 
            hour, day_of_week, exam_period, temperature,
            is_weekend, special_event, holiday, sports_event,
            new_term_start, previous_day_occupancy,
            occupancy_percentage
        FROM occupancy_data
        """
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            print(f"❌ Error getting training data: {e}")
            return pd.DataFrame()
    
    def save_prediction(self, features, prediction, model_version="v1.0"):
        """Save a prediction to history"""
        query = """
        INSERT INTO predictions 
        (hour, day_of_week, exam_period, temperature, is_weekend,
         special_event, holiday, sports_event, new_term_start,
         previous_day_occupancy, predicted_occupancy, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (
                features['hour'], features['day_of_week'], features['exam_period'],
                features['temperature'], features['is_weekend'], features['special_event'],
                features['holiday'], features['sports_event'], features['new_term_start'],
                features['previous_day_occupancy'], prediction, model_version
            ))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"❌ Error saving prediction: {e}")
            return None
    
    def get_recent_predictions(self, limit=50):
        """Get recent predictions for dashboard"""
        query = f"""
        SELECT * FROM predictions 
        ORDER BY prediction_time DESC 
        LIMIT {limit}
        """
        try:
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            print(f"❌ Error getting predictions: {e}")
            return pd.DataFrame()
    
    def add_model_version(self, model_name, model_type, accuracy):
        """Register a new model version"""
        query = """
        INSERT INTO model_versions (model_name, model_type, accuracy_score)
        VALUES (?, ?, ?)
        """
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (model_name, model_type, accuracy))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"❌ Error adding model version: {e}")
            return None
    
    def get_active_models(self):
        """Get currently active models"""
        query = """
        SELECT * FROM model_versions 
        WHERE is_active = 1 
        ORDER BY training_date DESC
        """
        try:
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            print(f"❌ Error getting active models: {e}")
            return pd.DataFrame()
    
    def get_weekly_summary(self):
        """Get weekly occupancy summary for heatmap"""
        query = """
        SELECT 
            day_of_week,
            hour,
            AVG(occupancy_percentage) as avg_occupancy,
            COUNT(*) as record_count
        FROM occupancy_data
        GROUP BY day_of_week, hour
        ORDER BY day_of_week, hour
        """
        try:
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            print(f"❌ Error getting weekly summary: {e}")
            return pd.DataFrame()
    
    def get_daily_stats(self):
        """Get daily statistics for dashboard"""
        query = """
        SELECT 
            date,
            AVG(occupancy_percentage) as daily_avg,
            MAX(occupancy_percentage) as daily_max,
            MIN(occupancy_percentage) as daily_min,
            COUNT(*) as record_count
        FROM occupancy_data
        GROUP BY date
        ORDER BY date DESC
        LIMIT 30
        """
        try:
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            print(f"❌ Error getting daily stats: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
