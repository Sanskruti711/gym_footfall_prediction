"""
🎯 PREDICTION FUNCTIONS
Load models and make predictions
"""

import joblib
import numpy as np
import pandas as pd

class GymPredictor:
    def __init__(self):
        """Initialize predictor with loaded models"""
        try:
            self.lr_model = joblib.load('models/linear_model.pkl')
            self.rf_model = joblib.load('models/random_forest.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            self.models_loaded = True
        except:
            self.models_loaded = False
    
    def predict(self, features_dict):
        """
        Predict gym occupancy percentage
        
        Args:
            features_dict: Dictionary with feature values
                Required keys: hour, day_of_week, exam_period, temperature,
                              is_weekend, special_event, holiday, sports_event,
                              new_term_start, previous_day_occupancy
        
        Returns:
            Dictionary with predictions from both models
        """
        if not self.models_loaded:
            return {"error": "Models not loaded"}
        
        # Convert features to array in correct order
        feature_values = []
        for feature in self.feature_names:
            if feature in features_dict:
                feature_values.append(features_dict[feature])
            else:
                feature_values.append(0)  # Default value
        
        # Reshape for prediction
        X = np.array(feature_values).reshape(1, -1)
        
        # Make predictions
        lr_pred = self.lr_model.predict(X)[0]
        rf_pred = self.rf_model.predict(X)[0]
        avg_pred = (lr_pred + rf_pred) / 2
        
        # Determine crowd level
        if avg_pred < 30:
            crowd_level = "Low"
            recommendation = "Perfect time to workout!"
            color = "green"
        elif avg_pred < 60:
            crowd_level = "Moderate"
            recommendation = "Some equipment available"
            color = "orange"
        else:
            crowd_level = "High"
            recommendation = "Consider another time"
            color = "red"
        
        return {
            'linear_regression': round(lr_pred, 1),
            'random_forest': round(rf_pred, 1),
            'average': round(avg_pred, 1),
            'crowd_level': crowd_level,
            'recommendation': recommendation,
            'color': color,
            'features_used': features_dict
        }
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest"""
        if not self.models_loaded:
            return None
        
        # Extract feature importance
        importance = self.rf_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
