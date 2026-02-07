"""
🤖 MODEL TRAINING SCRIPT
Trains Linear Regression and Random Forest models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime
import sys
sys.path.append('.')
from data.database import GymDatabase

def prepare_features(df):
    """Prepare features for training"""
    # No encoding needed - all features are numerical!
    features = [
        'hour', 'day_of_week', 'exam_period', 'temperature',
        'is_weekend', 'special_event', 'holiday', 'sports_event',
        'new_term_start', 'previous_day_occupancy'
    ]
    
    X = df[features]
    y = df['occupancy_percentage']
    
    return X, y, features

def train_models():
    """Train and evaluate models"""
    print("=" * 60)
    print("🤖 GYM OCCUPANCY MODEL TRAINING")
    print("=" * 60)
    
    # Connect to database
    db = GymDatabase('data/gym.db')
    if not db.connect():
        return None
    
    # Get training data
    print("📥 Loading data from SQL database...")
    df = db.get_training_data()
    
    if len(df) < 100:
        print("❌ Not enough data for training")
        return None
    
    print(f"✅ Loaded {len(df)} records")
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Training set: {len(X_train)} records")
    print(f"📊 Testing set: {len(X_test)} records")
    
    # --- TRAIN LINEAR REGRESSION ---
    print("\n" + "-" * 40)
    print("1. Training Linear Regression...")
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    
    print(f"   📈 Mean Absolute Error: {lr_mae:.2f}%")
    print(f"   📊 R² Score: {lr_r2:.3f}")
    print(f"   🔢 Coefficients: {len(lr_model.coef_)} features")
    
    # --- TRAIN RANDOM FOREST ---
    print("\n2. Training Random Forest Regressor...")
    
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"   📈 Mean Absolute Error: {rf_mae:.2f}%")
    print(f"   📊 R² Score: {rf_r2:.3f}")
    print(f"   🌳 Number of trees: {len(rf_model.estimators_)}")
    
    # --- FEATURE IMPORTANCE ---
    print("\n3. Feature Importance (Random Forest):")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        print(f"   {row['feature']:25} {row['importance']:.3f}")
    
    # --- SAVE MODELS ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Save models
    joblib.dump(lr_model, 'models/linear_model.pkl')
    joblib.dump(rf_model, 'models/random_forest.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Save metadata
    metadata = {
        'training_date': timestamp,
        'data_samples': len(df),
        'test_samples': len(X_test),
        'models': {
            'linear_regression': {
                'mae': float(lr_mae),
                'r2': float(lr_r2),
                'file': 'models/linear_model.pkl',
                'coefficients': lr_model.coef_.tolist(),
                'intercept': float(lr_model.intercept_)
            },
            'random_forest': {
                'mae': float(rf_mae),
                'r2': float(rf_r2),
                'file': 'models/random_forest.pkl',
                'feature_importance': feature_importance.to_dict('records')
            }
        },
        'features': feature_names
    }
    
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Register models in database
    db.add_model_version(
        f"Linear_Regression_{timestamp}",
        "linear_regression",
        lr_r2
    )
    
    db.add_model_version(
        f"Random_Forest_{timestamp}",
        "random_forest",
        rf_r2
    )
    
    # --- DISPLAY RESULTS ---
    print("\n" + "=" * 60)
    print("🏆 MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    results_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest'],
        'MAE (%)': [lr_mae, rf_mae],
        'R² Score': [lr_r2, rf_r2],
        'Interpretation': [
            f'Linear patterns, interpretable coefficients',
            f'Non-linear patterns, feature importance available'
        ]
    })
    
    print(results_df.to_string(index=False))
    
    # Determine best model
    if lr_r2 > rf_r2:
        best_model = "Linear Regression"
        best_mae = lr_mae
        best_r2 = lr_r2
    else:
        best_model = "Random Forest"
        best_mae = rf_mae
        best_r2 = rf_r2
    
    print(f"\n✅ **Best Model**: {best_model}")
    print(f"🎯 **Average Error**: ±{best_mae:.1f}% occupancy")
    print(f"📊 **Prediction Power**: {best_r2:.1%} variance explained")
    
    # Show active models
    active_models = db.get_active_models()
    if not active_models.empty:
        print(f"\n📋 Active models in database: {len(active_models)}")
    
    db.close()
    
    return metadata

if __name__ == "__main__":
    train_models()
