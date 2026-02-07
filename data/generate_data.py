"""
🎯 GYM FOOTFALL DATA GENERATOR
Generates 3 months of realistic gym occupancy data with your specified parameters
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_database():
    """Create SQL database with all tables"""
    conn = sqlite3.connect('data/gym.db')
    cursor = conn.cursor()
    
    # Main occupancy table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS occupancy_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        date DATE,
        hour INTEGER CHECK (hour BETWEEN 6 AND 22),
        day_of_week INTEGER CHECK (day_of_week BETWEEN 0 AND 6),
        exam_period BOOLEAN,
        temperature REAL,
        is_weekend BOOLEAN,
        special_event BOOLEAN,
        holiday BOOLEAN,
        sports_event BOOLEAN,
        new_term_start BOOLEAN,
        previous_day_occupancy REAL CHECK (previous_day_occupancy BETWEEN 0 AND 100),
        occupancy_percentage REAL CHECK (occupancy_percentage BETWEEN 0 AND 100)
    )
    ''')
    
    # Predictions history
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        hour INTEGER,
        day_of_week INTEGER,
        exam_period BOOLEAN,
        temperature REAL,
        is_weekend BOOLEAN,
        special_event BOOLEAN,
        holiday BOOLEAN,
        sports_event BOOLEAN,
        new_term_start BOOLEAN,
        previous_day_occupancy REAL,
        predicted_occupancy REAL,
        actual_occupancy REAL NULL,
        model_version VARCHAR(20)
    )
    ''')
    
    # Model versions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_versions (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name VARCHAR(50),
        model_type VARCHAR(20),
        training_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        accuracy_score REAL,
        is_active BOOLEAN DEFAULT 1
    )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database tables created successfully!")

def generate_realistic_data(num_days=90):
    """Generate 3 months of realistic gym data"""
    data = []
    start_date = datetime(2026, 1, 1)
    
    # Pre-defined holidays (simplified)
    holidays = [
        '2026-01-01', '2026-01-26', '2026-03-25', '2026-05-01',
        '2026-08-15', '2026-10-02', '2026-10-24', '2026-12-25'
    ]
    
    # Track previous day's occupancy for each hour
    prev_day_occupancy = {hour: 50 for hour in range(6, 23)}  # Default 50%
    
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        date_str = current_date.strftime('%Y-%m-%d')
        day_of_week = current_date.weekday()
        
        # --- DETERMINE DAY CHARACTERISTICS ---
        # Academic weeks (16-week semester)
        semester_week = (day // 7) % 16 + 1
        
        # Exam periods (weeks 6-7 and 13-15)
        exam_period = 1 if semester_week in [6, 7, 13, 14, 15] else 0
        
        # Weekend
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Holidays
        holiday = 1 if date_str in holidays else 0
        
        # Special events (once a week on average)
        special_event = 1 if random.random() < 0.15 else 0
        
        # Sports events (more frequent in certain weeks)
        if semester_week in [3, 4, 9, 10]:  # Sports weeks
            sports_event = 1 if random.random() < 0.3 else 0
        else:
            sports_event = 1 if random.random() < 0.1 else 0
        
        # New term start (first 2 weeks of semester)
        new_term_start = 1 if semester_week <= 2 else 0
        
        # Temperature (seasonal pattern)
        month = current_date.month
        if month in [12, 1, 2]:  # Winter
            base_temp = random.uniform(10, 20)
        elif month in [3, 4, 5]:  # Spring
            base_temp = random.uniform(20, 35)
        elif month in [6, 7, 8, 9]:  # Monsoon
            base_temp = random.uniform(25, 32)
        else:  # Oct-Nov
            base_temp = random.uniform(18, 30)
        
        temperature = base_temp + random.uniform(-3, 3)
        
        # Generate hourly data
        for hour in range(6, 23):  # 6 AM to 10 PM
            # --- CALCULATE OCCUPANCY WITH REALISTIC PATTERNS ---
            base_occupancy = 30  # Base occupancy
            
            # 1. HOUR OF DAY PATTERNS (MOST IMPORTANT!)
            if 6 <= hour < 9:  # Morning crew (6-9 AM)
                base_occupancy += random.randint(10, 20)  # 40-50%
            elif 9 <= hour < 12:  # Late morning
                base_occupancy += random.randint(0, 10)   # 30-40%
            elif 12 <= hour < 14:  # Lunch dip
                base_occupancy -= random.randint(10, 20)  # 10-20%
            elif 14 <= hour < 17:  # Afternoon
                base_occupancy += random.randint(5, 15)   # 35-45%
            elif 17 <= hour < 21:  # PRIME TIME (5-9 PM)
                base_occupancy += random.randint(30, 50)  # 60-80%
            else:  # Late evening (9-10 PM)
                base_occupancy += random.randint(5, 15)   # 35-45%
            
            # 2. DAY OF WEEK EFFECTS
            if day_of_week == 0:  # Monday
                base_occupancy += random.randint(5, 15)   # New week motivation
            elif day_of_week == 4:  # Friday
                base_occupancy -= random.randint(10, 20)  # Weekend plans
            elif is_weekend:
                # Weekends: Different pattern
                if hour < 10:
                    base_occupancy -= random.randint(15, 25)  # Sleep in
                elif 10 <= hour < 18:
                    base_occupancy += random.randint(0, 10)   # Steady flow
                else:
                    base_occupancy -= random.randint(5, 15)   # Less night crowd
            
            # 3. EXAM PERIOD EFFECT (students study, don't gym)
            if exam_period:
                base_occupancy -= random.randint(15, 25)
            
            # 4. TEMPERATURE EFFECT
            if temperature < 15:  # Too cold
                base_occupancy += random.randint(5, 15)  # Gym is warm
            elif temperature > 35:  # Too hot
                base_occupancy += random.randint(5, 15)  # Gym has AC
            else:  # Comfortable temperature
                base_occupancy -= random.randint(0, 5)   # Might go outside
            
            # 5. SPECIAL EVENT EFFECT
            if special_event:
                base_occupancy -= random.randint(10, 20)  # People at event
            
            # 6. HOLIDAY EFFECT
            if holiday:
                base_occupancy -= random.randint(20, 40)  # Campus empty
            
            # 7. SPORTS EVENT EFFECT
            if sports_event:
                # If sports event in evening, gym might be empty
                if 17 <= hour <= 20:
                    base_occupancy -= random.randint(15, 25)
            
            # 8. NEW TERM START EFFECT (New Year resolutions!)
            if new_term_start:
                base_occupancy += random.randint(10, 20)
            
            # 9. PREVIOUS DAY PATTERN (momentum)
            if prev_day_occupancy[hour] > 60:
                base_occupancy += random.randint(0, 5)   # Busy hour stays busy
            elif prev_day_occupancy[hour] < 30:
                base_occupancy -= random.randint(0, 5)   # Quiet hour stays quiet
            
            # Add randomness
            base_occupancy += random.randint(-8, 8)
            
            # Ensure realistic bounds
            occupancy = max(5, min(95, base_occupancy))
            
            # Update previous day occupancy for tomorrow
            prev_day_occupancy[hour] = occupancy
            
            data.append({
                'date': date_str,
                'hour': hour,
                'day_of_week': day_of_week,
                'exam_period': exam_period,
                'temperature': round(temperature, 1),
                'is_weekend': is_weekend,
                'special_event': special_event,
                'holiday': holiday,
                'sports_event': sports_event,
                'new_term_start': new_term_start,
                'previous_day_occupancy': round(prev_day_occupancy[hour], 1),
                'occupancy_percentage': round(occupancy, 1)
            })
    
    return pd.DataFrame(data)

def save_to_database(df):
    """Save generated data to SQL database"""
    conn = sqlite3.connect('data/gym.db')
    
    # Insert data
    for _, row in df.iterrows():
        conn.execute('''
        INSERT INTO occupancy_data 
        (date, hour, day_of_week, exam_period, temperature, is_weekend, 
         special_event, holiday, sports_event, new_term_start, 
         previous_day_occupancy, occupancy_percentage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['date'], row['hour'], row['day_of_week'], row['exam_period'],
            row['temperature'], row['is_weekend'], row['special_event'],
            row['holiday'], row['sports_event'], row['new_term_start'],
            row['previous_day_occupancy'], row['occupancy_percentage']
        ))
    
    conn.commit()
    
    # Verify
    count = conn.execute("SELECT COUNT(*) FROM occupancy_data").fetchone()[0]
    print(f"✅ Saved {len(df)} records to database")
    print(f"📊 Total records in database: {count}")
    
    # Show sample
    sample = pd.read_sql_query("SELECT * FROM occupancy_data LIMIT 5", conn)
    print("\n📋 Sample data:")
    print(sample[['date', 'hour', 'occupancy_percentage', 'exam_period', 'temperature']])
    
    conn.close()

def main():
    """Main function to generate and save data"""
    print("=" * 60)
    print("🏋️  GYM FOOTFALL DATA GENERATOR")
    print("=" * 60)
    
    # Create database
    create_database()
    
    # Generate data
    print("\n📈 Generating realistic gym occupancy data...")
    df = generate_realistic_data(num_days=90)  # 3 months data
    
    # Save to database
    save_to_database(df)
    
    # Also save as CSV for backup
    df.to_csv('data/gym_occupancy.csv', index=False)
    print(f"\n💾 Backup CSV saved: data/gym_occupancy.csv ({len(df)} records)")
    
    print("\n" + "=" * 60)
    print("✅ DATA GENERATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
