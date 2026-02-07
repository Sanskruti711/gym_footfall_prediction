# create_sample_files.py
import sqlite3
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Create tiny sample database (10 records)
conn = sqlite3.connect('data/sample.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS sample_occupancy (
    hour INTEGER,
    day_of_week INTEGER,
    occupancy REAL
)
''')

# Insert 10 sample records
for i in range(10):
    cursor.execute('INSERT INTO sample_occupancy VALUES (?, ?, ?)', 
                  (i+6, i%7, 50 + np.random.randint(-20, 20)))

conn.commit()
conn.close()

# 2. Create tiny sample model
X = np.array([[i] for i in range(10)])
y = np.array([50 + i*2 + np.random.randn() for i in range(10)])
model = LinearRegression()
model.fit(X, y)
joblib.dump(model, 'models/sample_model.pkl')

print("✅ Created sample files for GitHub")
