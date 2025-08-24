from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained model (we'll create a simple one if the saved model doesn't exist)
def create_simple_model():
    """Create a simple model if the saved model doesn't exist"""
    try:
        # Try to load the saved model first
        model = joblib.load('optimized_insurance_model.pkl')
        return model
    except:
        # If model doesn't exist, create a simple one with dummy data
        print("Creating a simple model for demonstration...")
        
        # Create sample data for training a basic model
        np.random.seed(42)
        sample_size = 1000
        
        sample_data = {
            'age': np.random.randint(18, 65, sample_size),
            'bmi': np.random.normal(28, 6, sample_size),
            'children': np.random.randint(0, 4, sample_size),
            'smoker_bmi_interaction': np.random.normal(0, 10, sample_size),
            'sex_male': np.random.choice([0, 1], sample_size),
            'smoker_yes': np.random.choice([0, 1], sample_size),
            'region_northwest': np.random.choice([0, 1], sample_size),
            'region_southeast': np.random.choice([0, 1], sample_size),
            'region_southwest': np.random.choice([0, 1], sample_size),
        }
        
        X_sample = pd.DataFrame(sample_data)
        # Create target with some correlation to features
        y_sample = (X_sample['age'] * 100 + 
                   X_sample['bmi'] * 200 + 
                   X_sample['smoker_yes'] * 20000 + 
                   np.random.normal(0, 1000, sample_size))
        
        # Create and train a simple model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_sample, y_sample)
        
        return model

# Load or create model
model = create_simple_model()

# Initialize database with sample data
def init_db():
    conn = sqlite3.connect('insurance.db')
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS insurance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            sex TEXT,
            bmi REAL,
            children INTEGER,
            smoker TEXT,
            region TEXT,
            charges REAL
        )
    ''')
    
    # Check if table has data
    cursor.execute('SELECT COUNT(*) FROM insurance')
    count = cursor.fetchone()[0]
    
    # If no data, insert sample data
    if count == 0:
        sample_data = [
            (25, 'female', 22.5, 0, 'no', 'southwest', 3000),
            (45, 'male', 30.0, 2, 'yes', 'southeast', 35000),
            (35, 'female', 25.0, 1, 'no', 'northwest', 8000),
            (55, 'male', 28.5, 3, 'yes', 'northeast', 40000),
            (30, 'female', 24.0, 0, 'no', 'southwest', 4500),
        ]
        
        cursor.executemany('''
            INSERT INTO insurance (age, sex, bmi, children, smoker, region, charges)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', sample_data)
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

def get_db_stats():
    """Get database statistics for the dashboard"""
    conn = sqlite3.connect('insurance.db')
    cursor = conn.cursor()
    
    stats = {}
    
    # Total policyholders
    cursor.execute('SELECT COUNT(*) FROM insurance')
    stats['total_policyholders'] = cursor.fetchone()[0]
    
    # Average charges by region
    cursor.execute('''
        SELECT region, AVG(charges) as avg_charges 
        FROM insurance 
        GROUP BY region 
        ORDER BY avg_charges DESC
    ''')
    stats['region_avg'] = cursor.fetchall()
    
    # Average charges by smoking status
    cursor.execute('''
        SELECT smoker, AVG(charges) as avg_charges 
        FROM insurance 
        GROUP BY smoker
    ''')
    stats['smoker_avg'] = cursor.fetchall()
    
    # Highest claims by region
    cursor.execute('''
        SELECT region, MAX(charges) as max_charges 
        FROM insurance 
        GROUP BY region
    ''')
    stats['region_max'] = cursor.fetchall()
    
    conn.close()
    return stats

def prepare_features(age, sex, bmi, children, smoker, region):
    """Prepare features for model prediction - matches notebook exactly"""
    # Calculate BMI interaction
    smoker_bmi_interaction = bmi if smoker == 'yes' else 0
    
    # BMI Category (exactly like in notebook)
    if bmi < 18.5:
        bmi_category = 'Underweight'
    elif bmi <= 24.9:
        bmi_category = 'Normal weight'
    elif bmi <= 29.9:
        bmi_category = 'Overweight'
    else:
        bmi_category = 'Obese'
    
    # Age Group (exactly like in notebook)
    if age < 18:
        age_group = 'Youth'
    elif age < 35:
        age_group = 'Young Adult'
    elif age < 50:
        age_group = 'Middle Aged'
    else:
        age_group = 'Senior'
    
    # Create feature dictionary with ALL expected columns
    features = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'smoker_bmi_interaction': smoker_bmi_interaction,
        
        # Sex (drop_first=True means only male is kept)
        'sex_male': 1 if sex == 'male' else 0,
        
        # Smoker (drop_first=True means only yes is kept)
        'smoker_yes': 1 if smoker == 'yes' else 0,
        
        # Region (drop_first=True means northeast is dropped)
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0,
        
        # BMI Category (drop_first=True means Underweight is dropped)
        'bmi_category_Normal weight': 1 if bmi_category == 'Normal weight' else 0,
        'bmi_category_Obese': 1 if bmi_category == 'Obese' else 0,
        'bmi_category_Overweight': 1 if bmi_category == 'Overweight' else 0,
        
        # Age Group (drop_first=True means Youth is dropped)
        'age_group_Middle Aged': 1 if age_group == 'Middle Aged' else 0,
        'age_group_Senior': 1 if age_group == 'Senior' else 0,
        'age_group_Young Adult': 1 if age_group == 'Young Adult' else 0,
    }
    
    return pd.DataFrame([features])

@app.route('/')
def home():
    stats = get_db_stats()
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        
        # Prepare features
        features_df = prepare_features(age, sex, bmi, children, smoker, region)
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Round to 2 decimal places
        prediction = round(prediction, 2)
        
        return render_template('result.html', 
                             prediction=prediction,
                             age=age, sex=sex, bmi=bmi, 
                             children=children, smoker=smoker, region=region)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/analytics')
def analytics():
    stats = get_db_stats()
    return render_template('analytics.html', stats=stats)

if __name__ == '__main__':
    app.run(debug=True)