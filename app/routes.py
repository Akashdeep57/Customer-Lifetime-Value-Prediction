from flask import render_template, request
from app import app
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'online_retail_II.csv')
data = pd.read_csv(data_path)

# Convert InvoiceDate to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Drop duplicates
data.drop_duplicates(inplace=True)

# Handle missing values
data.dropna(subset=['Customer ID'], inplace=True)

# Calculate Total Amount for each transaction
data['Total Amount'] = data['Quantity'] * data['Price']

# Set reference date
now = dt.datetime.now()

# Feature Engineering
rfm_table = data.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (now - x.max()).days,
    'Invoice': 'nunique',
    'Total Amount': 'sum'
}).reset_index()

rfm_table.rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Total Amount': 'Monetary'}, inplace=True)

rfm_table['Tenure'] = data.groupby('Customer ID')['InvoiceDate'].apply(lambda x: (x.max() - x.min()).days).values

# Prepare data for modeling
features = ['Recency', 'Frequency', 'Monetary', 'Tenure']
X = rfm_table[features]
y = rfm_table['Monetary']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict function
def predict_clv(recency, frequency, monetary, tenure):
    input_data = pd.DataFrame([[recency, frequency, monetary, tenure]], columns=features)
    prediction = model.predict(input_data)
    return prediction[0]

# Routes
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    recency = float(request.form['recency'])
    frequency = float(request.form['frequency'])
    monetary = float(request.form['monetary'])
    tenure = float(request.form['tenure'])
    
    prediction = predict_clv(recency, frequency, monetary, tenure)
    
    return render_template('result.html', prediction=prediction)
