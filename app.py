from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained models
rf_model = joblib.load('models/random_forest_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')

try:
    scaler = joblib.load('models/scaler.pkl')
except FileNotFoundError:
    scaler = None
    print("Scaler not found. Please ensure the scaler is saved during model training.")

# Load the historical data for reference
df = pd.read_csv('D:/Crypto Currency Price Prediction/data/crypto_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Preprocess input for predicting future price
def preprocess_future_input():
    recent_data = df[['Open', 'High', 'Low', 'price']].values[-1].reshape(1, -1)
    
    if scaler is not None:
        recent_data = scaler.transform(recent_data)
    else:
        print("Scaler is not available. Proceeding without scaling.")
    
    return recent_data

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        days_into_future = int(request.form['days'])
    except ValueError:
        return render_template('index.html', prediction="Invalid input. Please enter a valid number of days.", days=None)

    # Preprocess the most recent input data
    recent_data = preprocess_future_input()

    # List to store predictions for each day
    future_prices = []

    # Predict the future price for the specified number of days
    future_price_rf = recent_data.copy()  
    future_price_knn = recent_data.copy()  

    for day in range(1, days_into_future + 1):
        # Predict the next day's price using Random Forest model
        predicted_price_rf = rf_model.predict(future_price_rf)[0]
        
        # Predict the next day's price using KNN model
        predicted_price_knn = knn_model.predict(future_price_knn)[0]

        # Average the predictions
        combined_prediction = (predicted_price_rf + predicted_price_knn) / 2

        # Store the day and the predicted prices in the list
        future_prices.append(
            (day, 
             f"${float(predicted_price_rf):,.2f}", 
             f"${float(predicted_price_knn):,.2f}", 
             f"${float(combined_prediction):,.2f}")
        )

        # Update the features for the next prediction based on the combined prediction
        open_price = combined_prediction  
        high_price = combined_prediction * (1 + np.random.uniform(0.01, 0.05)) 
        low_price = combined_prediction * (1 - np.random.uniform(0.01, 0.05))  
        close_price = combined_prediction  

        future_price_rf = np.array([[open_price, high_price, low_price, close_price]])
        future_price_knn = np.array([[open_price, high_price, low_price, close_price]])

        if scaler is not None:
            future_price_rf = scaler.transform(future_price_rf)
            future_price_knn = scaler.transform(future_price_knn)

    return render_template('index.html', prediction=future_prices)

if __name__ == "__main__":
    app.run(debug=True)
