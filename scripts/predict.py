import joblib
from preprocess import load_and_preprocess_data

def predict_price(model_path, new_data):
    model = joblib.load(model_path)
    prediction = model.predict(new_data)
    return prediction

if __name__ == "__main__":
    # Load and preprocess the data
    file_path = 'D:/Crypto Currency Price Prediction/data/crypto_data.csv'
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)

    # Predict using Random Forest
    rf_prediction = predict_price('models/random_forest_model.pkl', X_test)
    print(f"Random Forest Predictions: {rf_prediction}")
    
    # Predict using KNN
    knn_prediction = predict_price('models/knn_model.pkl', X_test)
    print(f"KNN Predictions: {knn_prediction}")
