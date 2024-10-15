from sklearn.metrics import mean_squared_error
import joblib
from train import load_and_preprocess_data
import numpy as np

def evaluate_model(model_path, X_test, y_test):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    return mse, rmse

if __name__ == "__main__":
    # Load test data
    _, X_test, _, y_test, _ = load_and_preprocess_data('D:/Crypto Currency Price Prediction/data/crypto_data.csv')
    
    # Evaluate Random Forest
    print("Evaluating Random Forest:")
    evaluate_model('../models/random_forest_model.pkl', X_test, y_test)
    
    # Evaluate KNN
    print("Evaluating KNN:")
    evaluate_model('../models/knn_model.pkl', X_test, y_test)
