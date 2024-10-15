from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
import os
import numpy as np

models_dir = os.path.join(os.getcwd(), 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Select features and target
    X = df[['Open', 'High', 'Low', 'price']].values
    y = df['Close'].values
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train Random Forest with RandomizedSearchCV to speed up training
def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    random_search_rf = RandomizedSearchCV(
        estimator=rf, param_distributions=param_dist, n_iter=5, cv=3, random_state=42, n_jobs=-1, verbose=1
    )
    random_search_rf.fit(X_train, y_train)
    
    # Save the best model
    best_rf = random_search_rf.best_estimator_
    model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    joblib.dump(best_rf, model_path)
    print(f"Best Random Forest parameters: {random_search_rf.best_params_}")
    print(f"Random Forest model saved to {model_path}")

# Train KNN with RandomizedSearchCV to speed up training
def train_knn(X_train, y_train):
    knn = KNeighborsRegressor()
    param_dist = {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  
    }
    random_search_knn = RandomizedSearchCV(
        estimator=knn, param_distributions=param_dist, n_iter=5, cv=3, random_state=42, n_jobs=-1, verbose=1
    )
    random_search_knn.fit(X_train, y_train)
    
    # Save the best model
    best_knn = random_search_knn.best_estimator_
    model_path = os.path.join(models_dir, 'knn_model.pkl')
    joblib.dump(best_knn, model_path)
    print(f"Best KNN parameters: {random_search_knn.best_params_}")
    print(f"KNN model saved to {model_path}")

if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('D:/Crypto Currency Price Prediction/data/crypto_data.csv')
    
    # Train Random Forest and KNN models
    train_random_forest(X_train, y_train)
    train_knn(X_train, y_train)
    
    # Save the scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
