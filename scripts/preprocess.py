import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to load and preprocess data for future price prediction
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df['Target'] = df['price'].shift(-1)
    df.dropna(inplace=True)

    # Select features and target for prediction
    X = df[['Open', 'High', 'Low', 'price']].values  
    y = df['Target'].values  

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    file_path = 'D:/Crypto Currency Price Prediction/data/crypto_data.csv'
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)
    print("Data Preprocessed")
