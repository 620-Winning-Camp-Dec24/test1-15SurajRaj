import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Mock data processing script
def preprocess_data():
    print("Starting data preprocessing...")
    # Replace with your dataset path
    data_path = "../data/transaction_data.csv"
    try:
        df = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
        
        # Assume 'is_fraud' is the target column
        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        print("Data preprocessing complete.")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("Dataset not found. Please check the file path.")

if __name__ == "__main__":
    preprocess_data()
