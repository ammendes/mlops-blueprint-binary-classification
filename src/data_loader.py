import requests
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

def load_data(test_size=0.2, random_state=42):
    # Load config
    with open(os.path.join(os.path.dirname(__file__), '../config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    url = config['data']['dataset_url']
    file_path = config['data']['local_path']
    features = config['data']['features']
    fillna_cfg = config['data']['fillna']

    # Check if the Titanic dataset exists, otherwise download it
    if not os.path.isfile(file_path):
        print("Titanic dataset file not found. Downloading...")
        response = requests.get(url)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded dataset to {file_path}")
        else:
            print("Failed to download dataset.")
    else:
        print(f"Dataset file found at {file_path}")
        
    # Load the Titanic dataset
    df = pd.read_csv(file_path)

    # Fill missing values using config
    if fillna_cfg['Age'] == 'median':
        df['Age'] = df['Age'].fillna(df['Age'].median())
    else:
        df['Age'] = df['Age'].fillna(fillna_cfg['Age'])

    if fillna_cfg['Fare'] == 'median':
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    else:
        df['Fare'] = df['Fare'].fillna(fillna_cfg['Fare'])

    df['Embarked'] = df['Embarked'].fillna(fillna_cfg['Embarked'])

    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    X = df[features]
    y = df['Survived']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test