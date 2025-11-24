import requests
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

def load_data(csv_path=None, test_size=0.2, random_state=42, return_df=False):
    # Load config
    with open(os.path.join(os.path.dirname(__file__), '../config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    url = config['data']['dataset_url']
    file_path = config['data']['local_path'] if csv_path is None else csv_path
    features = config['data']['features']
    fillna_cfg = config['data']['fillna']

    # Check if the Titanic dataset exists, otherwise download it
    if not os.path.isfile(file_path) and csv_path is None:
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
        

    # Try to load the Titanic dataset, handle encoding/corruption errors
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"File '{file_path}' is empty.")
    except UnicodeDecodeError:
        raise ValueError(f"File '{file_path}' has encoding issues.")
    except Exception as e:
        raise ValueError(f"Could not load file '{file_path}': {e}")

    # Handle empty DataFrame
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"File '{file_path}' is empty or has no columns.")

    # Check for required columns
    required = ['Survived'] + features
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Handle files with only one row or one column
    if df.shape[0] == 1:
        print(f"Warning: File '{file_path}' has only one row.")
    if df.shape[1] == 1:
        raise ValueError(f"File '{file_path}' has only one column.")


    # Data cleaning: trim whitespace and normalize string columns
    string_cols = ['Name', 'Embarked']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].str.upper()

    # Remove or correct invalid entries (e.g., negative ages, impossible fares)
    if 'Age' in df.columns:
        df = df[df['Age'].isnull() | (df['Age'] >= 0)]
    if 'Fare' in df.columns:
        df = df[df['Fare'].isnull() | (df['Fare'] >= 0)]

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

    # Handle extra columns by selecting only expected features
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    # Ensure all expected dummy columns are present
    for col in features:
        if col not in df.columns:
            df[col] = 0

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

    if return_df:
        return X_train, X_test, y_train, y_test, df
    else:
        return X_train, X_test, y_train, y_test