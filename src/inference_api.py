from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import numpy as np
import yaml
import os

class TitanicInput(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex_male: int
    Embarked_Q: int
    Embarked_S: int

 # Load config
with open(os.path.join(os.path.dirname(__file__), '../config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI server
app = FastAPI()

# Load model from MLflow using config

# Set MLflow tracking URI to use relative path for SQLite DB
tracking_uri = config['mlflow']['tracking_uri']
if tracking_uri.startswith("sqlite:///"):
    # Ensure path is relative for Docker
    db_path = tracking_uri.replace("sqlite:///", "")
    tracking_uri = f"sqlite:///./{os.path.basename(db_path)}"
mlflow.set_tracking_uri(tracking_uri)

# Load model from MLflow Model Registry
model_uri = f"models:/{config['mlflow']['model_name']}/{config['mlflow']['registry_stage']}"
model = mlflow.sklearn.load_model(model_uri)

@app.post("/predict")
def predict(input: TitanicInput):
    data = np.array([[input.Pclass, input.Age, input.SibSp, input.Parch, input.Fare, input.Sex_male, input.Embarked_Q, input.Embarked_S]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}