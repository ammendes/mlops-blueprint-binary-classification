
# MLOps Project: Titanic Survival Prediction

This repository implements a production-ready MLOps workflow for binary classification using the Titanic dataset. It covers data ingestion, preprocessing, model training, evaluation, experiment tracking, and artifact management with MLflow. The structure and practices follow current industry standards for collaborative, reproducible machine learning projects.

## Project Structure

```
mlops-project-1/
├── config.yaml
├── requirements.txt
├── README.md
├── mlflow.db                # MLflow tracking database (SQLite)
├── mlruns/                  # MLflow artifact storage (auto-managed)
├── tmp/                     # Temporary directory for local artifacts (auto-cleaned)
├── data/
│   └── titanic.csv          # Dataset (auto-downloaded)
├── docker/                  # Docker configs
├── notebooks/               # Jupyter notebooks
├── src/
│   ├── data_loader.py       # Data loading and preprocessing
│   └── train.py             # Model training, evaluation, and MLflow logging
│   └── inference_api.py     # FastAPI inference service

```

## Features & Best Practices

- **Automated data download and preprocessing**
- **Model training and evaluation with scikit-learn**
- **Experiment tracking and artifact management with MLflow (database backend)**
- **Metrics and plots logged as MLflow artifacts (confusion matrix, ROC curve, classification report)**
- **Temporary files auto-cleaned after each run**
- **Requirements managed in `requirements.txt`**
- **Project root kept clean; artifacts organized in MLflow UI**

## Setup

1. **Clone the repository:**
	```sh
	git clone <your-repo-url>
	cd mlops-project-1
	```

2. **Python version:**
	- Recommended: Python 3.10 or newer

3. **Create and activate a Python virtual environment (recommended):**
	```sh
	python -m venv venv
	source venv/bin/activate
	```

4. **Install dependencies:**
	```sh
	pip install -r requirements.txt
	```

## Dockerized Workflow (Recommended)
All steps below are performed from the project root. This workflow ensures reproducibility and clean separation of environments.

### 1. Build the Docker image:
```sh
docker build -f docker/Dockerfile -t titanic-inference .
```

### 2. Train the model inside Docker:
```sh
docker run -it --rm \
  -v $PWD/mlruns:/app/mlruns \
  -v $PWD/mlflow.db:/app/mlflow.db \
  -v $PWD/data:/app/data \
  titanic-inference python src/train.py
```

This will create all necessary experiemtn and model artifact in mounted volumes.

### 3. Start the MLflow UI inside Docker:
```sh
docker run -p 5001:5001 \
  -v $PWD/mlruns:/app/mlruns \
  -v $PWD/mlflow.db:/app/mlflow.db \
  titanic-inference mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001 --host 0.0.0.0
```

- Access at http://localhost:5001
- Regiter the trained model in the Model Registry from a run created inside the container

### 4. Run the FastAPI inference service inside Docker:
```sh
docker run -p 8000:8000 \
  -v $PWD/mlruns:/app/mlruns \
  -v $PWD/mlflow.db:/app/mlflow.db \
  titanic-inference
```

- The API will be available at http://localhost:8000
- Interactive docs: http://localhost:8000/docs

## Usage

### 1. Model Training and Logging
```sh
python src/train.py
```
- Downloads the Titanic dataset (if not present), preprocesses data, trains a RandomForest model, evaluates metrics, logs results and artifacts to MLflow, and cleans up temporary files.

### 2. MLflow UI for Experiment Tracking
```sh
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
- Access the UI at [http://localhost:5001](http://localhost:5001)

### 3. FastAPI Inference Service

#### Start the API server:
```sh
uvicorn src.inference_api:app --reload
```
- The API will be available at [http://localhost:8000](http://localhost:8000)
- Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

#### /predict Endpoint
- Accepts a JSON payload with the following fields:
	- `Pclass` (int)
	- `Age` (float)
	- `SibSp` (int)
	- `Parch` (int)
	- `Fare` (float)
	- `Sex_male` (int)
	- `Embarked_Q` (int)
	- `Embarked_S` (int)
- Example request:
```json
{
	"Pclass": 3,
	"Age": 22.0,
	"SibSp": 1,
	"Parch": 0,
	"Fare": 7.25,
	"Sex_male": 1,
	"Embarked_Q": 0,
	"Embarked_S": 1
}
```

#### Testing the API
- Use the Swagger UI at `/docs` to test requests interactively.
- Or use `curl`:
	```sh
	curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"Pclass":3,"Age":22.0,"SibSp":1,"Parch":0,"Fare":7.25,"Sex_male":1,"Embarked_Q":0,"Embarked_S":1}'
	```

#### Model Loading
- The API loads the latest model from the MLflow Model Registry (ensure the model is registered and in the correct stage, e.g., `Production`).
- Input validation is handled by Pydantic for robust and secure inference.

## MLflow Tracking & Artifacts

- **Tracking URI:** Uses SQLite database (`mlflow.db`) for robust experiment tracking.
- **Artifacts:** Metrics, plots, and reports are logged to organized subdirectories in MLflow (`plots/`, `reports/`).
- **Model:** Saved in MLflow’s MLmodel format for easy deployment or reproducibility.

## Cleaning Up

- Temporary files are saved in `tmp/` during each run and automatically deleted after logging to MLflow.
- The `mlruns/` folder is managed by MLflow and should be added to `.gitignore`.
