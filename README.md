
# Production-Ready MLOps blueprint: Titanic Survival Prediction

![Python 3](https://img.shields.io/badge/python-3-brightgreen.svg)
[![GitHub license](https://img.shields.io/github/license/ammendes/mlops-blueprint-binary-classification)](https://github.com/ammendes/mlops-blueprint-binary-classification/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/ammendes/mlops-blueprint-binary-classification)](https://github.com/ammendes/mlops-blueprint-binary-classification/stargazers)
![GitHub forks](https://img.shields.io/github/forks/ammendes/mlops-blueprint-binary-classification?style=social)
![Coverage](https://img.shields.io/codecov/c/github/ammendes/mlops-blueprint-binary-classification/main?label=coverage)

Welcome to a robust, production-ready MLOps template for binary classification, built around the Titanic dataset. This project is designed for data scientists, ML engineers, and teams seeking a practical, end-to-end workflow for collaborative, reproducible machine learning. It demonstrates best practices in data ingestion, preprocessing, model training, evaluation, experiment tracking, and deployment using MLflow, Docker, and FastAPI.

## Who Is This For?
- **Data scientists and ML engineers** looking for a proven template to accelerate real-world projects.
- **Teams and organizations** needing reproducible, collaborative ML workflows with robust experiment tracking and artifact management.
- **Hiring managers and technical leads** evaluating candidates with hands-on expertise in modern MLOps tools and practices.

By using this project, you benefit from a clean architecture, automated workflows, and deployment-ready code that can be adapted to your own use cases or serve as a foundation for more complex solutions.

<p align="center">
  <img src="docs/architecture_diagram.png" alt="Project Architecture" />
</p>

## Project Structure

```
mlops-blueprint-binary-classification/
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

- **Automated data download and preprocessing** for seamless onboarding and reproducibility.
- **Configurable pipeline** using YAML for flexible experimentation and easy adaptation to new datasets or requirements.
- **Model training and evaluation with scikit-learn**, including robust metrics and visualizations.
- **Experiment tracking and artifact management with MLflow (database backend)** for versioning, reproducibility, and team collaboration.
- **Metrics, plots, and reports logged as MLflow artifacts** (confusion matrix, ROC curve, classification report) for transparent model evaluation.
- **Model registry and deployment-ready packaging** using MLflow and FastAPI, enabling rapid transition from research to production.
- **REST API for real-time inference** with input validation and documentation (Swagger/OpenAPI).
- **Dockerized workflow** for environment consistency, easy sharing, and scalable deployment.
- **Makefile shortcuts** to automate common tasks and enforce best practices.
- **Temporary files auto-cleaned after each run** to keep the workspace organized.
- **Requirements managed in `requirements.txt`** for reliable dependency management.
- **Clean project structure** with artifacts organized in MLflow UI and version control best practices.

## Setup

1. **Clone the repository:**
	```sh
	git clone <your-repo-url>
    cd mlops-blueprint-binary-classification
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

1. **Build the Docker image:**
    ```sh
    docker build -f docker/Dockerfile -t titanic-inference .
    ```

2. **Train the model inside Docker:**
    ```sh
    docker run -it --rm \
    -v $PWD/mlruns:/app/mlruns \
    -v $PWD/mlflow.db:/app/mlflow.db \
    -v $PWD/data:/app/data \
    titanic-inference python src/train.py
    ```

This will create all necessary experiemtn and model artifact in mounted volumes.

3. **Start the MLflow UI inside Docker:**
    ```sh
    docker run -p 5001:5001 \
    -v $PWD/mlruns:/app/mlruns \
    -v $PWD/mlflow.db:/app/mlflow.db \
    titanic-inference mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001 --host 0.0.0.0
    ```

- Access at http://localhost:5001
- Register the trained model in the Model Registry from a run created inside the container.

4. **Run the FastAPI inference service inside Docker:**
    ```sh
    docker run -p 8000:8000 \
    -v $PWD/mlruns:/app/mlruns \
    -v $PWD/mlflow.db:/app/mlflow.db \
    titanic-inference
    ```

- The API will be available at http://localhost:8000
- Interactive docs: http://localhost:8000/docs

## Makefile Shortcuts
For convenience, you can use the provided Makefile to run common project tasks with simple commands:

- **Build the Docker image:**
    ```sh
    make build
    ```

- **Train the model in Docker:**
    ```sh
    make train
    ````

- **Start the MLflow UI in Docker:**
    ```sh
    make mlflow-ui
    ````

- **Run the FastAPI inference service in Docker:**
    ```sh
    make run-api
    ````

These commands automate the recommended workflow and ensure reproducibility.

Make sure you are in the project root directory before running any `make`command.

## Usage

1. **Model Training and Logging**
    ```sh
    python src/train.py
    ```
- Downloads the Titanic dataset (if not present), preprocesses data, trains a RandomForest model, evaluates metrics, logs results and artifacts to MLflow, and cleans up temporary files.

2. **MLflow UI for Experiment Tracking**
    ```sh
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
    ```
- Access the UI at [http://localhost:5001](http://localhost:5001)

3. **FastAPI Inference Service**

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

## Testing

Unit and integration tests are provided to ensure reliability of data loading, preprocessing, and model training components.

- **Run all tests:**
    ```sh
    PYTHONPATH=. pytest
    ```

**Setup:**  
Make sure you have installed all dependencies and activated your Python environment before running tests.

**Notes:**  
Some tests (e.g., data download) may require internet access and will temporarily rename the dataset file to verify download logic. All files are restored after tests complete.

Test results will be shown in the terminal. All tests should pass for a healthy codebase.


---
## License

This project is licensed under the [MIT License](LICENSE).
