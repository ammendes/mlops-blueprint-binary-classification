# Build the Docker image for training and inference
build:
	docker build -f docker/Dockerfile -t titanic-inference .

# Train the model in Docker
# - Mounts mlruns, mlflow.db, and data to /app in the container
# - MLflow artifacts will be stored in /app/mlruns (mapped to ./mlruns on host)
# - If you change artifact location in config.yaml, update this mount accordingly
train:
	docker run -it --rm \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/mlflow.db:/app/mlflow.db \
		-v $(PWD)/data:/app/data \
		titanic-inference python -m src.train

# Run MLflow UI in Docker
# - Mounts mlruns and mlflow.db for experiment tracking
mlflow-ui:
	docker run -p 5001:5001 \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/mlflow.db:/app/mlflow.db \
		titanic-inference mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001 --host 0.0.0.0

# Run the API in Docker
# - Mounts mlruns and mlflow.db for model serving
run-api:
	docker run -p 8000:8000 \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/mlflow.db:/app/mlflow.db \
		titanic-inference

# Run tests locally
# - Uses PYTHONPATH=. so src imports work
# - MLflow artifact location is handled in train.py (defaults to ./mlruns for local)
test:
	PYTHONPATH=. pytest