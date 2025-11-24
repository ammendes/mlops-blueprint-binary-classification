import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from src.data_loader import load_data
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import yaml

def main():
    # Load config
    with open(os.path.join(os.path.dirname(__file__), '../config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    # Create a temporary directory for artifacts
    tmp_dir = os.path.join(os.path.dirname(__file__), f"../{config['artifacts']['tmp_dir']}")
    os.makedirs(tmp_dir, exist_ok=True)

    # Hyperparameters from config
    params = config['model']['params']

    # Load data
    print(f"Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Data loaded. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Set the tracking URI to a SQLite format database from config
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    # Set MLflow experiment from config
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Get artifact location from env or config, handle literal string from config.yaml
    artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION")
    if not artifact_location:
        raw_location = config['mlflow'].get('artifact_location', './mlruns')
        # If config contains a literal string with env expansion, use default
        if raw_location.startswith('${MLFLOW_ARTIFACT_LOCATION'):
            artifact_location = './mlruns'
        else:
            artifact_location = raw_location

    # Start MLflow run
    with mlflow.start_run():
        # Set artifact location as a tag (for tracking)
        mlflow.set_tag("mlflow.artifact_location", artifact_location)

        # Define parameter logging
        mlflow.log_params(params)

        # Train model
        print(f"Training model...")
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        # Predict on test data to evaluate
        print(f"Evaluating model...")
        preds = clf.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Log metrics
        print("Logging metrics...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(tmp_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # ROC Curve
        y_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        roc_path = os.path.join(tmp_dir, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # Save report as JSON before logging
        print("Generating classification report...")
        report = classification_report(y_test, preds, output_dict=True)
        report_path = os.path.join(tmp_dir, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f)
        mlflow.log_artifact(report_path, artifact_path="reports")

        # Log model with input_example for signature
        print("Logging model...")
        import numpy as np
        input_example = np.zeros((1, X_train.shape[1]))
        mlflow.sklearn.log_model(clf, name="model", input_example=input_example)

        # Clean up temporary directory
        shutil.rmtree(tmp_dir)
        
        print("Finished!")

if __name__ == "__main__":
    main()

        