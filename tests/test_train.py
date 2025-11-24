import pytest
import os
import shutil
import mlflow
from src.train import main
import yaml
from unittest import mock
import time

@pytest.fixture(scope="session")
def mlflow_run_dir():
    # Run training once for all tests
    main()
    run_dir = "mlruns/1/"
    subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and d != "models"]
    assert subdirs, "No run subdirectories found."
    # Select latest run by creation time
    latest_run_id = max(subdirs, key=lambda d: os.path.getctime(os.path.join(run_dir, d)))
    latest_run_dir = os.path.join(run_dir, latest_run_id)
    yield latest_run_dir, latest_run_id
    # Cleanup after all tests
    if os.path.isdir(latest_run_dir):
        shutil.rmtree(latest_run_dir)


def test_training_runs_without_errors(mlflow_run_dir):
    latest_run_dir, _ = mlflow_run_dir
    assert os.path.exists(latest_run_dir)


def test_model_artifact_created(mlflow_run_dir):
    _, _ = mlflow_run_dir
    models_dir = "mlruns/1/models/"
    assert os.path.exists(models_dir), "Models directory does not exist."
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    assert subdirs, "No model run subdirectories found."
    latest_model_dir = os.path.join(models_dir, sorted(subdirs)[-1], "artifacts")
    expected_files = ["MLmodel", "model.pkl", "conda.yaml", "python_env.yaml", "requirements.txt"]
    for fname in expected_files:
        fpath = os.path.join(latest_model_dir, fname)
        assert os.path.exists(fpath), f"Expected model artifact {fname} not found in {latest_model_dir}."


def test_metrics_are_logged(mlflow_run_dir):
    """Test that accuracy, precision, recall, and F1 score are calculated and logged in MLflow."""
    _, latest_run_id = mlflow_run_dir
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(latest_run_id)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        assert metric in run.data.metrics, f"Metric {metric} not found in MLflow run."
        value = run.data.metrics[metric]
        assert value is not None, f"Metric {metric} value is None."


def run_training_with_csv(csv_path):
    import yaml
    # Patch config.yaml to use the synthetic CSV
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["data"]["local_path"] = csv_path
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)
    main()
    run_dir = "mlruns/1/"
    subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and d != "models"]
    assert subdirs, "No run subdirectories found."
    latest_run_id = max(subdirs, key=lambda d: os.path.getctime(os.path.join(run_dir, d)))
    latest_run_dir = os.path.join(run_dir, latest_run_id)
    return latest_run_id, latest_run_dir


def test_metrics_on_synthetic_known():
    """Test that metrics are correct for a known input/output using synthetic_known.csv."""
    # Expected values (for this synthetic dataset, you may want to adjust)
    expected_metrics = {
        "accuracy": 1.0,  # For a perfect classifier, adjust if not perfect
        # Add expected precision, recall, f1 if you know them
    }
    latest_run_id, latest_run_dir = run_training_with_csv("tests/synthetic_known.csv")
    client = mlflow.MlflowClient()
    run = client.get_run(latest_run_id)
    for metric, expected in expected_metrics.items():
        assert metric in run.data.metrics, f"Metric {metric} not found in MLflow run."
        value = run.data.metrics[metric]
        assert abs(value - expected) < 1e-6, f"Metric {metric} value {value} != expected {expected}"
    # Cleanup
    if os.path.isdir(latest_run_dir):
        shutil.rmtree(latest_run_dir)


def test_plots_are_generated(mlflow_run_dir):
    """Test that ROC curve and confusion matrix plots are generated and saved to the expected location."""
    latest_run_dir, _ = mlflow_run_dir
    plots_dir = os.path.join(latest_run_dir, "artifacts", "plots")
    expected_plots = ["roc_curve.png", "confusion_matrix.png"]
    for plot in expected_plots:
        plot_path = os.path.join(plots_dir, plot)
        assert os.path.exists(plot_path), f"Plot {plot} not found in {plots_dir}."


def test_mlflow_logging_functions_called(mlflow_run_dir):
    """Test that MLflow logging functions are called during training."""
    with mock.patch("mlflow.log_metric") as mock_log_metric, \
        mock.patch("mlflow.log_artifact") as mock_log_artifact, \
        mock.patch("mlflow.log_params") as mock_log_params:
        main()
        # Check that log_metric was called for each metric
        for metric in ["accuracy", "precision", "recall", "f1"]:
            assert any(metric in str(call) for call in mock_log_metric.call_args_list), f"mlflow.log_metric not called for {metric}"
        # Check that log_artifact was called at least once
        assert mock_log_artifact.call_count > 0, "mlflow.log_artifact was not called"
        # Check that log_params was called
        assert mock_log_params.call_count > 0, "mlflow.log_params was not called"


def test_model_and_metrics_artifacts_created(mlflow_run_dir):
    """Test that model and metrics artifacts are created in the MLflow run and model registry directories."""
    latest_run_dir, latest_run_id = mlflow_run_dir
    mlruns_dir = os.path.join(os.getcwd(), "mlruns", "1")
    # Check for metrics artifacts (plots)
    model_files = ["MLmodel", "model.pkl", "conda.yaml", "python_env.yaml", "requirements.txt"]
    model_registry_dir = os.path.join(mlruns_dir, "models")
    model_subdirs = [d for d in os.listdir(model_registry_dir) if os.path.isdir(os.path.join(model_registry_dir, d))]
    latest_model = sorted(model_subdirs)[-1]
    model_artifacts_dir = os.path.join(model_registry_dir, latest_model, "artifacts")
    for fname in model_files:
        fpath = os.path.join(model_artifacts_dir, fname)
        assert os.path.exists(fpath), f"Model artifact {fname} not found in {model_artifacts_dir}."
    plots_dir = os.path.join(latest_run_dir, "artifacts", "plots")
    expected_plots = ["roc_curve.png", "confusion_matrix.png"]
    for plot in expected_plots:
        plot_path = os.path.join(plots_dir, plot)
        assert os.path.exists(plot_path), f"Plot {plot} not found in {plots_dir}."
    reports_dir = os.path.join(latest_run_dir, "artifacts", "reports")
    report_file = os.path.join(reports_dir, "classification_report.json")
    assert os.path.exists(report_file), f"Classification report not found in {reports_dir}."


def run_training_with_csv_expect_error(csv_path, expected_exception):
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["data"]["local_path"] = csv_path
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)
    with pytest.raises(expected_exception):
        main()


def test_full_training_pipeline_integration(mlflow_run_dir):
    """Integration test: run full training pipeline and verify all outputs."""
    latest_run_dir, latest_run_id = mlflow_run_dir
    # Check model artifacts in registry
    models_dir = "mlruns/1/models/"
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    assert subdirs, "No model run subdirectories found."
    latest_model_dir = os.path.join(models_dir, sorted(subdirs)[-1], "artifacts")
    model_files = ["MLmodel", "model.pkl", "conda.yaml", "python_env.yaml", "requirements.txt"]
    for fname in model_files:
        fpath = os.path.join(latest_model_dir, fname)
        assert os.path.exists(fpath), f"Model artifact {fname} not found in {latest_model_dir}."
    # Check metrics in MLflow
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(latest_run_id)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        assert metric in run.data.metrics, f"Metric {metric} not found in MLflow run."
    # Check plots
    plots_dir = os.path.join(latest_run_dir, "artifacts", "plots")
    expected_plots = ["roc_curve.png", "confusion_matrix.png"]
    for plot in expected_plots:
        plot_path = os.path.join(plots_dir, plot)
        assert os.path.exists(plot_path), f"Plot {plot} not found in {plots_dir}."
    # Check classification report
    reports_dir = os.path.join(latest_run_dir, "artifacts", "reports")
    report_file = os.path.join(reports_dir, "classification_report.json")
    assert os.path.exists(report_file), f"Classification report not found in {reports_dir}."
