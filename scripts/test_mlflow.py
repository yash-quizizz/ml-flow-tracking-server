import mlflow

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("test_experiment")

with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.89)
    mlflow.log_artifact("test_mlflow.py")
