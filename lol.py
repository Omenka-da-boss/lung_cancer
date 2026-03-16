import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

client = mlflow.tracking.MlflowClient()

run = client.get_run("2db5018d04244f519734b965401009d9")

print(run)