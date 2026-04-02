# import os
# from evidently.ui.workspace import CloudWorkspace
# from evidently.errors import EvidentlyError

# # Load your environment variables
# from dotenv import load_dotenv
# load_dotenv()

# ws_url = os.getenv("EVIDENTLY_WORKSPACE_URL", "https://app.evidently.cloud")
# ws_key = os.getenv("EVIDENTLY_API_KEY")

# # Connect to Evidently Cloud workspace
# ws = CloudWorkspace(
#     token=ws_key,
#     url=ws_url
# )

# project_name = "lung_cancer_pred"

# orgs = os.getenv("ORGS_ID")

# try:
#     # Attempt to list projects
#     projects = ws.list_projects()
#     print("List of projects fetched successfully.")
# except EvidentlyError as e:
#     print("Error listing projects:", e)
#     projects = []

# # Try to find the project by name
# project = next((p for p in projects if p.name == project_name), None)

# if project:
#     print(f"🟢 Project already exists: {project.name} (ID: {project.id})")
# else:
#     try:
#         # Create the new project
#         project = ws.create_project(name=project_name,org_id=orgs)
#         # Save to actually persist the project
#         project.description = "A Lung Cancer Prediction Project With ML Pipeline"
#         project.save()
#         print(f"✅ Project '{project_name}' created successfully with ID: {project.id}")
#     except EvidentlyError as e:
#         print("❌ Failed to create project:", e)

import mlflow
from mlflow.tracking import MlflowClient

# Set your tracking URI (adjust if needed)
mlflow.set_tracking_uri("http://localhost:5000")  # or your MLflow server URL
client = MlflowClient()

experiment_name = "Nested Runs"
exp = client.get_experiment_by_name(experiment_name)
if exp is None:
    print(f"Experiment '{experiment_name}' not found!")
    exit()

experiment_id = exp.experiment_id

# First, get the top runs (nested or not) to see what's there
runs = client.search_runs(
    experiment_id,
    order_by=["metrics.roc_auc DESC"],   # you can change metric name
    max_results=5
)

print(f"Found {len(runs)} runs:")
for run in runs:
    run_id = run.info.run_id
    metrics = run.data.metrics
    print(f"\nRun ID: {run_id}")
    print(f"  Metrics: {metrics}")
    # List all top-level artifacts
    artifacts = client.list_artifacts(run_id)
    print(f"  Top-level artifacts: {[a.path for a in artifacts]}")
    # If you want to go deeper, you can list subdirectories
    for art in artifacts:
        if art.is_dir:
            sub_artifacts = client.list_artifacts(run_id, path=art.path)
            print(f"    Contents of '{art.path}': {[a.path for a in sub_artifacts]}")
            
all_runs = client.search_runs(experiment_id, max_results=100)
for run in all_runs:
    artifacts = client.list_artifacts(run.info.run_id)
    if artifacts:
        print(f"Run {run.info.run_id}: artifacts = {[a.path for a in artifacts]}")
    else:
        print("Not found")