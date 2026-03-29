import os
from evidently.ui.workspace import CloudWorkspace
from evidently.errors import EvidentlyError

# Load your environment variables
from dotenv import load_dotenv
load_dotenv()

ws_url = os.getenv("EVIDENTLY_WORKSPACE_URL", "https://app.evidently.cloud")
ws_key = os.getenv("EVIDENTLY_API_KEY")

# Connect to Evidently Cloud workspace
ws = CloudWorkspace(
    token=ws_key,
    url=ws_url
)

project_name = "lung_cancer_pred"

orgs = os.getenv("ORGS_ID")

try:
    # Attempt to list projects
    projects = ws.list_projects()
    print("List of projects fetched successfully.")
except EvidentlyError as e:
    print("Error listing projects:", e)
    projects = []

# Try to find the project by name
project = next((p for p in projects if p.name == project_name), None)

if project:
    print(f"🟢 Project already exists: {project.name} (ID: {project.id})")
else:
    try:
        # Create the new project
        project = ws.create_project(name=project_name,org_id=orgs)
        # Save to actually persist the project
        project.description = "A Lung Cancer Prediction Project With ML Pipeline"
        project.save()
        print(f"✅ Project '{project_name}' created successfully with ID: {project.id}")
    except EvidentlyError as e:
        print("❌ Failed to create project:", e)