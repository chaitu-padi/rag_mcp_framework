import os
import uuid
import yaml

WORKFLOWS_DIR = os.path.join(os.path.dirname(__file__), "config", "workflows")

def save_workflow(config_data):
    os.makedirs(WORKFLOWS_DIR, exist_ok=True)
    workflow_id = str(uuid.uuid4())
    workflow_path = os.path.join(WORKFLOWS_DIR, f"{workflow_id}.yaml")
    with open(workflow_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    return workflow_id, workflow_path

def list_workflows():
    if not os.path.exists(WORKFLOWS_DIR):
        return []
    workflows = []
    for f in os.listdir(WORKFLOWS_DIR):
        if f.endswith(".yaml"):
            workflow_id = f[:-5]  # remove .yaml
            workflows.append({"workflow_id": workflow_id, "filename": f})
    return workflows

def load_workflow(workflow_id):
    workflow_path = os.path.join(WORKFLOWS_DIR, f"{workflow_id}.yaml")
    if not os.path.exists(workflow_path):
        return None
    with open(workflow_path) as f:
        return yaml.safe_load(f)
