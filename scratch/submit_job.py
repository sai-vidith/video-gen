import os
import requests
import json

# Setup parameters
BACKEND_URL = "https://cluster.vjstartup.com/be/api/v1/jobs/submit"
SCRIPT_PATH = r"c:\Users\Sai Divya\Desktop\Video-Dag_GEn\cluster_job\train.py"
REQ_PATH = r"c:\Users\Sai Divya\Desktop\Video-Dag_GEn\cluster_job\requirements.txt"
CONFIG_PATH = r"c:\Users\Sai Divya\Desktop\Video-Dag_GEn\cluster_job\config.yaml"
DATASET_PATH = r"c:\Users\Sai Divya\Desktop\cluster\cluster-compute\examples\DatSet.zip"

print("Submitting training job to the cluster backend...")

# Metadata fields
data = {
    "name": "Fine-tuning SDXL",
    "description": "LoRA fine-tuning of SDXL on a custom character/style dataset using Multi-GPU DDP.",
    "project": "Video-DAG-Gen",
    "queue_id": "bronze",  # Use bronze (A2000) or silver (Ada 2000) depending on queue config
    "requested_gpus": 1,
    "requested_cpus": 4,
    "requested_ram_gb": 16,
    "framework": "pytorch"
}

files = {}
try:
    files["script"] = ("train.py", open(SCRIPT_PATH, "rb"), "text/x-python")
    if os.path.exists(REQ_PATH):
        files["requirements"] = ("requirements.txt", open(REQ_PATH, "rb"), "text/plain")
    if os.path.exists(CONFIG_PATH):
        files["config"] = ("config.yaml", open(CONFIG_PATH, "rb"), "text/yaml")
    if os.path.exists(DATASET_PATH):
        files["dataset"] = ("DatSet.zip", open(DATASET_PATH, "rb"), "application/zip")

    # In a real environment, request needs authorization header if JWT is enabled.
    # If the backend is running in dev mode or doesn't require active JWT verification on submit:
    # Here we submit the request
    response = requests.post(BACKEND_URL, data=data, files=files, verify=False)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error submitting job: {e}")
finally:
    for f in files.values():
        f[1].close()
