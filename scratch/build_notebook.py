import json

# Read the Python code from colab_gpu_worker.py
with open("colab_gpu_worker.py", "r", encoding="utf-8") as f:
    fastapi_code = f.readlines()

# Build the cells for LTX-Video and SDXL
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Consistent Face Video DAG Engine - Kaggle/Colab GPU Worker Node\n",
            "Select T4 GPU (or better) under Accelerator in Notebook settings before running."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Install dependencies\n",
            "print(\"Installing dependencies (FastAPI, Pyngrok, Diffusers, Accelerate, edge-tts, etc.)...\")\n",
            "!pip install -q fastapi uvicorn pyngrok diffusers transformers accelerate safetensors pillow numpy duckduckgo-search requests edge-tts pillow-heif opencv-python-headless\n",
            "print(\"✅ Environment Setup Complete!\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": fastapi_code
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("colab_gpu_worker.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("[SUCCESS] Successfully generated colab_gpu_worker.ipynb!")
