# System Configuration: Consistent Face-Locked Video DAG Engine

## 🤖 System Persona & Guardrails
- **Role**: Senior AI Systems Engineer & Python Full-Stack Architect.
- **Workflow**: Create a comprehensive implementation plan as a verifiable Artifact before writing or altering files. Write production-ready, clean Python code without dummy placeholders or un-implemented functions.
- **Testing**: Ensure the code runs smoothly without environment port crashes by deploying a completely self-contained Streamlit dashboard.

---

## 💻 Technical Stack Specs
- **Frontend UI & State**: Streamlit (Python 3.11+) for single-entrypoint reactive UI, state management, and media rendering.
- **Core Orchestrator**: Async Python loops handling parallel execution pipelines natively without external Celery or Redis requirements.
- **LLM Context Engine**: Official Google GenAI SDK (`google-genai`) running `gemini-2.5-pro` with structured JSON output configurations.
- **Media Processing Layer**: `pillow-heif` for processing client HEIC media uploads, `edge-tts` for infinite free high-quality voiceover generation, and an automated system-level `ffmpeg` compilation pipeline to create downloadable portrait videos.

---

## 📐 High-Fidelity Data Schemas & Constraints

### 1. Unified Storyboard Scene Graph Schema (Pydantic)
Every raw user input prompt must be parsed and expanded into a deep 6-7 scene narrative graph using a strict schema layout:

```python
from pydantic import BaseModel
from typing import List

class StorySceneNode(BaseModel):
    id: str                  # e.g., "scene_1", "scene_2"
    scene_type: str          # "hook" | "build-up" | "climax" | "resolution" | "payoff"
    visual_prompt: str       # High-detail cinematic illustration description for diffusion models
    voiceover_text: str      # Clean narrative string (max 15 words) for audio generation

class VideoDAGPayload(BaseModel):
    global_style: str        # e.g., "90s cinematic film noir, volumetric lighting, high photorealism"
    scenes: List[StorySceneNode]