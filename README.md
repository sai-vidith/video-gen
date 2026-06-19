<p align="center">
  <h1 align="center">🎬 Video-DAG-Gen</h1>
  <p align="center">
    <strong>LLM-Orchestrated Face-Consistent Video Generation Engine</strong>
  </p>
  <p align="center">
    Transform story prompts into cinematic short-form video reels with AI-driven storyboarding, face-locked keyframes, neural voiceover, and automated scene compilation.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-Llama_3.3-F55036?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LTX--Video-I2V-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Edge--TTS-Neural-00A4EF?style=for-the-badge" />
</p>

---

## 🧠 What This Does

**Video-DAG-Gen** is a multi-agent AI pipeline that takes a simple text story and produces a complete 60-second vertical video reel — with face-consistent characters, cinematic visual design, and synchronized neural voiceover.

### The Pipeline

```
"Bob's restaurant was failing because his dishwasher was throwing steaks in the dumpster"

    ↓  Groq Llama 3.3 (Storyboard)

10 cinematic scene nodes with visual prompts, lighting, camera angles, color grading, mood

    ↓  3-Stage LLM-Guided Image Selector

For each scene: DDG multi-query search → automated filtering → Groq Vision scoring (1-10)

    ↓  Kaggle T4 GPU Worker

Best reference image → LTX-Video Image-to-Video animation (or SDXL face-lock → LTX-Video I2V)

    ↓  Edge-TTS + FFmpeg

Neural voiceover synthesis → Scene-audio alignment → 9:16 portrait reel compilation

    ↓  Output

📥 60-second downloadable .mp4 with voiceover
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                       │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Face Profile │  │ Story Input  │  │ Live Preview Grid │  │
│  │ Manager      │  │ + DAG View   │  │ + Download Button │  │
│  └─────────────┘  └──────┬───────┘  └───────────────────┘  │
└──────────────────────────┼──────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   CONTEXT ENGINE        │
              │   Groq Llama-3.3-70B    │
              │   Structured JSON output│
              │   10-scene storyboard   │
              └────────────┬────────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │    3-STAGE IMAGE SELECTOR          │
         │                                     │
         │  Stage 1: DDG Multi-Query (9 imgs) │
         │  Stage 2: Auto-Filter (size/ratio) │
         │  Stage 3: Groq Vision Score (1-10) │
         │                                     │
         │  score ≥ 6 → Reference Image Path  │
         │  score < 6 → SDXL Fallback Path    │
         └───────┬──────────────┬─────────────┘
                 │              │
    ┌────────────▼──┐   ┌──────▼──────────────┐
    │ REFERENCE PATH│   │ FACE-LOCK PATH      │
    │ LTX-Video I2V │   │ SDXL+IP-Adapter     │
    │ (12GB VRAM)   │   │ → del → LTX-Video   │
    └────────────┬──┘   └──────┬──────────────┘
                 │              │
         ┌───────▼──────────────▼───────┐
         │     POST-PROCESSING          │
         │  Edge-TTS voiceover synth    │
         │  FFmpeg duration alignment   │
         │  FFmpeg concat → 9:16 reel   │
         └──────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Implementation |
|---|---|
| **LLM Storyboarding** | Groq `llama-3.3-70b-versatile` with structured JSON output — parses any story into 10 cinematic scenes with lighting, lens, camera angle, mood, and color grading |
| **3-Stage Visual Retrieval** | Multi-query DDG search → resolution/aspect/color filtering → Groq Vision (`llama-3.2-90b-vision-preview`) scoring — ensures only high-quality reference images reach the video model |
| **Face Consistency** | SDXL + IP-Adapter Plus Face (ViT-H) for character scenes — maintains face identity across multiple scenes from a single photo upload |
| **Video Animation** | LTX-Video Image-to-Video via HuggingFace `diffusers` — animates keyframes into high-fidelity video clips directly on GPU |
| **Neural Voiceover** | Edge-TTS with 6 voice profiles (male/female narrator, child, deep voice, British male/female) — infinite free high-quality speech synthesis |
| **Smart VRAM Management** | Sequential model loading with full cleanup (del + gc + CUDA cache) — runs SDXL (7GB) and LTX-Video (12GB) on a single 16GB T4 without conflicts |
| **Portrait Video Output** | Automated center-crop + scale to 720×1280 (9:16) — optimized for Instagram Reels / TikTok / YouTube Shorts |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg installed and on PATH
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Kaggle/Colab GPU runtime (T4 or better) for video generation

### Local Setup (Dashboard)

```bash
# Clone
git clone https://github.com/yourusername/Video-DAG-Gen.git
cd Video-DAG-Gen

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your GROQ_API_KEY and COLAB_GPU_URL

# Run
streamlit run app.py
```

### GPU Worker Setup (Kaggle)

1. Open `colab_gpu_worker.ipynb` in Kaggle with **GPU T4 ×2** accelerator
2. Set your `NGROK_TOKEN` in the first cell
3. Run all cells — the worker will print an ngrok URL
4. Copy the URL to your local `.env` as `COLAB_GPU_URL`

---

## 📁 Project Structure

```
Video-DAG-Gen/
├── app.py                    # Streamlit dashboard (UI + orchestration)
├── context_engine.py         # LLM storyboard generation (Groq/Gemini/Llama)
├── image_selector.py         # 3-stage reference image selection pipeline
├── video_pipeline.py         # TTS, video dispatch, FFmpeg compilation
├── colab_gpu_worker.py       # Kaggle GPU worker (SDXL + LTX-Video diffusers)
├── colab_gpu_worker.ipynb    # Kaggle notebook for GPU deployment
├── requirements.txt          # Python dependencies
├── Agents.md                 # System configuration & data schemas
├── .env                      # API keys (gitignored)
├── profiles/                 # Saved face profile crops (gitignored)
└── client/                   # Reference materials
```

---

## 🔧 Configuration

### Environment Variables (`.env`)

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GEMINI_API_KEY=your_gemini_key_here          # Optional fallback
COLAB_GPU_URL=https://your-ngrok-url.ngrok-free.dev
```

### Storyboard Schema (Pydantic)

```python
class StorySceneNode(BaseModel):
    id: str                      # "scene_1" ... "scene_10"
    scene_type: str              # hook | build-up | climax | resolution | payoff
    visual_prompt: str           # Cinematic scene description
    lighting: str                # Volumetric god rays, neon rim lighting, etc.
    camera_angle: str            # Low-angle close-up, aerial establishing, etc.
    camera_movement: str         # Slow dolly, static, zoom-in, pan-left
    lens: str                    # 35mm cinematic, 85mm portrait, 24mm wide
    color_palette: str           # Teal-orange, monochrome, sepia, neon pink
    mood: str                    # Suspenseful, melancholic, triumphant
    voiceover_text: str          # Clean narration (max 15 words)
    needs_face: bool             # Whether this scene shows the character
    search_queries: List[str]    # 3 diverse DDG image search queries
```

---

## 🧪 Technical Details

### VRAM Management Strategy

The system runs two large models on a single 16GB T4 GPU by **never loading both simultaneously**:

```
Timeline:
t0  ──── SDXL loaded (7GB) ──── keyframe generated ──── SDXL deleted
t1  ──── gc.collect() + torch.cuda.empty_cache() ────
t2  ──── LTX-Video loaded (12GB) ──── video generated ──── LTX-Video deleted
t3  ──── gc.collect() ──── ready for next scene
```

### Image Selection Quality Gate

The Groq Vision model evaluates each candidate with this scoring rubric:
- **9-10**: Perfect match — composition, lighting, mood all align
- **7-8**: Strong match — correct subject/setting, minor style differences
- **5-6**: Acceptable — right category but noticeable mismatches
- **3-4**: Poor — wrong setting, lighting, or mood
- **1-2**: Irrelevant — completely different subject matter

Only images scoring **≥ 6** are used. Below that threshold, the system falls back to SDXL keyframe generation.


---

## 🙏 Acknowledgments

- [LTX-Video](https://github.com/Lightricks/LTX-Video) — Open-source real-time video generation model
- [Groq](https://groq.com) — Ultra-fast LLM inference
- [Edge-TTS](https://github.com/rany2k/edge-tts) — Microsoft Edge neural text-to-speech
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) — Face-consistent image generation
- [Streamlit](https://streamlit.io) — Python web application framework
