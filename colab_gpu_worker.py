"""
Kaggle/Colab GPU Worker — LTX-Video + SDXL Face-Lock Pipeline

Architecture:
- Single-process FastAPI server (no ComfyUI dependency)
- Two generation paths:
  Path A (Reference Image): Downloads web-found image → LTX-Video I2V animation
  Path B (Face Scene): SDXL+IP-Adapter keyframe → unload → LTX-Video I2V animation
- Sequential VRAM management: only one model loaded at a time
- FP8 quantization + CPU offload for T4 16GB compatibility
"""

import os
import io
import gc
import json
import time
import base64
import tempfile
import threading
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List, Optional
from pyngrok import ngrok

# ── Ngrok Tunnel Setup ──
NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "YOUR_NGROK_AUTH_TOKEN_HERE")
if NGROK_TOKEN != "YOUR_NGROK_AUTH_TOKEN_HERE":
    ngrok.set_auth_token(NGROK_TOKEN)
    public_url = ngrok.connect(8000)
    print(f"🔗 Copy this URL to your local .env: {public_url}")

app = FastAPI(title="Video-DAG-Gen GPU Worker")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Global pipeline holders (lazy-loaded to manage VRAM) ──
_sdxl_pipe = None
_ltx_pipe = None


# ── Request Schema ──
class GenerateRequest(BaseModel):
    prompt: str
    reference_image: Optional[str] = None  # Base64-encoded reference image (from web search)
    face_images: List[str] = []            # Base64-encoded face crops (for SDXL+IP-Adapter)
    needs_face: bool = False               # Whether this scene requires face consistency
    width: int = 768
    height: int = 512
    num_frames: int = 49                   # ~2 seconds at 24fps


# ── VRAM Management ──
def _flush_vram():
    """Aggressively clear all GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _unload_sdxl():
    """Fully unload SDXL pipeline from memory."""
    global _sdxl_pipe
    if _sdxl_pipe is not None:
        print("♻️ Unloading SDXL from memory...")
        del _sdxl_pipe
        _sdxl_pipe = None
        _flush_vram()
        print(f"   VRAM after unload: {torch.cuda.memory_allocated()/1e9:.2f}GB")


def _unload_ltx():
    """Fully unload LTX-Video pipeline from memory."""
    global _ltx_pipe
    if _ltx_pipe is not None:
        print("♻️ Unloading LTX-Video from memory...")
        del _ltx_pipe
        _ltx_pipe = None
        _flush_vram()
        print(f"   VRAM after unload: {torch.cuda.memory_allocated()/1e9:.2f}GB")


# ── Model Loaders ──
def _load_sdxl():
    """Load SDXL + IP-Adapter Plus Face for face-consistent keyframe generation."""
    global _sdxl_pipe
    if _sdxl_pipe is not None:
        return _sdxl_pipe

    # Ensure LTX is unloaded first
    _unload_ltx()

    from diffusers import StableDiffusionXLPipeline
    from transformers import CLIPVisionModelWithProjection

    print("🔄 Loading SDXL + IP-Adapter Plus Face...")
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype=torch.float16
    )

    _sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    _sdxl_pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors"
    )
    _sdxl_pipe.set_ip_adapter_scale(0.7)
    _sdxl_pipe.to(device)

    print(f"✅ SDXL loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return _sdxl_pipe


def _load_ltx():
    """Load LTX-Video Image-to-Video pipeline with T4-optimized settings."""
    global _ltx_pipe
    if _ltx_pipe is not None:
        return _ltx_pipe

    # Ensure SDXL is unloaded first
    _unload_sdxl()

    from diffusers import LTXImageToVideoPipeline

    print("🔄 Loading LTX-Video I2V pipeline in float16 precision...")

    _ltx_pipe = LTXImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-dev",
        torch_dtype=torch.float16
    )
    
    # Enable CPU offload for T4 VRAM management
    _ltx_pipe.enable_model_cpu_offload()

    print(f"✅ LTX-Video loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return _ltx_pipe


# ── Generation Functions ──
def generate_keyframe_with_sdxl(prompt: str, face_images: List[Image.Image]) -> Image.Image:
    """Generate a face-consistent keyframe using SDXL + IP-Adapter."""
    pipe = _load_sdxl()
    pipe.set_ip_adapter_scale(0.7)

    print(f"🎨 Generating face-locked keyframe: {prompt[:60]}...")
    keyframe = pipe(
        prompt=prompt,
        ip_adapter_image=[face_images],
        width=768,
        height=512,
        num_inference_steps=25,
        guidance_scale=7.0
    ).images[0]

    print("✅ Keyframe generated.")
    return keyframe


def animate_image_with_ltx(
    image: Image.Image,
    prompt: str,
    width: int = 768,
    height: int = 512,
    num_frames: int = 49
) -> bytes:
    """Animate a still image into video using LTX-Video I2V."""
    pipe = _load_ltx()

    # Ensure image matches target resolution
    image = image.resize((width, height), Image.LANCZOS)

    print(f"🎬 Animating with LTX-Video ({num_frames} frames at {width}x{height})...")
    print(f"   Prompt: {prompt[:80]}...")

    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted, watermark"

    video_frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=40,
        guidance_scale=5.0,
    ).frames[0]

    # Export frames to video bytes
    from diffusers.utils import export_to_video
    tmp_path = os.path.join(tempfile.gettempdir(), f"ltx_output_{int(time.time())}.mp4")
    export_to_video(video_frames, tmp_path, fps=24)

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()

    # Cleanup temp file
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    print(f"✅ LTX-Video animation complete ({len(video_bytes)} bytes)")
    return video_bytes


# ── API Endpoint ──
@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Main generation endpoint.
    
    Path A: reference_image provided → decode → animate with LTX-Video
    Path B: needs_face=True + face_images → SDXL keyframe → unload → animate with LTX-Video
    Path C: No reference, no face → SDXL text-only keyframe → animate
    """
    print(f"\n{'='*60}")
    print(f"📥 New request: needs_face={req.needs_face}, "
          f"has_reference={req.reference_image is not None}, "
          f"num_faces={len(req.face_images)}")
    print(f"   Prompt: {req.prompt[:80]}...")
    print(f"{'='*60}")

    starting_image = None

    # ── Path A: Using web-sourced reference image ──
    if req.reference_image and not req.needs_face:
        print("📸 Path A: Using web-sourced reference image")
        try:
            img_bytes = base64.b64decode(req.reference_image)
            starting_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            print(f"   Reference image decoded: {starting_image.size}")
        except Exception as e:
            print(f"   ⚠️ Failed to decode reference image: {e}")
            starting_image = None

    # ── Path B: Face-locked keyframe with SDXL ──
    if starting_image is None and req.needs_face and req.face_images:
        print("👤 Path B: Generating face-locked keyframe with SDXL")
        face_pil_images = []
        for img_b64 in req.face_images:
            try:
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                face_pil_images.append(img)
            except Exception as e:
                print(f"   ⚠️ Failed to decode face image: {e}")

        if face_pil_images:
            try:
                starting_image = generate_keyframe_with_sdxl(req.prompt, face_pil_images)
            except Exception as e:
                print(f"   ❌ SDXL keyframe generation failed: {e}")
                starting_image = None

    # ── Path C: No reference, no face → SDXL text-only generation ──
    # Note: Because the IP-Adapter weight is loaded into the UNet model, we MUST satisfy the
    # config's expectation for image embeddings even for text-only generation.
    # We do this by feeding a dummy black image with an IP-Adapter scale of 0.0.
    if starting_image is None:
        print("🖼️ Path C: Generating keyframe with SDXL (text-only + dummy IP-Adapter scale 0.0)")
        try:
            pipe = _load_sdxl()
            pipe.set_ip_adapter_scale(0.0)
            dummy_img = Image.new("RGB", (224, 224), color="black")
            
            starting_image = pipe(
                prompt=req.prompt,
                ip_adapter_image=[[dummy_img]],
                width=req.width,
                height=req.height,
                num_inference_steps=25,
                guidance_scale=7.0
            ).images[0]
        except Exception as e:
            print(f"   ❌ SDXL text-only generation failed: {e}")
            return Response(
                content=json.dumps({"error": f"Keyframe generation failed: {e}"}),
                status_code=500,
                media_type="application/json"
            )

    # ── Animate with LTX-Video ──
    try:
        video_bytes = animate_image_with_ltx(
            image=starting_image,
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames
        )
    except Exception as e:
        print(f"❌ LTX-Video animation failed: {e}")
        import traceback
        traceback.print_exc()
        return Response(
            content=json.dumps({"error": f"Video animation failed: {e}"}),
            status_code=500,
            media_type="application/json"
        )

    return Response(content=video_bytes, media_type="video/mp4")


# ── Health Check ──
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0,
    }


# ── Server Entry Point ──
def start_server():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    print("🚀 Starting Video-DAG-Gen GPU Worker (LTX-Video + SDXL)")
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("Starting FastAPI server in a background thread...")
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print("✅ FastAPI server started on port 8000!")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping server...")
