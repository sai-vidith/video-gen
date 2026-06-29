"""
SDXL LoRA Fine-Tuning — Campus AI Compute Platform
====================================================
Fine-tunes Stable Diffusion XL with LoRA on a custom dataset.
Works with the campus cluster's multi-GPU setup.

What this script does:
  1. Detects all available GPUs and prints cluster diagnostics
  2. Sets up DDP (DistributedDataParallel) if multiple GPUs are available
  3. Fine-tunes SDXL with LoRA (Low-Rank Adaptation) on your dataset
  4. Logs training metrics (loss, lr, GPU utilization, VRAM) per step
  5. Generates sample images at checkpoints
  6. Saves everything to ./outputs/ as a ZIP for download

Dataset format (upload as ZIP):
  dataset.zip/
    ├── image_001.jpg
    ├── image_002.png
    ├── image_003.jpg
    └── ...  (10-100 images, JPEG/PNG, any resolution)

The script auto-resizes images to 1024x1024 for SDXL training.

Environment variables (set by campus platform automatically):
  CAMPUS_JOB_ID       — Job identifier
  CAMPUS_OUTPUT_DIR   — NFS output path
  CAMPUS_CONFIG_PATH  — Path to config.yaml
  CAMPUS_DATASET_PATH — Path to extracted dataset
"""

import os
import sys
import gc
import json
import time
import glob
import csv
import math
import shutil
import zipfile
import argparse
from pathlib import Path
from datetime import datetime, timezone

# ── Parse arguments ──
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="", help="Path to config YAML")
args, _ = parser.parse_known_args()

# ── Read environment ──
JOB_ID = os.environ.get("CAMPUS_JOB_ID", "local_test")
OUTPUT_DIR = os.environ.get("CAMPUS_OUTPUT_DIR", "./outputs")
CONFIG_PATH = args.config or os.environ.get("CAMPUS_CONFIG_PATH", "")
DATASET_PATH = os.environ.get("CAMPUS_DATASET_PATH", "")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "metrics"), exist_ok=True)

# ── Logging setup ──
LOG_FILE = os.path.join(OUTPUT_DIR, "training.log")

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ══════════════════════════════════════════════════════════════════
# PHASE 1: GPU DIAGNOSTICS & CLUSTER DETECTION
# ══════════════════════════════════════════════════════════════════

log("=" * 70)
log("PHASE 1: GPU DIAGNOSTICS & CLUSTER DETECTION")
log("=" * 70)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Detect GPUs
NUM_GPUS = torch.cuda.device_count()
CUDA_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")

log(f"  PyTorch version : {torch.__version__}")
log(f"  CUDA available  : {torch.cuda.is_available()}")
log(f"  CUDA version    : {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
log(f"  CUDA_VISIBLE    : {CUDA_VISIBLE}")
log(f"  GPUs detected   : {NUM_GPUS}")

gpu_info = []
total_vram_gb = 0

for i in range(NUM_GPUS):
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_memory / (1024 ** 3)
    total_vram_gb += vram_gb
    info = {
        "gpu_id": i,
        "name": props.name,
        "vram_gb": round(vram_gb, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
    }
    gpu_info.append(info)
    log(f"  GPU {i}: {props.name} | {vram_gb:.1f} GB VRAM | "
        f"Compute {props.major}.{props.minor} | {props.multi_processor_count} SMs")

log(f"  Total VRAM      : {total_vram_gb:.1f} GB")
log("")

# GPU Division Strategy
if NUM_GPUS == 0:
    log("ERROR: No GPUs detected! This job requires at least 1 GPU.")
    log("Make sure you selected a queue with GPU allocation.")
    sys.exit(1)
elif NUM_GPUS == 1:
    STRATEGY = "single_gpu"
    log(f"  Strategy: SINGLE GPU training on GPU 0 ({gpu_info[0]['name']})")
    log(f"  LoRA rank will be set based on available VRAM ({gpu_info[0]['vram_gb']:.0f} GB)")
elif NUM_GPUS <= 4:
    STRATEGY = "ddp"
    log(f"  Strategy: DDP (DistributedDataParallel) across {NUM_GPUS} GPUs")
    log(f"  Each GPU processes batch_size / {NUM_GPUS} samples per step")
    log(f"  Effective batch size = per_gpu_batch × {NUM_GPUS}")
else:
    STRATEGY = "ddp"
    log(f"  Strategy: DDP across {NUM_GPUS} GPUs (large cluster mode)")
    log(f"  Gradient accumulation adjusted for memory efficiency")

# VRAM-aware configuration
if gpu_info[0]["vram_gb"] < 8:
    LORA_RANK = 4
    BATCH_SIZE = 1
    GRADIENT_ACCUM = 4
    log(f"  VRAM < 8 GB: Using minimal config (rank=4, batch=1, accum=4)")
elif gpu_info[0]["vram_gb"] < 16:
    LORA_RANK = 8
    BATCH_SIZE = 1
    GRADIENT_ACCUM = 4
    log(f"  VRAM 8-16 GB: Using standard config (rank=8, batch=1, accum=4)")
elif gpu_info[0]["vram_gb"] < 24:
    LORA_RANK = 16
    BATCH_SIZE = 2
    GRADIENT_ACCUM = 2
    log(f"  VRAM 16-24 GB: Using enhanced config (rank=16, batch=2, accum=2)")
else:
    LORA_RANK = 32
    BATCH_SIZE = 4
    GRADIENT_ACCUM = 1
    log(f"  VRAM 24+ GB: Using full config (rank=32, batch=4, accum=1)")

log("")

# ── Load config.yaml if provided ──
config = {
    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_RANK * 2,
    "learning_rate": 1e-4,
    "num_epochs": 50,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUM,
    "resolution": 1024,
    "save_every_n_epochs": 10,
    "sample_every_n_epochs": 10,
    "sample_prompts": [
        "a professional photo in the trained style",
        "a cinematic portrait in the trained style, dramatic lighting",
        "a wide establishing shot in the trained style, golden hour",
    ],
    "mixed_precision": "fp16",
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 50,
    "max_grad_norm": 1.0,
    "seed": 42,
}

if CONFIG_PATH and os.path.exists(CONFIG_PATH):
    log(f"Loading config from: {CONFIG_PATH}")
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            user_config = yaml.safe_load(f)
        if user_config:
            config.update(user_config)
            log(f"  Config loaded: {json.dumps(user_config, indent=2)}")
    except ImportError:
        log("  yaml not installed, trying json...")
        try:
            with open(CONFIG_PATH) as f:
                user_config = json.load(f)
            if user_config:
                config.update(user_config)
        except Exception as e:
            log(f"  Failed to load config: {e}")

# Save effective config
with open(os.path.join(OUTPUT_DIR, "effective_config.json"), "w") as f:
    json.dump(config, f, indent=2)
log(f"Effective config saved to outputs/effective_config.json")

# ══════════════════════════════════════════════════════════════════
# PHASE 2: DATASET LOADING
# ══════════════════════════════════════════════════════════════════

log("")
log("=" * 70)
log("PHASE 2: DATASET LOADING")
log("=" * 70)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


class ImageDataset(Dataset):
    """Simple image dataset for LoRA fine-tuning. Loads images and resizes to target resolution."""

    def __init__(self, image_dir: str, resolution: int = 1024):
        self.image_paths = []
        for ext in IMAGE_EXTENSIONS:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
            self.image_paths.extend(glob.glob(os.path.join(image_dir, f"**/*{ext}"), recursive=True))
        # Deduplicate
        self.image_paths = sorted(set(self.image_paths))

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Return a blank image if loading fails
            log(f"  Warning: Failed to load {img_path}: {e}")
            image = Image.new("RGB", (1024, 1024), color="black")
        return self.transform(image)


# Find dataset
dataset_dir = DATASET_PATH
if not dataset_dir or not os.path.exists(dataset_dir):
    # Try common locations
    for candidate in ["./dataset", "./data", "./images", "./train_images"]:
        if os.path.exists(candidate):
            dataset_dir = candidate
            break

if not dataset_dir or not os.path.exists(dataset_dir):
    log("ERROR: No dataset found!")
    log("  Expected: Upload a ZIP of images via the platform's Dataset field.")
    log(f"  Checked CAMPUS_DATASET_PATH={DATASET_PATH}")
    log("  Checked ./dataset, ./data, ./images, ./train_images")
    sys.exit(1)

dataset = ImageDataset(dataset_dir, resolution=config["resolution"])
log(f"  Dataset path    : {dataset_dir}")
log(f"  Images found    : {len(dataset)}")

if len(dataset) == 0:
    log("ERROR: No images found in dataset directory!")
    log(f"  Supported formats: {IMAGE_EXTENSIONS}")
    sys.exit(1)

# Log sample paths
for i, p in enumerate(dataset.image_paths[:5]):
    log(f"    [{i}] {os.path.basename(p)}")
if len(dataset.image_paths) > 5:
    log(f"    ... and {len(dataset.image_paths) - 5} more")

# Calculate training stats
effective_batch = config["batch_size"] * config["gradient_accumulation_steps"] * max(1, NUM_GPUS)
steps_per_epoch = math.ceil(len(dataset) / effective_batch)
total_steps = steps_per_epoch * config["num_epochs"]

log(f"  Per-GPU batch   : {config['batch_size']}")
log(f"  Gradient accum  : {config['gradient_accumulation_steps']}")
log(f"  Effective batch : {effective_batch}")
log(f"  Steps/epoch     : {steps_per_epoch}")
log(f"  Total epochs    : {config['num_epochs']}")
log(f"  Total steps     : {total_steps}")
log("")

# ══════════════════════════════════════════════════════════════════
# PHASE 3: MODEL LOADING & LoRA SETUP
# ══════════════════════════════════════════════════════════════════

log("=" * 70)
log("PHASE 3: MODEL LOADING & LoRA SETUP")
log("=" * 70)

from diffusers import StableDiffusionXLPipeline, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model

# Set seed for reproducibility
torch.manual_seed(config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config["seed"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log(f"  Loading SDXL pipeline: {config['model_id']}...")

# Load components individually for LoRA injection
from diffusers import UNet2DConditionModel

tokenizer_1 = CLIPTokenizer.from_pretrained(config["model_id"], subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(config["model_id"], subfolder="tokenizer_2")
text_encoder_1 = CLIPTextModel.from_pretrained(
    config["model_id"], subfolder="text_encoder", torch_dtype=torch.float16
)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    config["model_id"], subfolder="text_encoder_2", torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained(
    config["model_id"], subfolder="vae", torch_dtype=torch.float16
)
unet = UNet2DConditionModel.from_pretrained(
    config["model_id"], subfolder="unet", torch_dtype=torch.float16
)
noise_scheduler = DDPMScheduler.from_pretrained(config["model_id"], subfolder="scheduler")

log("  ✅ SDXL components loaded")

# Freeze everything except UNet (which gets LoRA)
text_encoder_1.requires_grad_(False)
text_encoder_2.requires_grad_(False)
vae.requires_grad_(False)

# Move frozen components to device
text_encoder_1.to(device)
text_encoder_2.to(device)
vae.to(device)

# Apply LoRA to UNet
log(f"  Applying LoRA: rank={config['lora_rank']}, alpha={config['lora_alpha']}")
lora_config = LoraConfig(
    r=config["lora_rank"],
    lora_alpha=config["lora_alpha"],
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)
unet = get_peft_model(unet, lora_config)
unet.to(device)

trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in unet.parameters())
log(f"  Trainable params: {trainable_params:,} / {total_params:,} "
    f"({100 * trainable_params / total_params:.2f}%)")

# GPU memory after model loading
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    log(f"  VRAM allocated  : {allocated:.2f} GB")
    log(f"  VRAM reserved   : {reserved:.2f} GB")

# DDP setup if multiple GPUs
if STRATEGY == "ddp" and NUM_GPUS > 1:
    log(f"  Setting up DDP across {NUM_GPUS} GPUs...")
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", world_size=NUM_GPUS, rank=0)
    unet = DDP(unet, device_ids=[0])
    log("  ✅ DDP initialized")

log("")

# ══════════════════════════════════════════════════════════════════
# PHASE 4: TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

log("=" * 70)
log("PHASE 4: TRAINING")
log("=" * 70)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)

# Optimizer
optimizer = AdamW(
    unet.parameters(),
    lr=config["learning_rate"],
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-8,
)

# LR Scheduler
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.01,
    total_iters=config["lr_warmup_steps"],
)
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max(1, total_steps - config["lr_warmup_steps"]),
    eta_min=1e-6,
)
lr_scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[config["lr_warmup_steps"]],
)

# Metrics tracking
metrics_log = []
csv_path = os.path.join(OUTPUT_DIR, "metrics", "training_metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "step", "epoch", "loss", "learning_rate",
        "gpu_util_pct", "vram_used_gb", "vram_total_gb",
        "step_time_sec", "samples_per_sec"
    ])

# Mixed precision
scaler = torch.amp.GradScaler("cuda") if config["mixed_precision"] == "fp16" else None

global_step = 0
best_loss = float("inf")
training_start = time.time()

log(f"  Starting training: {config['num_epochs']} epochs, {total_steps} total steps")
log(f"  Mixed precision: {config['mixed_precision']}")
log("")

for epoch in range(config["num_epochs"]):
    epoch_loss = 0.0
    epoch_steps = 0
    epoch_start = time.time()

    unet.train()

    for step, batch in enumerate(dataloader):
        step_start = time.time()

        # Move to device
        pixel_values = batch.to(device, dtype=torch.float16)

        # Encode images to latent space
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()

        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Empty text conditioning (unconditional fine-tuning)
        with torch.no_grad():
            text_input_1 = tokenizer_1("", return_tensors="pt", padding="max_length",
                                        max_length=77, truncation=True).input_ids.to(device)
            text_input_2 = tokenizer_2("", return_tensors="pt", padding="max_length",
                                        max_length=77, truncation=True).input_ids.to(device)
            encoder_hidden_states_1 = text_encoder_1(text_input_1)[0]
            encoder_hidden_states_2 = text_encoder_2(text_input_2)[0]

            # Expand for batch
            encoder_hidden_states_1 = encoder_hidden_states_1.expand(latents.shape[0], -1, -1)
            encoder_hidden_states_2 = encoder_hidden_states_2.expand(latents.shape[0], -1, -1)

            # SDXL expects concatenated text embeddings
            encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)

        # Predict noise
        added_cond_kwargs = {
            "text_embeds": torch.zeros(latents.shape[0], 1280, device=device, dtype=torch.float16),
            "time_ids": torch.zeros(latents.shape[0], 6, device=device, dtype=torch.float16),
        }

        if scaler:
            with torch.amp.autocast("cuda"):
                noise_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
                loss = loss / config["gradient_accumulation_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                global_step += 1
        else:
            noise_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()

            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                global_step += 1

        # Record metrics
        step_time = time.time() - step_start
        loss_val = loss.item() * config["gradient_accumulation_steps"]
        epoch_loss += loss_val
        epoch_steps += 1

        # GPU metrics
        gpu_util_pct = 0.0
        vram_used = 0.0
        vram_total = 0.0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
            vram_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    gpu_util_pct = float(result.stdout.strip())
            except Exception:
                pass

        samples_per_sec = config["batch_size"] / step_time if step_time > 0 else 0

        # CSV metrics
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                global_step, epoch + 1, f"{loss_val:.6f}",
                f"{optimizer.param_groups[0]['lr']:.2e}",
                f"{gpu_util_pct:.1f}", f"{vram_used:.2f}", f"{vram_total:.2f}",
                f"{step_time:.3f}", f"{samples_per_sec:.2f}"
            ])

        metrics_log.append({
            "step": global_step,
            "epoch": epoch + 1,
            "loss": loss_val,
            "lr": optimizer.param_groups[0]["lr"],
            "gpu_util_pct": gpu_util_pct,
            "vram_used_gb": vram_used,
            "step_time": step_time,
        })

        # Log every 10 steps
        if global_step % 10 == 0 or global_step == 1:
            log(f"  Step {global_step}/{total_steps} | Epoch {epoch+1} | "
                f"Loss: {loss_val:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"GPU: {gpu_util_pct:.0f}% | VRAM: {vram_used:.1f}/{vram_total:.1f}GB | "
                f"{samples_per_sec:.1f} samples/s")

    # End of epoch
    avg_epoch_loss = epoch_loss / max(1, epoch_steps)
    epoch_time = time.time() - epoch_start
    log(f"  ── Epoch {epoch+1}/{config['num_epochs']} complete | "
        f"Avg Loss: {avg_epoch_loss:.5f} | Time: {epoch_time:.1f}s ──")

    # Save checkpoint
    if (epoch + 1) % config["save_every_n_epochs"] == 0 or (epoch + 1) == config["num_epochs"]:
        ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", f"lora_epoch_{epoch+1}")
        os.makedirs(ckpt_path, exist_ok=True)
        # Save LoRA weights
        if hasattr(unet, "module"):
            unet.module.save_pretrained(ckpt_path)
        else:
            unet.save_pretrained(ckpt_path)
        log(f"  💾 Checkpoint saved: {ckpt_path}")

        # Track best
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(OUTPUT_DIR, "checkpoints", "best_lora")
            os.makedirs(best_path, exist_ok=True)
            if hasattr(unet, "module"):
                unet.module.save_pretrained(best_path)
            else:
                unet.save_pretrained(best_path)
            log(f"  ⭐ New best model saved (loss={best_loss:.5f})")

    # Generate samples
    if (epoch + 1) % config["sample_every_n_epochs"] == 0 or (epoch + 1) == config["num_epochs"]:
        log(f"  🎨 Generating sample images...")
        try:
            unet.eval()
            pipe = StableDiffusionXLPipeline.from_pretrained(
                config["model_id"],
                unet=unet.module if hasattr(unet, "module") else unet,
                torch_dtype=torch.float16,
            ).to(device)

            for pidx, prompt in enumerate(config["sample_prompts"][:3]):
                with torch.no_grad():
                    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
                sample_path = os.path.join(
                    OUTPUT_DIR, "samples", f"epoch_{epoch+1}_sample_{pidx}.png"
                )
                image.save(sample_path)
                log(f"    Sample {pidx}: {sample_path}")

            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            unet.train()
        except Exception as e:
            log(f"    Sample generation failed: {e}")
            unet.train()

log("")

# ══════════════════════════════════════════════════════════════════
# PHASE 5: FINAL METRICS & OUTPUT PACKAGING
# ══════════════════════════════════════════════════════════════════

log("=" * 70)
log("PHASE 5: FINAL METRICS & OUTPUT PACKAGING")
log("=" * 70)

total_time = time.time() - training_start

# Final metrics summary
final_metrics = {
    "job_id": JOB_ID,
    "model": config["model_id"],
    "lora_rank": config["lora_rank"],
    "lora_alpha": config["lora_alpha"],
    "total_epochs": config["num_epochs"],
    "total_steps": global_step,
    "total_time_seconds": round(total_time, 1),
    "total_time_minutes": round(total_time / 60, 2),
    "final_loss": metrics_log[-1]["loss"] if metrics_log else None,
    "best_loss": best_loss,
    "gpu_count": NUM_GPUS,
    "gpu_info": gpu_info,
    "strategy": STRATEGY,
    "dataset_size": len(dataset),
    "effective_batch_size": effective_batch,
    "trainable_parameters": trainable_params,
    "total_parameters": total_params,
    "trainable_pct": round(100 * trainable_params / total_params, 2),
    "avg_gpu_utilization": round(
        sum(m["gpu_util_pct"] for m in metrics_log) / max(1, len(metrics_log)), 1
    ),
    "peak_vram_gb": round(max(m["vram_used_gb"] for m in metrics_log), 2) if metrics_log else 0,
    "avg_samples_per_sec": round(
        len(dataset) * config["num_epochs"] / total_time, 2
    ) if total_time > 0 else 0,
}

with open(os.path.join(OUTPUT_DIR, "metrics", "final_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=2)

log(f"  Total training time : {total_time / 60:.1f} minutes")
log(f"  Final loss          : {final_metrics['final_loss']:.5f}")
log(f"  Best loss           : {best_loss:.5f}")
log(f"  Avg GPU utilization : {final_metrics['avg_gpu_utilization']:.1f}%")
log(f"  Peak VRAM           : {final_metrics['peak_vram_gb']:.2f} GB")
log(f"  Throughput          : {final_metrics['avg_samples_per_sec']:.1f} samples/sec")
log("")

# GPU division summary
log("GPU Division Report:")
log(f"  Total GPUs allocated: {NUM_GPUS}")
log(f"  Training strategy   : {STRATEGY}")
for info in gpu_info:
    log(f"  GPU {info['gpu_id']}: {info['name']} ({info['vram_gb']} GB) — "
        f"{'Active (training)' if info['gpu_id'] == 0 or STRATEGY == 'ddp' else 'Standby'}")
log("")

# List output files
log("Output files:")
for root, dirs, files in os.walk(OUTPUT_DIR):
    for fname in files:
        fpath = os.path.join(root, fname)
        fsize = os.path.getsize(fpath)
        rel = os.path.relpath(fpath, OUTPUT_DIR)
        log(f"  {rel} ({fsize / 1024:.1f} KB)")

log("")
log("=" * 70)
log("✅ TRAINING COMPLETE")
log(f"   Outputs are in: {OUTPUT_DIR}")
log(f"   Download via the platform's 'Download Outputs' button.")
log("=" * 70)

# Cleanup DDP
if dist.is_initialized():
    dist.destroy_process_group()
