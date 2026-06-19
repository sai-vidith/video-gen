"""
3-Stage LLM-Guided Reference Image Selector

Stage 1: Multi-query DDG image search (3 queries × 3 results = 9 candidates)
Stage 2: Automated filtering (resolution, aspect ratio, color variance)
Stage 3: Groq Vision scoring (llama-3.2-90b-vision-preview) — rates 1-10

Returns the best-scoring image (≥ 6) or None if no good match is found.
"""

import os
import io
import base64
import tempfile
import hashlib
import asyncio
import requests
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from PIL import Image
from duckduckgo_search import DDGS

# Minimum acceptable vision score (1-10 scale)
VISION_SCORE_THRESHOLD = 6
# Target resolution for reference images fed to Wan I2V
TARGET_WIDTH = 832
TARGET_HEIGHT = 480


@dataclass
class ImageCandidate:
    """A candidate reference image with metadata and quality scores."""
    url: str
    local_path: str = ""
    width: int = 0
    height: int = 0
    color_variance: float = 0.0
    vision_score: float = 0.0
    rejection_reason: str = ""


def generate_search_queries(scene) -> List[str]:
    """
    Generates 3 diverse search queries from scene attributes.
    Uses different combinations of visual_prompt, lighting, mood, and reference_search_query
    to cast a wide net for relevant images.
    """
    queries = []
    
    # Query 1: Use the dedicated reference_search_query if available
    if hasattr(scene, 'reference_search_query') and scene.reference_search_query:
        queries.append(scene.reference_search_query)
    elif hasattr(scene, 'search_queries') and scene.search_queries:
        queries.append(scene.search_queries[0])
    else:
        # Fallback: extract key nouns from visual_prompt
        queries.append(scene.visual_prompt[:80])
    
    # Query 2: Mood + setting combination
    mood = getattr(scene, 'mood', 'cinematic')
    visual = getattr(scene, 'visual_prompt', '')
    # Take first 50 chars of visual prompt + mood for a different angle
    queries.append(f"{mood} {visual[:50]} photography")
    
    # Query 3: Lighting + color palette style reference
    lighting = getattr(scene, 'lighting', '')
    color = getattr(scene, 'color_palette', '')
    if lighting and color:
        queries.append(f"{lighting} {color} cinematic still")
    else:
        queries.append(f"cinematic {mood} scene film still")
    
    # If scene has pre-generated search_queries from LLM, use those instead
    if hasattr(scene, 'search_queries') and len(scene.search_queries) >= 3:
        queries = scene.search_queries[:3]
    
    return queries[:3]


def _search_ddg_images(query: str, max_results: int = 3) -> List[str]:
    """Searches DuckDuckGo Images and returns URLs of top results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_results))
            return [r["image"] for r in results if "image" in r]
    except Exception as e:
        print(f"[ImageSelector] DDG search failed for '{query}': {e}")
        return []


def _download_image(url: str, save_dir: str) -> Optional[str]:
    """Downloads an image from URL and saves to a local temp file."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type and "octet-stream" not in content_type:
            return None
        
        # Generate unique filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        ext = ".jpg"
        if "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"
        
        local_path = os.path.join(save_dir, f"ref_{url_hash}{ext}")
        
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return local_path
    except Exception as e:
        print(f"[ImageSelector] Download failed for {url[:60]}...: {e}")
        return None


def _apply_automated_filters(candidates: List[ImageCandidate]) -> List[ImageCandidate]:
    """
    Stage 2: Apply automated quality filters.
    Rejects images that are too small, have wrong aspect ratio, or are near-solid colors.
    """
    filtered = []
    
    for candidate in candidates:
        if not candidate.local_path or not os.path.exists(candidate.local_path):
            candidate.rejection_reason = "download_failed"
            continue
        
        try:
            img = Image.open(candidate.local_path).convert("RGB")
            w, h = img.size
            candidate.width = w
            candidate.height = h
            
            # Filter 1: Minimum resolution (reject thumbnails)
            if w < 400 or h < 300:
                candidate.rejection_reason = f"too_small ({w}x{h})"
                continue
            
            # Filter 2: Aspect ratio sanity (reject extreme banners/strips)
            aspect = w / h
            if aspect > 4.0 or aspect < 0.25:
                candidate.rejection_reason = f"extreme_aspect_ratio ({aspect:.2f})"
                continue
            
            # Filter 3: Color variance check (reject near-solid/gradient images)
            img_small = img.resize((64, 64))
            pixels = np.array(img_small, dtype=np.float32)
            variance = np.var(pixels)
            candidate.color_variance = float(variance)
            
            if variance < 200:  # Very low variance = likely solid color or simple gradient
                candidate.rejection_reason = f"low_color_variance ({variance:.0f})"
                continue
            
            filtered.append(candidate)
            
        except Exception as e:
            candidate.rejection_reason = f"image_open_error: {e}"
            continue
    
    return filtered


def _score_with_groq_vision(
    image_path: str,
    scene_description: str,
    api_key: str
) -> float:
    """
    Stage 3: Uses Groq Vision (llama-3.2-90b-vision-preview) to score
    how well an image matches the desired scene description.
    Returns a score from 1-10.
    """
    try:
        # Read and encode image to base64
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        
        # Resize if too large (Groq has payload limits)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.LANCZOS)
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        scoring_prompt = (
            f"You are an expert cinematographer evaluating reference images for a video production.\n\n"
            f"DESIRED SCENE: {scene_description}\n\n"
            f"Rate this image from 1-10 for how well it matches the desired scene.\n"
            f"Consider: subject matter, setting, lighting mood, color palette, composition.\n\n"
            f"Scoring guide:\n"
            f"9-10: Perfect match — composition, lighting, mood all align\n"
            f"7-8: Strong match — correct subject/setting, minor style differences\n"
            f"5-6: Acceptable — right category but noticeable mismatches\n"
            f"3-4: Poor — wrong setting, lighting, or mood\n"
            f"1-2: Irrelevant — completely different subject matter\n\n"
            f"Respond with ONLY a single integer from 1 to 10. Nothing else."
        )
        
        payload = {
            "model": "llama-3.2-90b-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": scoring_prompt
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 5
        }
        
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        r.raise_for_status()
        
        result = r.json()
        raw_score = result["choices"][0]["message"]["content"].strip()
        
        # Parse the numeric score
        score = int("".join(c for c in raw_score if c.isdigit())[:2])
        score = max(1, min(10, score))
        
        print(f"[ImageSelector] Vision score for {os.path.basename(image_path)}: {score}/10")
        return float(score)
        
    except Exception as e:
        print(f"[ImageSelector] Groq Vision scoring failed: {e}")
        return 0.0


def _prepare_reference_image(image_path: str, output_path: str) -> str:
    """
    Resizes and crops the selected reference image to the target resolution
    for Wan 2.1 I2V input (832×480 landscape).
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # Calculate crop to match target aspect ratio (832/480 = 1.733)
    target_aspect = TARGET_WIDTH / TARGET_HEIGHT
    current_aspect = w / h
    
    if current_aspect > target_aspect:
        # Image is wider — crop sides
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        # Image is taller — crop top/bottom
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    
    # Resize to exact target
    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
    img.save(output_path, "PNG")
    
    return output_path


async def select_reference_image(
    scene,
    groq_api_key: str,
    output_dir: Optional[str] = None
) -> Optional[ImageCandidate]:
    """
    Full 3-stage reference image selection pipeline.
    
    Args:
        scene: A StorySceneNode with visual_prompt, mood, lighting, etc.
        groq_api_key: Groq API key for vision scoring
        output_dir: Directory to save downloaded images (defaults to temp)
    
    Returns:
        Best ImageCandidate with score ≥ VISION_SCORE_THRESHOLD, or None
    """
    if not output_dir:
        output_dir = os.path.join(tempfile.gettempdir(), "video_dag_refs")
    os.makedirs(output_dir, exist_ok=True)
    
    scene_desc = getattr(scene, 'visual_prompt', str(scene))
    scene_id = getattr(scene, 'id', 'unknown')
    
    print(f"\n[ImageSelector] === Processing {scene_id} ===")
    print(f"[ImageSelector] Scene: {scene_desc[:80]}...")
    
    # ── Stage 1: Multi-Query DDG Search ──
    queries = generate_search_queries(scene)
    print(f"[ImageSelector] Stage 1: Searching with {len(queries)} queries...")
    
    all_urls = []
    for q in queries:
        urls = _search_ddg_images(q, max_results=3)
        print(f"  Query '{q[:50]}...' → {len(urls)} results")
        all_urls.extend(urls)
    
    # Deduplicate
    all_urls = list(dict.fromkeys(all_urls))
    print(f"[ImageSelector] Total unique URLs: {len(all_urls)}")
    
    if not all_urls:
        print(f"[ImageSelector] No images found. Falling back to SDXL.")
        return None
    
    # Download all candidates
    candidates = []
    for url in all_urls:
        local_path = _download_image(url, output_dir)
        candidates.append(ImageCandidate(url=url, local_path=local_path or ""))
    
    # ── Stage 2: Automated Filters ──
    print(f"[ImageSelector] Stage 2: Filtering {len(candidates)} candidates...")
    filtered = _apply_automated_filters(candidates)
    print(f"[ImageSelector] {len(filtered)} candidates passed filters")
    
    # Log rejections
    rejected = [c for c in candidates if c.rejection_reason]
    for c in rejected:
        print(f"  Rejected: {c.rejection_reason}")
    
    if not filtered:
        print(f"[ImageSelector] All candidates filtered out. Falling back to SDXL.")
        return None
    
    # ── Stage 3: Groq Vision Scoring ──
    if not groq_api_key:
        print(f"[ImageSelector] No Groq API key — skipping vision scoring, using first candidate.")
        best = filtered[0]
        best.vision_score = 7.0  # Assume acceptable without scoring
        return best
    
    print(f"[ImageSelector] Stage 3: Scoring {len(filtered)} candidates with Groq Vision...")
    
    # Score candidates (limit to top 5 to stay within rate limits)
    for candidate in filtered[:5]:
        candidate.vision_score = _score_with_groq_vision(
            candidate.local_path,
            scene_desc,
            groq_api_key
        )
        # Small delay to respect Groq rate limits
        await asyncio.sleep(0.5)
    
    # Sort by vision score descending
    scored = sorted(filtered[:5], key=lambda c: c.vision_score, reverse=True)
    best = scored[0]
    
    print(f"[ImageSelector] Best candidate: score={best.vision_score}/10, "
          f"resolution={best.width}x{best.height}")
    
    if best.vision_score >= VISION_SCORE_THRESHOLD:
        # Prepare the image for Wan I2V input
        prepared_path = os.path.join(
            output_dir,
            f"prepared_{scene_id}.png"
        )
        _prepare_reference_image(best.local_path, prepared_path)
        best.local_path = prepared_path
        print(f"[ImageSelector] ✅ Reference image selected and prepared: {prepared_path}")
        return best
    else:
        print(f"[ImageSelector] ❌ Best score {best.vision_score} < {VISION_SCORE_THRESHOLD}. "
              f"Falling back to SDXL.")
        return None


def load_reference_as_base64(image_path: str) -> str:
    """Reads a prepared reference image and returns it as a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
