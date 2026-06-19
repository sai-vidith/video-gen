import os
from dotenv import load_dotenv
load_dotenv(override=True)
import base64
import asyncio
import tempfile
import subprocess
import glob
import cv2
import httpx
from PIL import Image
from pillow_heif import register_heif_opener
import edge_tts
from duckduckgo_search import DDGS

register_heif_opener()

gpu_semaphore = None

def check_ffmpeg() -> bool:
    """Checks if ffmpeg is registered in the environment PATH."""
    try:
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo)
        return True
    except FileNotFoundError:
        return False

# --- OpenCV Face Detector (with defensive guard) ---
def detect_and_crop_face(image_path: str, output_path: str) -> bool:
    """Detects, crops, and resizes a face to 512x512 using OpenCV Haar Cascades."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Warning: OpenCV cascade data path is empty. Falling back to centering crop.")
        # Try a basic central crop as a fallback if the XML wasn't found
        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) / 2
            top = (h - min_dim) / 2
            right = (w + min_dim) / 2
            bottom = (h + min_dim) / 2
            cropped = img.crop((left, top, right, bottom)).resize((512, 512))
            cropped.save(output_path, "JPEG")
            return True
        except Exception:
            return False
            
    # Handle HEIC files by converting them to JPG first
    if image_path.lower().endswith(".heic"):
        try:
            img_pil = Image.open(image_path).convert("RGB")
            temp_jpg = os.path.splitext(image_path)[0] + "_temp.jpg"
            img_pil.save(temp_jpg, "JPEG")
            image_path = temp_jpg
        except Exception as e:
            print(f"HEIC conversion failed for {image_path}: {e}")
            return False

    img = cv2.imread(image_path)
    if img is None:
        return False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    
    if len(faces) == 0:
        # No face found, do a central square crop fallback
        h_dim, w_dim = img.shape[:2]
        min_dim = min(h_dim, w_dim)
        y1 = (h_dim - min_dim) // 2
        x1 = (w_dim - min_dim) // 2
        crop = cv2.resize(img[y1:y1+min_dim, x1:x1+min_dim], (512, 512))
        cv2.imwrite(output_path, crop)
        return True
        
    x, y, w, h = faces[0]
    pad = int(max(w, h) * 0.3)
    y1, y2 = max(0, y - pad), min(img.shape[0], y + h + pad)
    x1, x2 = max(0, x - pad), min(img.shape[1], x + w + pad)
    
    crop = cv2.resize(img[y1:y2, x1:x2], (512, 512))
    cv2.imwrite(output_path, crop)
    return True

# --- Face Profile Manager ---
def create_face_profile(profile_name: str, image_paths: list) -> int:
    """Processes multiple uploaded images, crops faces, and saves to profiles/{name}/"""
    profile_dir = os.path.join("profiles", profile_name)
    os.makedirs(profile_dir, exist_ok=True)
    
    # Remove existing crops in the directory first to refresh
    existing = glob.glob(os.path.join(profile_dir, "face_*.jpg"))
    for f in existing:
        try:
            os.remove(f)
        except Exception:
            pass
            
    success_count = 0
    for idx, path in enumerate(image_paths):
        out_path = os.path.join(profile_dir, f"face_{idx}.jpg")
        if detect_and_crop_face(path, out_path):
            success_count += 1
            
    return success_count

def load_profile_base64_list(profile_name: str) -> list:
    """Reads all face crops for the specified profile and returns them as a list of Base64 strings."""
    profile_dir = os.path.join("profiles", profile_name)
    if not os.path.exists(profile_dir):
        return []
    faces = sorted(glob.glob(os.path.join(profile_dir, "face_*.jpg")))
    
    b64_list = []
    for face_path in faces:
        with open(face_path, "rb") as f:
            b64_list.append(base64.b64encode(f.read()).decode("utf-8"))
    return b64_list

# --- DuckDuckGo Visual Context Search ---
def fetch_reference_image(query: str) -> str | None:
    """Searches DuckDuckGo Images and returns the URL of the top result."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=1))
            if results:
                return results[0]["image"]
    except Exception as e:
        print(f"DuckDuckGo image search failed: {e}")
    return None

# --- TTS & Video Pipeline ---
def get_video_duration(video_path: str) -> float:
    """Extracts exact video duration using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, startupinfo=startupinfo)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        print(f"ffprobe read failed for video {video_path}: {e}")
    return 3.24  # Default output length from Colab Wan model (81 frames at 25fps)

def get_audio_duration(audio_path: str, fallback_text: str = "") -> float:
    """Extracts exact audio file duration using ffprobe with a word-count fallback."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, startupinfo=startupinfo)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        print(f"ffprobe read failed: {e}")
        
    # Word-count duration fallback: ~2.5 words per second + 1.5s margin
    words_count = len(fallback_text.split())
    return max(3.0, (words_count / 2.5) + 1.5)

async def synthesize_voiceover(text: str, output_path: str, voice_type: str = "male_narrator"):
    """Synthesizes text to speech voiceover using Edge-TTS neural speech network."""
    voice_map = {
        "male_narrator": "en-US-GuyNeural",
        "female_narrator": "en-US-JennyNeural",
        "child": "en-US-AnaNeural",
        "deep_voice": "en-US-SteffanNeural",
        "British_male": "en-GB-RyanNeural",
        "British_female": "en-GB-SoniaNeural"
    }
    voice = voice_map.get(voice_type, "en-US-GuyNeural")
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
    except Exception as e:
        print(f"Edge-TTS synthesis failed: {e}. Creating a silent dummy audio file.")
        # Word-count duration calculation: ~2.5 words per second + 1.5s margin
        words_count = len(text.split())
        dur = max(3.0, (words_count / 2.5) + 1.5)
        
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        # Compile silent MP3 of correct duration
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", str(dur), "-c:a", "libmp3lame", output_path
        ]
        subprocess.run(cmd, startupinfo=startupinfo, shell=False)

async def generate_scene_video(assembled_prompt: str, camera_movement: str, face_b64_list: list, scene, groq_api_key: str | None, progress_callback) -> str:
    """Dispatches payload to Google Colab GPU endpoint, falling back to local simulation if offline."""
    global gpu_semaphore
    if gpu_semaphore is None:
        gpu_semaphore = asyncio.Semaphore(1)
        
    colab_url = os.getenv("COLAB_GPU_URL", "").strip()
    
    print(f"[DEBUG] COLAB_GPU_URL = '{colab_url}'")
    print(f"[DEBUG] face_b64_list length = {len(face_b64_list)}")
    
    # ── Path A: 3-Stage Reference Image selection (if not needs_face) ──
    reference_image_b64 = None
    needs_face = getattr(scene, 'needs_face', False)
    
    if not needs_face:
        progress_callback("Running 3-stage LLM-guided reference image selector...")
        try:
            from image_selector import select_reference_image, load_reference_as_base64
            best_candidate = await select_reference_image(scene, groq_api_key or "")
            if best_candidate and best_candidate.local_path:
                reference_image_b64 = load_reference_as_base64(best_candidate.local_path)
                progress_callback(f"Reference image selected (Score: {best_candidate.vision_score:.1f}/10)")
            else:
                progress_callback("No high-quality reference image found. Relying on SDXL keyframe.")
        except Exception as e:
            progress_callback(f"Reference selector error: {e}. Relying on SDXL keyframe.")

    if not colab_url:
        reason = "No COLAB_GPU_URL"
        print(f"[DEBUG] Mock fallback triggered: {reason}")
        progress_callback(f"Running Mock GPU simulation ({reason})...")
        await asyncio.sleep(1.5)
        
        mock_dir = tempfile.gettempdir()
        mock_path = os.path.join(mock_dir, f"mock_scene_{abs(hash(assembled_prompt))}.mp4")
        
        if not os.path.exists(mock_path):
            color = "blue"
            if "dumpster" in assembled_prompt.lower() or "alley" in assembled_prompt.lower():
                color = "darkgreen"
            elif "glowing" in assembled_prompt.lower() or "server" in assembled_prompt.lower():
                color = "darkblue"
            elif "pocket watch" in assembled_prompt.lower():
                color = "brown"
                
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=720x1280:d=10",
                "-pix_fmt", "yuv420p", mock_path
            ]
            subprocess.run(cmd, startupinfo=startupinfo, shell=False)
            
        progress_callback("Completed")
        return mock_path

    # Live Colab T4 Pipeline Path
    progress_callback("Forwarding payload to Google Colab...")
    async with gpu_semaphore:
        print(f"[DEBUG] Sending POST to {colab_url}/generate ...")
        
        payload = {
            "prompt": assembled_prompt,
            "face_images": face_b64_list,
            "reference_image": reference_image_b64,
            "needs_face": needs_face,
            "camera_movement": camera_movement,
            "width": 768,
            "height": 512,
            "num_frames": 97
        }
        
        # CRITICAL: ngrok-skip-browser-warning bypasses free ngrok's interstitial HTML page
        headers = {
            "ngrok-skip-browser-warning": "true",
            "Content-Type": "application/json"
        }
        
        # Large 300.0s timeout to survive sequential T4 processing backlogs
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(f"{colab_url}/generate", json=payload, headers=headers)
                print(f"[DEBUG] Colab response status: {response.status_code}")
                print(f"[DEBUG] Colab response content-type: {response.headers.get('content-type', 'unknown')}")
                print(f"[DEBUG] Colab response size: {len(response.content)} bytes")
                
                if response.status_code == 200 and b"video" in response.headers.get("content-type", "").encode():
                    temp_video = os.path.join(tempfile.gettempdir(), f"scene_{abs(hash(assembled_prompt))}.mp4")
                    with open(temp_video, "wb") as f:
                        f.write(response.content)
                    progress_callback("Completed")
                    return temp_video
                else:
                    body_preview = response.content[:500]
                    print(f"[DEBUG] Unexpected Colab response body: {body_preview}")
                    progress_callback(f"Colab returned unexpected response (status {response.status_code}). Fallback triggered.")
        except Exception as e:
            print(f"[DEBUG] Colab connection exception: {type(e).__name__}: {e}")
            progress_callback(f"Colab link error: {e}. Falling back to simulation...")
        
    # Standard fallback if connection fails
    await asyncio.sleep(1.0)
    mock_path = os.path.join(tempfile.gettempdir(), f"mock_scene_fallback_{abs(hash(assembled_prompt))}.mp4")
    if not os.path.exists(mock_path):
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=720x1280:d=10",
            "-pix_fmt", "yuv420p", mock_path
        ]
        subprocess.run(cmd, startupinfo=startupinfo, shell=False)
    progress_callback("Completed (Fallback)")
    return mock_path

def compile_final_video(video_clips: list, audio_clips: list, voiceovers_text: list, output_path: str):
    """Aligns video durations to match audio exactly using freeze-frame padding, and stitches into portrait reel."""
    temp_dir = tempfile.gettempdir()
    inputs_args = []
    filter_complex = ""
    adjusted_vids = []
    
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    for idx, (vid, aud, text) in enumerate(zip(video_clips, audio_clips, voiceovers_text)):
        target_dur = get_audio_duration(aud, text)
        vid_dur = get_video_duration(vid)
        adj_vid = os.path.join(temp_dir, f"adjusted_scene_{idx}.mp4")
        
        # Build safe list-based arguments to avoid shell parsing bugs on Windows
        if vid_dur >= target_dur:
            cmd = [
                "ffmpeg", "-y", "-i", vid,
                "-vf", "scale=720:1280,setsar=1",
                "-t", str(target_dur),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", adj_vid
            ]
        else:
            freeze_dur = target_dur - vid_dur
            cmd = [
                "ffmpeg", "-y", "-i", vid,
                "-vf", f"tpad=stop_mode=clone:stop_duration={freeze_dur},scale=720:1280,setsar=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", adj_vid
            ]
            
        subprocess.run(cmd, startupinfo=startupinfo, shell=False)
        adjusted_vids.append(adj_vid)
        
        inputs_args.extend(["-i", adj_vid, "-i", aud])
        filter_complex += f"[{2*idx}:v][{2*idx+1}:a]"
        
    filter_complex += f"concat=n={len(video_clips)}:v=1:a=1[v][a]"
    
    full_cmd = ["ffmpeg", "-y"]
    full_cmd.extend(inputs_args)
    full_cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p", output_path
    ])
    
    subprocess.run(full_cmd, startupinfo=startupinfo, shell=False)
