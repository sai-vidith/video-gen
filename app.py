import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import os
import glob
import asyncio
import tempfile
import shutil
from PIL import Image
from pillow_heif import register_heif_opener
from dotenv import load_dotenv
from context_engine import generate_storyboard, build_assembled_prompt
from video_pipeline import (
    create_face_profile,
    load_profile_base64_list,
    fetch_reference_image,
    synthesize_voiceover,
    generate_scene_video,
    compile_final_video,
    check_ffmpeg
)

# Load configuration values from environment variables
load_dotenv(override=True)
register_heif_opener()

st.set_page_config(
    page_title="Consistent Face Video DAG Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling matching standard AI products
st.markdown("""
<style>
    body {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .main {
        background-color: #0b0f19;
    }
    .stTextArea textarea {
        background-color: #1e293b;
        color: #f1f5f9;
        border: 1px solid #475569;
        border-radius: 12px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.8rem;
        transition: transform 0.15s ease, opacity 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        opacity: 0.95;
    }
</style>
""", unsafe_allow_html=True)

# System-level dependencies check
ffmpeg_installed = check_ffmpeg()

st.title("🎬 Consistent Face-Locked Video DAG Engine")
st.write("Orchestrate parallel scene generation and build customizable, face-consistent short-form video reels.")

if not ffmpeg_installed:
    st.warning("⚠️ **FFmpeg not detected in PATH**: Splicing operations will execute dummy pipelines. Please install FFmpeg on your host machine to support full compilation.")

# Read active API profiles
llm_configured = bool(os.getenv("GROQ_API_KEY")) or bool(os.getenv("GEMINI_API_KEY"))
colab_url = os.getenv("COLAB_GPU_URL", "")

if not llm_configured:
    st.info("ℹ️ **LLM API Key** (GROQ_API_KEY or GEMINI_API_KEY) not configured in `.env`. The app will run using fallback story scripts.")
if not colab_url:
    st.info("ℹ️ **COLAB_GPU_URL** not configured in `.env`. The app will trigger local mock video simulations.")
else:
    st.success(f"🔗 **Colab Worker Connection Enabled**: Routing requests to `{colab_url}`")

# Sidebar - Configurations & Image Selectors
with st.sidebar:
    st.header("👤 Face Profile Manager")
    
    # 1. Create New Profile
    with st.expander("➕ Create New Profile", expanded=False):
        new_profile_name = st.text_input("Profile Name:", placeholder="e.g., Bob")
        uploaded_files = st.file_uploader(
            "Select reference photos (HEIC/PNG/JPG):",
            type=["heic", "png", "jpg", "jpeg"],
            accept_multiple_files=True
        )
        
        if st.button("Save Profile"):
            if not new_profile_name.strip():
                st.error("Please enter a profile name.")
            elif not uploaded_files:
                st.error("Please upload at least one image.")
            else:
                with st.spinner("Processing face inputs..."):
                    # Save files to a temporary directory to process
                    temp_dir = tempfile.mkdtemp()
                    temp_paths = []
                    for idx, uf in enumerate(uploaded_files):
                        suffix = os.path.splitext(uf.name)[1]
                        temp_path = os.path.join(temp_dir, f"raw_{idx}{suffix}")
                        with open(temp_path, "wb") as f:
                            f.write(uf.getbuffer())
                        temp_paths.append(temp_path)
                        
                    processed = create_face_profile(new_profile_name.strip(), temp_paths)
                    shutil.rmtree(temp_dir)
                    
                    if processed > 0:
                        st.success(f"Successfully created profile '{new_profile_name}' with {processed} faces!")
                        st.rerun()
                    else:
                        st.error("Could not find faces in any of the uploaded images. Try different angles/lighting.")

    # 2. Select Profile
    profiles_dir = "profiles"
    os.makedirs(profiles_dir, exist_ok=True)
    existing_profiles = [os.path.basename(p) for p in glob.glob(os.path.join(profiles_dir, "*")) if os.path.isdir(p)]
    
    active_profile = st.selectbox(
        "Select Active Face Profile:",
        options=["[ None / Mock Simulation ]"] + existing_profiles
    )
    
    # 3. Preview Active Profile Thumbnail Gallery
    if active_profile != "[ None / Mock Simulation ]":
        st.subheader("🔍 Reference Crops")
        profile_path = os.path.join(profiles_dir, active_profile)
        face_images = glob.glob(os.path.join(profile_path, "face_*.jpg"))
        
        if face_images:
            cols = st.columns(min(len(face_images), 3))
            for idx, img_path in enumerate(face_images):
                col_idx = idx % len(cols)
                try:
                    img = Image.open(img_path)
                    cols[col_idx].image(img, use_container_width=True, caption=f"Crop {idx+1}")
                except Exception as e:
                    cols[col_idx].error("Error")
        else:
            st.info("No face crops found in this profile directory.")

# Form submission
st.subheader("💡 Narrative input prompt")
user_prompt = st.text_area(
    "Input your video script or storyline idea:",
    value="Bob's restaurant was failing... then he found out his dishwasher was throwing steaks in the dumpster.",
    height=120
)

async def process_scene(scene, face_b64_list, global_style, global_voice, progress_placeholder, temp_dir):
    def update_status(msg):
        progress_placeholder.info(f"🌀 **{scene.id}** ({scene.scene_type}): {msg}")
        
    # Step 1: Synthesize voiceover
    temp_audio = os.path.join(temp_dir, f"audio_{scene.id}.mp3")
    await synthesize_voiceover(scene.voiceover_text, temp_audio, voice_type=global_voice)
    
    # Step 3: Build assembled DP prompt
    assembled_prompt = build_assembled_prompt(scene, global_style)
    
    # Step 4: Dispatch payload to Colab
    groq_api_key = os.getenv("GROQ_API_KEY")
    video_path = await generate_scene_video(
        assembled_prompt,
        scene.camera_movement,
        face_b64_list,
        scene,
        groq_api_key,
        update_status
    )
    return video_path, temp_audio

if st.button("🚀 Trigger Generation Pipeline"):
    if not user_prompt.strip():
        st.error("Please provide a story prompt input.")
        st.stop()

    face_b64_list = []
    if active_profile != "[ None / Mock Simulation ]":
        face_b64_list = load_profile_base64_list(active_profile)
        if not face_b64_list:
            st.warning("Selected profile contains no face crops. Operating in Mock GPU mode.")
    
    temp_dir = tempfile.gettempdir()
    
    # 1. Structured storyboard generation
    with st.spinner("Compiling storyboard DAG structure..."):
        try:
            storyboard = generate_storyboard(user_prompt)
            st.session_state.storyboard = storyboard
            st.success("Storyboard DAG Compiled!")
        except Exception as e:
            st.error(f"Storyboard compilation failed: {e}")
            st.stop()
            
    # Present Storyboard DAG Details
    with st.expander("👁️ View Compiled Storyboard Scenes Details", expanded=True):
        st.write(f"**Global Aesthetic Style:** {storyboard.global_style}")
        st.write(f"**Global Voice Selection:** {storyboard.global_voice}")
        st.json(storyboard.model_dump())

    # 2. Sequential Generation Execution
    st.subheader("⚡ Real-Time Scene Generation Process")
    progress_placeholders = {}
    for scene in storyboard.scenes:
        progress_placeholders[scene.id] = st.empty()
        progress_placeholders[scene.id].info(f"⌛ **{scene.id}** ({scene.scene_type}): Queueing...")

    # Grid container for live previews
    st.subheader("🎥 Live Scene Previews")
    preview_grid_cols = st.columns(2)
    preview_placeholders = {}
    for idx, scene in enumerate(storyboard.scenes):
        col_idx = idx % 2
        with preview_grid_cols[col_idx]:
            st.write(f"**Scene {idx+1} ({scene.scene_type})**")
            preview_placeholders[scene.id] = st.empty()
            preview_placeholders[scene.id].info("Waiting for generation...")

    async def run_sequential_pipeline():
        video_clips = []
        audio_clips = []
        for idx, scene in enumerate(storyboard.scenes):
            progress_placeholder = progress_placeholders[scene.id]
            progress_placeholder.info(f"🌀 **{scene.id}** ({scene.scene_type}): Starting...")
            
            vid_path, aud_path = await process_scene(
                scene, face_b64_list, storyboard.global_style, storyboard.global_voice, progress_placeholder, temp_dir
            )
            
            video_clips.append(vid_path)
            audio_clips.append(aud_path)
            
            # Render video preview immediately in the UI
            with preview_placeholders[scene.id].container():
                if os.path.exists(vid_path):
                    st.video(vid_path)
                st.write(f"*{scene.voiceover_text}*")
                st.caption(f"Synthesized voice profile: `{storyboard.global_voice}`")
                
        return video_clips, audio_clips

    with st.spinner("Orchestrating scene worker pipeline..."):
        try:
            video_clips, audio_clips = asyncio.run(run_sequential_pipeline())
            voiceovers_text = [scene.voiceover_text for scene in storyboard.scenes]
            
            # 3. Stitch and Render Output video file
            st.info("Splicing scenes together with voiceover tracks...")
            final_video_path = os.path.join(temp_dir, "final_compiled_output.mp4")
            
            compile_final_video(video_clips, audio_clips, voiceovers_text, final_video_path)
            
            st.success("Video Render Complete!")
            
            # Display output delivery card
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("🎉 **Your 9:16 Vertical Video is ready for download!**")
                st.write(f"All generated scenes have been compiled. Voice profile: `{storyboard.global_voice}`.")
                if os.path.exists(final_video_path):
                    with open(final_video_path, "rb") as vf:
                        st.download_button(
                            label="📥 Download Portrait Video (.mp4)",
                            data=vf,
                            file_name="consistent_face_video.mp4",
                            mime="video/mp4"
                        )
            with col2:
                # Preview first frame scene clip as indicator
                if len(video_clips) > 0 and os.path.exists(video_clips[0]):
                    st.write("Preview of Scene 1:")
                    st.video(video_clips[0])
                    
        except Exception as e:
            st.error(f"Pipeline execution encountered an error: {e}")
            raise e
