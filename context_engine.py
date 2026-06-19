import os
from dotenv import load_dotenv
load_dotenv(override=True)
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

class StorySceneNode(BaseModel):
    id: str = Field(description="Scene ID, e.g. scene_1, scene_2")
    scene_type: str = Field(description="hook | build-up | climax | resolution | payoff")
    visual_prompt: str = Field(description="Highly descriptive prompt detailing the scene, character actions, and environment")
    lighting: str = Field(description="Cinematic lighting direction, e.g., volumetric god rays, single key light, warm candlelight")
    camera_angle: str = Field(description="Framing/angle, e.g., low-angle extreme close-up, high-angle establishing shot")
    camera_movement: str = Field(description="Camera movement instruction, e.g., slow dolly forward, panning left, zoom-in, static")
    lens: str = Field(description="Camera lens specs, e.g., 35mm cinematic, 85mm anamorphic, 16mm wide-angle")
    color_palette: str = Field(description="Color grading profile, e.g., desaturated teal and orange, vintage Kodachrome, monochrome high-contrast")
    mood: str = Field(description="Emotional mood, e.g., tense, melancholic, suspenseful, nostalgic")
    voiceover_text: str = Field(description="Clean narration sentence matching the visual action, max 15 words")
    reference_search_query: str = Field(description="A search query to fetch a matching style/setting background reference photo from DDG, e.g., 'dark restaurant kitchen at night moody'")
    needs_face: bool = Field(default=False, description="True if this scene shows the main character's face/body. False for establishing shots, objects, environments.")
    search_queries: List[str] = Field(default_factory=list, description="3 diverse DDG image search queries to find visual reference photos for this scene")

class VideoDAGPayload(BaseModel):
    global_style: str = Field(description="Global styling context, e.g., 90s cinematic film noir, volumetric lighting, high photorealism")
    global_voice: str = Field(default="male_narrator", description="Voice character: male_narrator | female_narrator | child | deep_voice | British_male | British_female")
    scenes: List[StorySceneNode]

def build_assembled_prompt(scene: StorySceneNode, global_style: str) -> str:
    """Assembles all the micro-attributes into a production-ready diffusion prompt."""
    return (
        f"{scene.visual_prompt}. Lighting: {scene.lighting}. "
        f"Shot framing: {scene.camera_angle}, shot on {scene.lens} lens. "
        f"Color grading: {scene.color_palette}. Mood: {scene.mood} mood. "
        f"Aesthetic global style: {global_style}."
    )

def refine_prompt(prompt: str) -> str:
    """
    Uses Cerebras Llama-3.3-70b to expand a simple user story prompt into a highly detailed,
    cinematic visual narrative with rich descriptions, key framing, lighting instructions,
    and a cohesive scene-by-scene storyline.
    """
    api_key = os.getenv("CEREBRAS_API_KEY", "csk-n94j36ew6vp5p3538kwvpnvpyj8tvvrvdvnc2hwthh25fhmk").strip()
    if not api_key:
        print("[Cerebras] No API key configured. Skipping prompt refinement.")
        return prompt
        
    print(f"[Cerebras] Refining prompt: '{prompt[:60]}...'")
    import requests
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    refinement_instruction = (
        "You are an expert Hollywood screenwriter and visual designer. "
        "Your task is to take a raw, simple story idea or script input and refine it into a highly detailed, "
        "expanded cinematic script suitable for generating visual storyboards. "
        "Expand the characters, environments, actions, lighting suggestions, and dramatic structure. "
        "Analyze the user's input for any specific requested art style, medium, or visual aesthetic "
        "(e.g., 3D animation, Pixar style, anime, pencil sketch, oil painting, photorealism) and you MUST "
        "explicitly maintain, emphasize, and describe this requested art style in the refined screenplay. "
        "Format the output as a clean, cohesive 10-paragraph narrative that tells the story sequentially. "
        "Do not include any intro, outro, or meta-commentary. Output the refined story text only."
    )
    
    payload = {
        "model": "gpt-oss-120b",
        "messages": [
            {"role": "system", "content": refinement_instruction},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    try:
        r = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )
        r.raise_for_status()
        result = r.json()
        refined_story = result["choices"][0]["message"]["content"].strip()
        print(f"[Cerebras] Prompt successfully refined. Size: {len(refined_story)} chars")
        return refined_story
    except Exception as e:
        print(f"[Cerebras] Refinement failed: {e}. Using original prompt.")
        return prompt

def generate_storyboard(prompt: str) -> VideoDAGPayload:
    # 1. Refine the raw prompt into a cinematic visual narrative using Cerebras
    prompt = refine_prompt(prompt)
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Define a helper function to compile a fallback using DuckDuckGo Free Llama-3 API
    def generate_free_llama_storyboard(prompt_text: str) -> VideoDAGPayload:
        print("Querying free DuckDuckGo Llama-3 API to compile storyboard...")
        from duckduckgo_search import DDGS
        import json
        
        system_instruction = (
            "You are an expert short-form viral video producer and cinematographer. Split the user story into "
            "exactly 10 storyboard scenes. Detect any specific visual art style requested by the user (e.g., 3D animation, cartoon, anime, sketch, photorealistic). "
            "Set the global_style field to exactly match this style (do not just copy the cinematic film noir example unless requested), and ensure all individual scene visual_prompts are written using terminology matching that style (for example, if style is 3D animation, describe elements as 'Pixar-style 3D rendered character model, stylized features'). "
            "You must respond ONLY with a raw JSON object matching the following Pydantic schema structure:\n"
            "{\n"
            "  \"global_style\": \"cinematic film noir, volumetric lighting, high photorealism (REPLACE WITH USER STYLE)\",\n"
            "  \"global_voice\": \"male_narrator | female_narrator | child | deep_voice | British_male | British_female\",\n"
            "  \"scenes\": [\n"
            "    {\n"
            "      \"id\": \"scene_1\",\n"
            "      \"scene_type\": \"hook\",\n"
            "      \"visual_prompt\": \"Description of scene matching the requested style\",\n"
            "      \"lighting\": \"Lighting description\",\n"
            "      \"camera_angle\": \"Camera angle\",\n"
            "      \"camera_movement\": \"Camera movement\",\n"
            "      \"lens\": \"Lens description\",\n"
            "      \"color_palette\": \"Color grade profile\",\n"
            "      \"mood\": \"Emotional mood\",\n"
            "      \"voiceover_text\": \"Voiceover caption sentence under 15 words\",\n"
            "      \"reference_search_query\": \"Short image search query matching the requested style\",\n"
            "      \"needs_face\": true,\n"
            "      \"search_queries\": [\"query 1\", \"query 2\", \"query 3\"]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "For needs_face, use true if the main character's face/body is in the scene, otherwise false. For search_queries, provide 3 diverse search queries to find visual reference photos matching the requested style. Do not include markdown wrappers (like ```json), commentary, or extra text. Output raw JSON only."
        )
        
        try:
            with DDGS() as ddgs:
                full_prompt = f"{system_instruction}\n\nUser Story: {prompt_text}"
                # Requesting DuckDuckGo Chat API (returns text responses using Llama-3)
                results = ddgs.chat(keywords=full_prompt, model="llama")
                
                # Sanitize response
                raw_text = results.strip()
                if raw_text.startswith("```"):
                    lines = raw_text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    raw_text = "\n".join(lines).strip()
                
                data = json.loads(raw_text)
                return VideoDAGPayload.model_validate(data)
        except Exception as e:
            print(f"DuckDuckGo Llama API failed: {e}. Falling back to default hardcoded storyboard.")
            # Return our high-quality default static storyboard to prevent crash
            return VideoDAGPayload(
                global_style="90s cinematic film noir, volumetric lighting, high photorealism",
                scenes=[
                    StorySceneNode(
                        id="scene_1",
                        scene_type="hook",
                        visual_prompt="A detective standing under a neon streetlamp looking down a rainy brick alley",
                        lighting="High-contrast neon blue and pink rim lighting, soft volumetric streetlamp glow",
                        camera_angle="Low-angle medium shot looking up from wet pavement",
                        camera_movement="Slow push-in toward the alleyway",
                        lens="35mm cinematic anamorphic",
                        color_palette="Deep teal shadows with vibrant pink highlights",
                        mood="suspenseful",
                        voiceover_text="It was a cold night in New York, the kind that washes away memories.",
                        reference_search_query="neon streetlamp rain brick alley night",
                        needs_face=False,
                        search_queries=["neon streetlamp rain brick alley night", "film noir rainy street pink neon", "cyberpunk alley wet pavement lights"]
                    ),
                    StorySceneNode(
                        id="scene_2",
                        scene_type="build-up",
                        visual_prompt="A dishwasher secretly throwing fresh prime rib steaks into a rusty metal dumpster behind a restaurant kitchen",
                        lighting="Harsh overhead security spotlight casting strong dark shadows, dim warm light from the open back door",
                        camera_angle="Eye-level medium shot from behind the dumpster",
                        camera_movement="Subtle handheld camera shake",
                        lens="50mm prime",
                        color_palette="Gritty industrial greens and shadow black",
                        mood="tense",
                        voiceover_text="Bob's restaurant was failing... because his dishwasher was throwing premium steaks in the dumpster.",
                        reference_search_query="rusty dumpster back alley restaurant night",
                        needs_face=False,
                        search_queries=["rusty dumpster back alley restaurant night", "commercial kitchen back door dark spotlight", "restaurant waste dumpster gritty urban"]
                    ),
                    StorySceneNode(
                        id="scene_3",
                        scene_type="build-up",
                        visual_prompt="A restaurant owner staring in shock at an empty cash drawer inside a dark wood-paneled diner",
                        lighting="Single overhead practical hanging lamp casting direct bright light down on the empty register",
                        camera_angle="Extreme close-up on the owner's stressed face showing sweat",
                        camera_movement="Slow zoom-in on the eyes",
                        lens="85mm portrait",
                        color_palette="Warm saturated amber tones with heavy grain",
                        mood="shocked",
                        voiceover_text="He was bleeding cash, blind to the theft happening right behind his back.",
                        reference_search_query="old cash register open empty diner",
                        needs_face=True,
                        search_queries=["old cash register open empty diner", "stressed restaurant owner face close up", "vintage diner hanging lamp amber mood"]
                    ),
                    StorySceneNode(
                        id="scene_4",
                        scene_type="climax",
                        visual_prompt="Close up of an antique golden pocket watch ticking slowly on a dark mahogany desk",
                        lighting="Soft side window light, light rays visible through dust motes",
                        camera_angle="Macro close-up shot",
                        camera_movement="Static locked tripod",
                        lens="100mm macro",
                        color_palette="Warm sepia tones and golden highlights",
                        mood="melancholic",
                        voiceover_text="Time was running out. Every tick felt like a step closer to the edge.",
                        reference_search_query="vintage gold pocket watch ticking on desk",
                        needs_face=False,
                        search_queries=["vintage gold pocket watch ticking on desk", "antique clock mahogany desk sepia", "golden timepiece dust motes sunlight macro"]
                    ),
                    StorySceneNode(
                        id="scene_5",
                        scene_type="climax",
                        visual_prompt="A glowing blue hard drive sitting on a brushed metal workstation desk inside a server room",
                        lighting="Cool server rack LED lights flashing, misty atmosphere",
                        camera_angle="High-angle wide shot looking down at the desk",
                        camera_movement="Slow tilt upwards",
                        lens="24mm wide-angle",
                        color_palette="Cool neon cyan and silver metallic",
                        mood="mysterious",
                        voiceover_text="But the truth would finally emerge, and it would rewrite everything.",
                        reference_search_query="server rack blue flashing lights room",
                        needs_face=False,
                        search_queries=["server rack blue flashing lights room", "data center neon cyan glow mist", "hard drive workstation metallic cyberpunk"]
                    ),
                    StorySceneNode(
                        id="scene_6",
                        scene_type="resolution",
                        visual_prompt="A stack of confidential files labeled CONFIDENTIAL on a wooden table, highlighted by flashlight",
                        lighting="Sharp focus beam from a flashlight cutting across the darkness",
                        camera_angle="High-angle closeup shot",
                        camera_movement="static",
                        lens="50mm anamorphic",
                        color_palette="Desaturated grey and stark yellow highlight",
                        mood="tense",
                        voiceover_text="Behind closed doors, the pieces of the puzzle were starting to connect.",
                        reference_search_query="confidential files under flashlight dark table",
                        needs_face=False,
                        search_queries=["confidential files under flashlight dark table", "secret documents spotlight noir", "manila folder wooden desk mystery"]
                    ),
                    StorySceneNode(
                        id="scene_7",
                        scene_type="resolution",
                        visual_prompt="The detective standing at a large glass window overlooking a rainy nighttime metropolis city skyline",
                        lighting="Distant city light reflections on glass, cool blue moonlight ambient",
                        camera_angle="Wide silhouette shot from inside the room",
                        camera_movement="slow dolly backward",
                        lens="24mm cinematic wide",
                        color_palette="Deep indigo and city gold reflections",
                        mood="reflective",
                        voiceover_text="Every city has its secrets, but some secrets refuse to remain buried.",
                        reference_search_query="rain city window night silhouette detective",
                        needs_face=True,
                        search_queries=["rain city window night silhouette detective", "man silhouette glass window city skyline", "noir detective overlooking metropolis indigo"]
                    ),
                    StorySceneNode(
                        id="scene_8",
                        scene_type="payoff",
                        visual_prompt="A secure server rack vault door slowly unlocking with green indicator LEDs turning on",
                        lighting="Vibrant green status LEDs illuminating the dark vault entryway",
                        camera_angle="Low-angle establishing shot",
                        camera_movement="slow tilt up",
                        lens="35mm cinematic",
                        color_palette="High-contrast dark and emerald green",
                        mood="triumphant",
                        voiceover_text="When the lock clicked open, the hidden archive lay fully exposed.",
                        reference_search_query="vault server door green led unlocking",
                        needs_face=False,
                        search_queries=["vault server door green led unlocking", "secure data center vault emerald glow", "server room door opening green lights"]
                    ),
                    StorySceneNode(
                        id="scene_9",
                        scene_type="payoff",
                        visual_prompt="A high quality portrait of the detective finally smiling in a brightly lit coffee shop",
                        lighting="Soft warm morning sun filling the room, cinematic bokeh",
                        camera_angle="Medium portrait shot",
                        camera_movement="static",
                        lens="85mm prime",
                        color_palette="Warm golden hour tones",
                        mood="hopeful",
                        voiceover_text="In the morning light, the truth brought a quiet sense of relief.",
                        reference_search_query="diner portrait man smiling warm morning light",
                        needs_face=True,
                        search_queries=["diner portrait man smiling warm morning light", "coffee shop golden hour warm portrait", "happy man cafe morning sunlight bokeh"]
                    ),
                    StorySceneNode(
                        id="scene_10",
                        scene_type="payoff",
                        visual_prompt="A cinematic view of the city streets at dawn, traffic moving smoothly as fog rises",
                        lighting="Soft dawn golden glow, long silhouettes",
                        camera_angle="Wide aerial shot",
                        camera_movement="slow zoom out",
                        lens="50mm wide-angle",
                        color_palette="Pastel orange and soft blue mist",
                        mood="peaceful",
                        voiceover_text="And just like that, the city woke up to a new beginning.",
                        reference_search_query="city skyline dawn morning mist cinematic",
                        needs_face=False,
                        search_queries=["city skyline dawn morning mist cinematic", "aerial city streets fog sunrise golden", "urban dawn traffic flow pastel sky"]
                    )
                ]
            )
 
    # 1. Check if Groq API key is present
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            return generate_groq_storyboard(prompt, groq_api_key)
        except Exception as e:
            print(f"Groq API request failed ({e}). Falling back to Gemini/Llama...")
 
    # 2. Check if Gemini API key is present
    if not api_key:
        return generate_free_llama_storyboard(prompt)
        
    # 3. Try to query Gemini
    try:
        client = genai.Client(api_key=api_key)
        
        system_instruction = (
            "You are an expert short-form viral video producer and cinematographer. Split the user prompt/story into "
            "exactly 10 storyboard scenes to make a full 60-second video. Analyze the prompt for any specific requested art style, medium, or aesthetic "
            "(e.g., 3D animation, Pixar style, anime, sketch, photorealism). Set the global_style to reflect this exact visual style, and ensure "
            "all individual scene visual_prompts are written using terminology matching that style (for example, if style is 3D animation, describe elements as 'Pixar-style 3D rendered character model, stylized features'). "
            "Keep each visual prompt highly descriptive. Generate detailed cinematic attributes (lighting, lens, camera angle, camera movement, color grading, mood, reference query, needs_face, search_queries) for diffusion model generation. "
            "Choose a global_voice (male_narrator | female_narrator | child | deep_voice | British_male | British_female) appropriate for the story. Keep each voiceover caption clean and under 15 words."
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=VideoDAGPayload,
                temperature=0.75
            )
        )
        return VideoDAGPayload.parse_raw(response.text)
    except Exception as e:
        print(f"Gemini API request failed ({e}). Falling back to free Llama-3 endpoint...")
        return generate_free_llama_storyboard(prompt)
 
def generate_groq_storyboard(prompt_text: str, api_key: str) -> VideoDAGPayload:
    print("Querying Groq API Llama-3.3-70b for storyboard...")
    import requests
    import json
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_instruction = (
        "You are an expert short-form viral video producer and cinematographer. Split the user story into "
        "exactly 10 storyboard scenes. Detect any specific visual art style requested by the user (e.g., 3D animation, cartoon, anime, sketch, photorealistic). "
        "Set the global_style field to exactly match this style (do not just copy the cinematic film noir example unless requested), and ensure all individual scene visual_prompts are written using terminology matching that style (for example, if style is 3D animation, describe elements as 'Pixar-style 3D rendered character model, stylized features'). "
        "You must respond ONLY with a raw JSON object matching the following Pydantic schema structure:\n"
        "{\n"
        "  \"global_style\": \"cinematic film noir, volumetric lighting, high photorealism (REPLACE WITH USER STYLE)\",\n"
        "  \"global_voice\": \"male_narrator | female_narrator | child | deep_voice | British_male | British_female\",\n"
        "  \"scenes\": [\n"
        "    {\n"
        "      \"id\": \"scene_1\",\n"
        "      \"scene_type\": \"hook\",\n"
        "      \"visual_prompt\": \"Description of scene matching the requested style\",\n"
        "      \"lighting\": \"Lighting description\",\n"
        "      \"camera_angle\": \"Camera angle\",\n"
        "      \"camera_movement\": \"Camera movement\",\n"
        "      \"lens\": \"Lens description\",\n"
        "      \"color_palette\": \"Color grade profile\",\n"
        "      \"mood\": \"Emotional mood\",\n"
        "      \"voiceover_text\": \"Voiceover caption sentence under 15 words\",\n"
        "      \"reference_search_query\": \"Short image search query matching the requested style\",\n"
        "      \"needs_face\": true,\n"
        "      \"search_queries\": [\"query 1\", \"query 2\", \"query 3\"]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "For needs_face, use true if the main character's face/body is in the scene, otherwise false. For search_queries, provide 3 diverse search queries to find visual reference photos matching the requested style. Do not include markdown wrappers (like ```json), commentary, or extra text. Output raw JSON only."
    )
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_text}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.7
    }
    
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    r.raise_for_status()
    result = r.json()
    raw_text = result["choices"][0]["message"]["content"]
    data = json.loads(raw_text.strip())
    return VideoDAGPayload.model_validate(data)


