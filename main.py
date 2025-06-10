# orchestrator.py

import os
import base64
from typing import Optional, Dict, Any
from store_facts import StoryState
from Tools.story_generator import StoryGeneratorTool
from Tools.extract_facts import ExtractFactsTool
from Tools.build_world import build_world
from Tools.generate_choices import generate_choices
from Tools.image_agent import generate_image
from Tools.audio_agent import generate_voice_narration  # your TTS tool

# No consistency or change-checking for now

# Output directories
AUDIO_DIR = "outputs/audio"
IMAGE_DIR = "outputs/images"

# Ensure output dirs exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


def advance_story(
    state: StoryState,
    initial_prompt: Optional[str] = None,
    last_choice: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs one step of the story pipeline without consistency or change checks:
      1. Generate scene
      2. Extract facts
      3. Update state & build world meta
      4. Generate choices
      5. Generate and save audio (TTS)
      6. Generate and save image
      7. Return all data
    """
    # 1) Generate scene
    scene_tool = StoryGeneratorTool()
    scene_args = {
        "context": state.get_context_window(n=3),
        "initial_prompt": initial_prompt,
        "last_choice": last_choice,
    }
    scene_text = scene_tool.forward(**scene_args)

    # 2) Extract facts
    fact_tool = ExtractFactsTool()
    new_facts = fact_tool.forward(scene_text)

    # 3) Update state & build world metadata
    state.update_facts(new_facts)
    state.append_scene(scene_text)
    world_meta = build_world(state.facts)
    state.update_world_meta(world_meta)

    # 4) Generate next-step choices
    choices = generate_choices(scene_text, state.facts)

    # 5) Generate and save audio (TTS)
    tts_tool = generate_voice_narration
    audio_bytes = tts_tool(scene_text)
    audio_path = os.path.join(AUDIO_DIR, f"scene_{len(state.scene_history)}.mp3")
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)
    state.record_audio(audio_path)

    # 6) Generate and save image
    img_bytes = generate_image(world_meta.get("setting_description", ""))
    image_path = os.path.join(IMAGE_DIR, f"scene_{len(state.scene_history)}.png")
    with open(image_path, "wb") as f:
        f.write(img_bytes)
    state.record_image(image_path)

    # 7) Package and return
    return {
        "scene_text":    scene_text,
        "choices":       choices,
        "audio_path":    audio_path,
        "image_path":    image_path,
        "updated_state": state.to_dict(),
    }
