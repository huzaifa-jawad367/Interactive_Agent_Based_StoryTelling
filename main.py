# orchestrator.py

import base64
from typing import Optional, Dict, Any
from store_facts import StoryState
from Tools.story_generator import StoryGeneratorTool
from Tools.extract_facts import ExtractFactsTool
from Tools.validate_consistency import validate_consistency
from Tools.build_world import build_world
from Tools.generate_choices import generate_choices
from Tools.extract_image import extract_scene
from Tools.imagedecider import check_significant_change
from Tools.image_agent import generate_image
from Tools.audio_agent import generate_voice_narration, generate_story_narration  # your TTS tool

# Maximum retries for consistency and image validation
MAX_FACT_RETRIES = 2
MAX_IMAGE_RETRIES = 2

def advance_story(
    state: StoryState,
    initial_prompt: Optional[str] = None,
    last_choice: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs one full step of the story pipeline:
      1. Generate scene
      2. Extract facts
      3. Validate consistency (retry up to MAX_FACT_RETRIES)
      4. Update state & build world meta
      5. Generate choices
      6. Generate audio (TTS)
      7. Decide & generate image (with consistency retries)
      8. Package and return everything
    """
    scene_tool       = StoryGeneratorTool()
    fact_tool        = ExtractFactsTool()
    consistency_tool = validate_consistency
    world_tool       = build_world
    choices_tool     = generate_choices
    image_decider    = check_significant_change
    image_generator  = generate_image
    tts_tool         = generate_voice_narration
    scene_extraction_tool = extract_scene

    # 1) Generate scene (initial vs. continuation)
    scene_args = {
        "context":   state.get_context_window(n=3),
        "initial_prompt": initial_prompt,
        "last_choice":    last_choice
    }
    scene_text = scene_tool.forward(**scene_args)

    # 2) Extract facts
    new_facts = fact_tool.forward(scene_text)

    # 3) Validate consistency with retry
    retries = 0
    while not consistency_tool(old_facts=state.facts, new_facts=new_facts):
        retries += 1
        if retries > MAX_FACT_RETRIES:
            # Give up and proceed anyway
            break
        # add hint to prompt and regenerate
        scene_args["last_choice"] = last_choice  # keep same inputs
        scene_text = scene_tool.forward(**scene_args)
        new_facts  = fact_tool.forward(scene_text)

    # 4) Update state & build world metadata
    state.update_facts(new_facts)
    state.append_scene(scene_text)
    world_meta = world_tool(facts=state.facts)

    # 5) Generate next‐step choices
    choices = choices_tool(scene_text, state.facts)

    # 6) Generate audio (TTS)
    audio_bytes = tts_tool.forward(scene_text)
    audio_b64   = base64.b64encode(audio_bytes).decode("utf-8")

    # 7) Decide if we need a new image
    image_b64 = None
    if image_decider.forward(old_facts=state.facts, new_facts=new_facts):
        img_retries = 0
        while img_retries <= MAX_IMAGE_RETRIES:
            img_bytes = image_generator.forward(world_meta["setting_description"])
            # Optionally validate image against context
            if extract_scene(context=state.get_context_window(3)):
                image_b64 = base64.b64encode(img_bytes).decode("utf-8")
                break
            img_retries += 1

    # 8) Package and return
    return {
        "scene_text":     scene_text,
        "updated_state":  state.to_dict(),
        "world_meta":     world_meta,
        "choices":        choices,
        "audio_b64":      audio_b64,
        "image_b64":      image_b64
    }

