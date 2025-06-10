# Tools/build_world.py

from typing import Dict, Any
from smolagents import tool
import json
import torch
# from llm_utils import tokenizer, model, generate_completion # These are loaded globally elsewhere

@tool
def build_world(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a structured `facts` dictionary, returns a world-building dictionary with keys:
      - setting_description: a vivid 2–3 sentence paragraph describing the environment.
      - flora: a list of 3–5 plant species commonly found here.
      - fauna: a list of 3–5 animals or creatures one might encounter.
      - ambiance: a list of 3–5 sensory details (sounds, smells, tactile sensations).

    Args:
        facts (Dict[str, Any]): The structured facts extracted from the scene.

    Returns:
        Dict[str, Any]: A dictionary with exactly the four keys:
            'setting_description', 'flora', 'fauna', and 'ambiance'.
    """
    # 1) Prepare the JSON-extraction prompt
    facts_json = json.dumps(facts, indent=2)
    prompt = f"""
You are a world-building assistant. Given these structured facts:

{facts_json}

Generate a JSON object with exactly these fields:
  1) setting_description: a 2–3 sentence vivid paragraph describing the environment.
  2) flora: a list of 3–5 plant species commonly found here.
  3) fauna: a list of 3–5 animals or creatures one might encounter.
  4) ambiance: a list of 3–5 sensory details (sounds, smells, tactile feelings).

Return ONLY valid JSON with those four keys.
"""

    # 2) Tokenize & move to device
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You convert structured facts into world-building JSON."},
            {"role": "user",   "content": prompt}
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True # Keep return_dict=True to get attention_mask
    )
    # Move each tensor to the model device
    for k,v in inputs.items():
        inputs[k] = v.to(model.device)

    # 3) Generate up to 256 new tokens via shared generate_completion
    with torch.no_grad():
        # Pass the input_ids tensor directly and the attention_mask
        outputs = model.generate(
            inputs["input_ids"], # Pass the tensor directly
            max_new_tokens=256,
            attention_mask=inputs.get("attention_mask") # Pass attention_mask if present
        )


    # 4) Slice off the prompt tokens
    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids    = outputs[0][prompt_len:]

    # 5) Decode the JSON string
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    start = raw.find("{")
    candidate = raw[start:] if start >= 0 else raw

    # 6) Parse, with a defaults fallback
    defaults = {
        "setting_description": "",
        "flora": [],
        "fauna": [],
        "ambiance": []
    }
    try:
        world_dict = json.loads(candidate)
    except Exception:
        world_dict = defaults.copy()

    # 7) Ensure all keys are present
    for key, val in defaults.items():
        world_dict.setdefault(key, val)

    return world_dict