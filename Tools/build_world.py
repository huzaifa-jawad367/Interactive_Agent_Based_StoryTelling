# tools.py

from typing import Dict, Any
from smolagents import tool
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# ————————————————
# Local Mistral-7B–Instruct model for world-building
WB_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

_tokenizer_wb = AutoTokenizer.from_pretrained(WB_MODEL_NAME)
_model_wb     = AutoModelForCausalLM.from_pretrained(
    WB_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
_model_wb.eval()
# ————————————————

@tool
def build_world(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a structured `facts` dictionary, returns a world-building dictionary with keys:
      - setting_description: a vivid 2–3 sentence paragraph describing the environment.
      - flora: a list of 3–5 plant species commonly found here.
      - fauna: a list of 3–5 animals or creatures one might encounter.
      - ambiance: a list of 3–5 sensory details (sounds, smells, tactile sensations).

    Args:
        facts (Dict[str, Any]): The structured facts extracted from the scene,
            containing fields such as 'location', 'weather', 'time_of_day', 'main_character',
            'npc_states', 'inventory_items', and 'events'.

    Returns:
        Dict[str, Any]: A dictionary with exactly the four keys:
            'setting_description', 'flora', 'fauna', and 'ambiance'.
    """
    # 1) Prepare the JSON-extraction prompt
    facts_json = json.dumps(facts, indent=2)
    prompt = f"""
              You are a world-building assistant. Given these structured facts:

              {facts_json}

              Generate a JSON object exactly with these fields:
                1) setting_description: a 2–3 sentence vivid paragraph describing the environment.
                2) flora: a list of 3–5 plant species commonly found here.
                3) fauna: a list of 3–5 animals or creatures one might encounter.
                4) ambiance: a list of 3–5 sensory details (sounds, smells, tactile feelings).

              Return ONLY valid JSON with those four keys.
              """

    # 2) Tokenize & move to device
    inputs_tensor = _tokenizer_wb.apply_chat_template(
        [{"role": "system", "content": "You convert structured facts into world-building JSON."},
         {"role": "user",   "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(_model_wb.device)

    # 3) Generate up to 256 new tokens
    with torch.no_grad():
        outputs = _model_wb.generate(inputs_tensor, max_new_tokens=256)

    # 4) Slice off the prompt tokens
    input_len = inputs_tensor.size(-1)
    gen_ids   = outputs[0][input_len:]

    # 5) Decode the JSON string
    raw = _tokenizer_wb.decode(gen_ids, skip_special_tokens=True)
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

