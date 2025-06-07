# tools.py

from typing import Dict, Any, List
from smolagents import tool
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# ——————————————————————————————
# Local Mistral-7B–Instruct for choice generation
CHOICE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
_tokenizer_ch = AutoTokenizer.from_pretrained(CHOICE_MODEL)
_model_ch     = AutoModelForCausalLM.from_pretrained(
    CHOICE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
_model_ch.eval()
# ——————————————————————————————

@tool
def generate_choices(scene_text: str, facts: Dict[str, Any]) -> List[str]:
    """
    Generate 2–4 next-step choices for the reader based on the scene and facts.

    Args:
        scene_text (str): The latest narrative paragraph.
        facts (Dict[str,Any]): Structured facts (location, weather, npc_states, etc.)

    Returns:
        List[str]: A list of between 2 and 4 short choice strings.
    """
    facts_json = json.dumps(facts, indent=2)
    prompt = f"""
You are an interactive-story choice generator. Given the scene and known facts below,
propose between 2 and 4 plausible next-step choices. Return *only* a JSON array of strings.

Scene:
\"\"\"
{scene_text}
\"\"\"

Facts:
{facts_json}

Requirements:
- 2 to 4 concise, actionable choices (max one sentence each).
- No extra commentary—just the JSON list.
"""

    # wrap in a chat template
    messages = [
        {"role": "system", "content": "You produce JSON arrays of story choices."},
        {"role": "user",   "content": prompt}
    ]

    # tokenize & move to device
    # Ensure apply_chat_template returns a dictionary
    inputs = _tokenizer_ch.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True # Explicitly request dictionary output
    ).to(_model_ch.device)


    # generate
    with torch.no_grad():
        # Pass the dictionary containing input_ids and attention_mask
        outputs = _model_ch.generate(**inputs, max_new_tokens=128)


    # slice off prompt
    # Access input_ids from the dictionary for prompt length
    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids    = outputs[0][prompt_len:]

    # decode, find JSON
    raw = _tokenizer_ch.decode(gen_ids, skip_special_tokens=True)
    start = raw.find("[")
    candidate = raw[start:] if start >= 0 else raw

    # parse JSON, fallback
    try:
        choices = json.loads(candidate)
        if (
            isinstance(choices, list)
            and 2 <= len(choices) <= 4
            and all(isinstance(c, str) for c in choices)
        ):
            return choices
    except:
        pass

    # fallback
    return ["Continue forward", "Turn back"]