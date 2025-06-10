# tools.py

from typing import Dict, Any
from smolagents import Tool
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_utils import tokenizer, model, generate_completion
import torch
import os
import json


class ExtractFactsTool(Tool):
    """
    Extracts structured facts from a scene using a local Transformers LLM.
    """

    name        = "extract_facts"
    description = (
        "Given a narrative paragraph, extracts and returns JSON with keys: "
        "location, weather, time_of_day, main_character, npc_states, "
        "inventory_items, events."
    )

    inputs = {
        "scene_text": {
            "type": "string",
            "description": "The narrative paragraph from which to extract facts.",
            "required": True
        }
    }
    # Change output_type from "json" or "dict" to "object"
    output_type = "object"

    def forward(self, scene_text: str) -> Dict[str, Any]:
        # 1) Build the instruction + content prompt
        prompt = f"""
                    You are a fact-extraction assistant. Extract exactly the following keys and output valid JSON:
                    1) location: e.g. "rainy_forest" or null
                    2) weather: e.g. "rainy" or null
                    3) time_of_day: e.g. "evening" or null
                    4) main_character: protagonist name or null
                    5) npc_states: dict of other characters → {{status, location}}, or {{}}
                    6) inventory_items: list of item names, or []
                    7) events: 1–2 sentence summary of what happened

                    Scene:
                    \"\"\"
                    {scene_text}
                    \"\"\"
                  """

        # 2) Tokenize using the chat template
        # The output of apply_chat_template with return_tensors="pt" is a single tensor.
        # It does not need to be converted to a dictionary for model.generate.
        inputs_tensor = tokenizer.apply_chat_template(
            [{"role":"system","content":"You extract JSON facts."},
             {"role":"user","content":prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # 3) Generate up to 256 new tokens
        with torch.no_grad():
            # Pass the tensor directly to generate
            outputs = model.generate(inputs_tensor, max_new_tokens=256)

        # 4) Slice off the prompt tokens
        input_len = inputs_tensor.size(-1) # Use inputs_tensor to get the original input length
        gen_ids   = outputs[0][input_len:]

        # 5) Decode and strip out anything before the first '{'
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        json_start = raw.find("{")
        candidate = raw[json_start:] if json_start >= 0 else raw

        # 6) Parse JSON (fallback to defaults on error)
        defaults = {
            "location": None,
            "weather": None,
            "time_of_day": None,
            "main_character": None,
            "npc_states": {},
            "inventory_items": [],
            "events": ""
        }
        try:
            fact_dict = json.loads(candidate)
        except Exception:
            fact_dict = defaults.copy()

        # 7) Ensure all required keys exist
        for k, v in defaults.items():
            fact_dict.setdefault(k, v)

        return fact_dict