# tools.py

from typing import Dict, Any, List
from smolagents import tool, Tool, InferenceClientModel
from typing import Optional
import json
import os

# Make sure your HF token is set in the environment already:
#   export HUGGINGFACE_API_TOKEN="Enter your hf token"
HF_TOKEN = os.getenv("Enter your hf token", "")
if not HF_TOKEN:
    raise RuntimeError("Please set HUGGINGFACE_API_TOKEN in your environment.")

# === Choose your HF model name here ===
# For instance, "gpt2‐hf‐chat", or any chat‐capable endpoint. 
# If you are using a locally‐deployed endpoint, point to its URL:
#    model_name = "https://api-inference.huggingface.co/models/your-username/your-chat-model"
# If you want to use an HF‐hosted chat LLM (e.g. a fine‐tuned Llama 2), use:
#    model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-2-7b-chat-hf"

class ExtractFactsTool(Tool):
    """
    Extracts structured facts from a given scene text. Returns a dictionary with:
      - location (string or null)
      - weather (string or null)
      - time_of_day (string or null)
      - main_character (string or null)
      - npc_states (dict mapping NPC name -> {status, location})
      - inventory_items (list of strings)
      - events (1-2 sentence open-ended summary of what happened)
    """

    name = "extract_facts"
    description = """
    Given a scene paragraph, extract a detailed JSON-formatted fact base including:
      • location
      • weather
      • time_of_day
      • main_character
      • npc_states (map of {name: {status, location}})
      • inventory_items (list of strings)
      • events (an open-ended 1-2 sentence summary of key happenings)
    """

    inputs = {
        "scene_text": {
            "type": "string",
            "description": "The narrative paragraph from which to extract facts.",
            "required": True,
        }
    }
    output_type = "dict"

    _hf_client: InferenceClientModel = None

    def _get_hf_client(self) -> InferenceClientModel:
        """
        Lazy-instantiates a single InferenceClientModel for the HF chat endpoint.
        """
        if ExtractFactsTool._hf_client is None:
            ExtractFactsTool._hf_client = InferenceClientModel(
                model_name=model_name,
                api_token=HF_TOKEN
            )
        return ExtractFactsTool._hf_client

    def forward(self, scene_text: str) -> Dict[str, Any]:
        prompt = f"""
          You are a fact-extraction assistant. Below is a single scene from an interactive story:

          \"\"\"
          {scene_text}
          \"\"\"

          Extract exactly the following fields and output valid JSON:
            1) location: (e.g. "rainy_forest" or null if not mentioned)
            2) weather: (e.g. "rainy", "sunny", or null)
            3) time_of_day: (e.g. "morning", "dusk", or null)
            4) main_character: (name of the protagonist, or null if not obvious)
            5) npc_states: a dictionary mapping any other named characters to their status and location.
              Example:
                "npc_states": {
                  "Grandma": {"status": "waiting", "location": "cottage"},
                  "Wolf": {"status": "lurking", "location": "forest_edge"}
                }
              If no NPCs are present, return an empty object `{{}}`.
            6) inventory_items: a list of any items the main character now has (e.g., ["lantern", "map"]), or [] if none.
            7) events: a 1-2 sentence open-ended summary of what key happened events occurred in this scene.

          Do NOT output any keys other than the seven listed above.  
          If a field is not mentioned, set it to `null` (for strings) or `[]`/`{{}}` as appropriate.
        """

        # Build chat messages
        system_msg = {
            "role": "system",
            "content": "You extract structured facts from a story scene and return valid JSON."
        }
        user_msg = {"role": "user", "content": prompt}

        # Call HF chat model via InferenceClientModel
        hf_client = self._get_hf_client()
        resp = hf_client.chat(
            messages=[system_msg, user_msg],
            temperature=0.0,
            max_tokens=400
        )

        # `resp` should be a single JSON string (or a plain dict if the HF endpoint returns JSON directly)
        json_str = resp.strip()

        # Safely parse the JSON; on failure, return empty/default schema
        try:
            fact_dict = json.loads(json_str)
        except Exception:
            fact_dict = {
                "location": None,
                "weather": None,
                "time_of_day": None,
                "main_character": None,
                "npc_states": {},
                "inventory_items": [],
                "events": ""
            }

        # Ensure all keys exist (fill missing with defaults)
        defaults = {
            "location": None,
            "weather": None,
            "time_of_day": None,
            "main_character": None,
            "npc_states": {},
            "inventory_items": [],
            "events": ""
        }
        for key, default_val in defaults.items():
            if key not in fact_dict:
                fact_dict[key] = default_val

        return fact_dict