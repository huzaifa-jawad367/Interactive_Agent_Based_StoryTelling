# tools.py

from typing import Dict, Any, List
from smolagents import tool, Tool, InferenceClientModel
from typing import Optional
import json
import os


# Make sure your HF token is set in the environment already:
#   export HUGGINGFACE_API_TOKEN="hf_YourTokenHere"
HF_TOKEN = os.getenv("REDACTED", "")
if not HF_TOKEN:
    raise RuntimeError("Please set HUGGINGFACE_API_TOKEN in your environment.")

# === Choose your HF model name here ===
# For instance, "gpt2‐hf‐chat", or any chat‐capable endpoint. 
# If you are using a locally‐deployed endpoint, point to its URL:
#    model_name = "https://api-inference.huggingface.co/models/your-username/your-chat-model"
# If you want to use an HF‐hosted chat LLM (e.g. a fine‐tuned Llama 2), use:
#    model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Lazily instantiate a single InferenceClientModel
_hf_client: Optional[InferenceClientModel] = None

def _get_hf_client() -> InferenceClientModel:
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClientModel(
            model_name=model_name,
            api_token=HF_TOKEN
        )
    return _hf_client

class StoryGeneratorTool(Tool):
    """
    A single tool that handles both:
    1) The very first user prompt (initial prompt → generate opening scene), and
      2) All subsequent steps (last_choice + context → generate next scene).
      
    INPUTS:
      - initial_prompt (optional): 
          The user's very first textual prompt (“There is a girl who walked…”). 
          Exactly one of initial_prompt or last_choice must be provided per call.
      - last_choice (optional): 
          The user’s selected choice from the previous step. 
      - context (required): 
          The concatenated last N scenes + facts (via StoryState.get_context_window).
          
    OUTPUT:
      - A single string: the newly generated “scene text” from the LLM.
    """

    name = "story_generator"
    description = """
    Generates the next paragraph (scene) of the interactive story. 
    If initial_prompt is provided, uses that as the story’s seed. 
    Otherwise, uses last_choice + context (recent scenes/facts) to continue.
    """

    inputs = {
        "initial_prompt": {
            "type": "string",
            "description": "The very first user prompt to start the story. Mutually exclusive with last_choice.",
            "required": False,
        },
        "last_choice": {
            "type": "string",
            "description": "The user’s choice from the previous step. Mutually exclusive with initial_prompt.",
            "required": False,
        },
        "context": {
            "type": "string",
            "description": (
                "Concatenated recent scenes + facts (via StoryState.get_context_window). "
                "If this is the first prompt, context can be an empty string."
            ),
            "required": True,
        },
    }

    output_type = "string"

    # We’ll lazily create a single InferenceClientModel instance and reuse it.
    _hf_client: Optional[InferenceClientModel] = None

    def _get_hf_client(self) -> InferenceClientModel:
        """
        Return a singleton InferenceClientModel, authenticated via HUGGINGFACE_API_TOKEN.
        """
        if StoryGeneratorTool._hf_client is None:
            StoryGeneratorTool._hf_client = InferenceClientModel(
                model_name=model_name,
                api_token=HF_TOKEN
            )
        return StoryGeneratorTool._hf_client

    def forward(
        self,
        initial_prompt: Optional[str] = None,
        last_choice: Optional[str] = None,
        context: str = "",
    ) -> str:
        """
        Build and send a dynamic LLM prompt depending on whether we're starting fresh
        (initial_prompt != None) or continuing (last_choice != None). Returns the new scene.
        """
        # 1) Both provided → error
        if initial_prompt is not None and last_choice is not None:
            raise ValueError(
                "Provide exactly one of `initial_prompt` or `last_choice`, not both."
            )

        # 1) If this is the very first call, we only have initial_prompt:
        if initial_prompt is not None:
            system_msg = {
                "role": "system",
                "content": (
                    "You are a creative, children’s‐book style storyteller. "
                    "The user has provided the very first seed of the story. "
                    "Generate a vivid opening scene."
                ),
            }
            user_content = f"User seed prompt:\n\"{initial_prompt}\"\n\nGenerate the opening scene."
            # We ignore `context` on the first call (it can be an empty string).
        elif last_choice is not None:
            # 2) We are in a follow‐up step (we have last_choice and some context):
            system_msg = {
                "role": "system",
                "content": (
                    "You are a creative, children’s‐book style storyteller. "
                    "Continue the story from the last choice, integrating the provided context."
                ),
            }
            user_content = (
                f"Context (recent scenes + facts):\n{context}\n\n"
                f"Last choice: \"{last_choice}\"\n\n"
                "Generate the next scene."
            )

        else:
            system_msg = (
                "You are a creative, children’s‐book style storyteller. "
                "Continue the story based on the given context alone."
            )
            user_content = f"Context (recent scenes + facts):\n{context}\n\nGenerate the next scene."

        messages = [
          {"role": "system", "content": system_msg["content"]},
          {"role": "user", "content": user_content},
        ]


        # 3) Call the HF chat model via InferenceClientModel:
        hf_client = self._get_hf_client()
        resp = hf_client.chat(messages=messages, temperature=0.7, max_tokens=400)

        # The HF inference endpoint usually returns a dict with {"generated_text": "..."} or
        # a list of completions. SmolAgents’ InferenceClientModel.chat() will normalize that
        # and just return the assistant’s reply string.
        scene_text = resp  # smolagents normalizes to a simple string

        return scene_text
    

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



@tool
def generate_choices(scene_text: str, facts: Dict[str, Any]) -> List[str]:
    """
    Generate between 2 and 4 next-step choices for the reader,
    based on the current scene_text and known facts.

    Inputs:
      - scene_text: The latest narrative paragraph.
      - facts: A dict of structured facts (e.g., location, weather, npc_states, etc.)

    Output:
      - A list of 2–4 short choice strings.
    """

    # Serialize facts to JSON for inclusion in the prompt
    facts_json = json.dumps(facts, indent=2)

    prompt = f"""
                You are an interactive‐story choice generator. 
                Given the following scene and the known facts, propose between 2 and 4 plausible next‐step choices.
                Return your answer as a JSON array of strings (e.g., ["Choice 1", "Choice 2", "Choice 3"]).

                Scene:
                \"\"\"
                {scene_text}
                \"\"\"

                Facts (JSON):
                {facts_json}

                Requirements:
                  - Provide at least 2 choices, and at most 4 choices.
                  - Each choice should be a concise action or decision (no more than one sentence each).
                  - Do not include any extra commentary—only output the JSON array.
              """

    system_msg = {
        "role": "system",
        "content": "You generate short, actionable choices for an interactive story."
    }
    user_msg = {"role": "user", "content": prompt}

    # Call the HF chat model via InferenceClientModel
    hf_client = _get_hf_client()
    resp = hf_client.chat(
        messages=[system_msg, user_msg],
        temperature=0.8,
        max_tokens=150
    )

    # The response should be a JSON array of strings
    raw = resp.strip()
    try:
        choices = json.loads(raw)
        # Ensure it's a list of strings and has between 2 and 4 elements
        if (
            isinstance(choices, list)
            and 2 <= len(choices) <= 4
            and all(isinstance(c, str) for c in choices)
        ):
            return choices
        # If the model returned something malformed or wrong length, fall back
        raise ValueError
    
    # TODO: handle specific parsing errors more gracefully
    except Exception:
        # Fallback: return two generic choices
        return ["Continue forward", "Turn back"]

@tool
def build_world(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a structured `facts` dictionary (e.g., containing keys like 'location',
    'weather', 'time_of_day', 'npc_states', etc.), return a detailed world-building
    dictionary with:
      - setting_description: a paragraph describing the environment vividly
      - flora: a list of typical plants or vegetation present
      - fauna: a list of common animals or creatures in this setting
      - ambiance: sensory details (sound, smell, feel) to enrich the scene
    """
    # Serialize facts into JSON for the prompt
    facts_json = json.dumps(facts, indent=2)

    prompt = f"""
                You are a world-building assistant. Given the following structured facts in JSON:

                {facts_json}

                Generate a JSON object with these fields:
                  1) setting_description: a 2-3 sentence vivid paragraph describing the environment.
                  2) flora: a list of 3-5 plant species or vegetation commonly found in this location.
                  3) fauna: a list of 3-5 animals or creatures one might encounter here.
                  4) ambiance: a list of 3-5 sensory details (sounds, smells, tactile sensations) that bring the scene to life.

                Return ONLY valid JSON with exactly those four keys. If a particular field is not applicable,
                you may return an empty list for flora/fauna/ambiance, but always include all four keys.
              """

    system_msg = {
        "role": "system",
        "content": "You convert structured facts into a detailed world-building dictionary."
    }
    user_msg = {"role": "user", "content": prompt}

    hf_client = _get_hf_client()
    resp = hf_client.chat(
        messages=[system_msg, user_msg],
        temperature=0.7,
        max_tokens=300
    )
    raw = resp.strip()

    try:
        world_dict = json.loads(raw)
    except Exception:
        # Fallback to a minimal structure using only the basic facts
        world_dict = {
            "setting_description": f"A {facts.get('weather', 'calm')} scene in the {facts.get('location', 'unknown location')}.",
            "flora": [],
            "fauna": [],
            "ambiance": []
        }

    # Ensure all required keys exist
    defaults = {
        "setting_description": "",
        "flora": [],
        "fauna": [],
        "ambiance": []
    }
    for key, default_val in defaults.items():
        if key not in world_dict:
            world_dict[key] = default_val

    return world_dict

@tool
def validate_consistency(old_facts: Dict[str, Any], new_facts: Dict[str, Any]) -> bool:
    """
    Validate that the new_facts do not contradict the old_facts.
    For each core key ("location", "weather", "time_of_day"):
      - If old_facts[key] is not None and new_facts[key] is not None
        and they differ, return False (inconsistent).
    Otherwise, return True.
    """
    core_keys = ["location", "weather", "time_of_day"]

    for key in core_keys:
        old_val = old_facts.get(key)
        new_val = new_facts.get(key)
        if old_val is not None and new_val is not None and old_val != new_val:
            return False

    return True