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

def _get_hf_client() -> InferenceClientModel:
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClientModel(
            model_name=model_name,
            api_token=HF_TOKEN
        )
    return _hf_client


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