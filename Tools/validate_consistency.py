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