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