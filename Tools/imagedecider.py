import json
from typing import Any, Dict

from your_tool_base import Tool  # Replace with your actual Tool base class
from your_inference_client import InferenceClientModel  # Replace with your actual HF client import

model_name = "meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = "your_hf_token_here"  # Ensure your HF token is set in environment or replaced here

class CheckSignificantChangeTool(Tool):
    """
    Compares two context windows (previous and current) and returns:
      - 1 if there is a significant change (e.g., a new major scene or environment change)
      - 0 otherwise.
    """

    name = "check_significant_change"
    description = """
    Given the previous context window and the current context window (each containing the last N scenes
    and current facts as JSON), determine if a major or visually important scene change has occurred.
    Return exactly 1 if there is a significant change (new environment, major plot shift) or 0 if not.
    """

    inputs = {
        "previous_context": {
            "type": "string",
            "description": "The previous context window (last N scenes + facts).",
            "required": True,
        },
        "current_context": {
            "type": "string",
            "description": "The current context window (last N scenes + facts).",
            "required": True,
        }
    }
    output_type = "int"  # 1 or 0

    _hf_client: InferenceClientModel = None

    def _get_hf_client(self) -> InferenceClientModel:
        """
        Lazy-instantiates a single InferenceClientModel for the HF chat endpoint.
        """
        if CheckSignificantChangeTool._hf_client is None:
            CheckSignificantChangeTool._hf_client = InferenceClientModel(
                model_name=model_name,
                api_token=HF_TOKEN
            )
        return CheckSignificantChangeTool._hf_client

    def forward(self, previous_context: str, current_context: str) -> int:
        """
        Compare previous_context and current_context. Return 1 if a significant change
        (new major scene or environment shift) is detected; otherwise return 0.
        """
        prompt = f"""
You are an assistant that compares two story context windows and determines whether a significant change has occurred.
A significant change means a new major scene, environment shift, or a visually important moment that was not present before.

Previous Context:
\"\"\"
{previous_context}
\"\"\"

Current Context:
\"\"\"
{current_context}
\"\"\"

Respond with exactly one number:
- 1 if there is a significant change (new room, new major event, or scene shift)
- 0 if there is no significant change (story continues in the same setting without a major shift)
"""

        system_msg = {
            "role": "system",
            "content": "You compare story contexts and return 1 or 0 based on significant change."
        }
        user_msg = {"role": "user", "content": prompt}

        hf_client = self._get_hf_client()
        resp = hf_client.chat(
            messages=[system_msg, user_msg],
            temperature=0.0,
            max_tokens=10
        )

        # Extract the first integer (1 or 0) from the model response
        text = resp.strip()
        if text.startswith("1"):
            return 1
        else:
            return 0
