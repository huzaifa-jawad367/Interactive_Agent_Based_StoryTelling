import json
from typing import Any, Dict

from your_tool_base import Tool            # Replace with wherever “Tool” is defined
from huggingface_hub.hf_api import HfFolder
from your_inference_client import InferenceClientModel   # Replace with your actual HF client import

# (Make sure HF_TOKEN is already defined in your environment)
model_name = "meta-llama/Llama-2-7b-chat-hf"

class ExtractSceneTool(Tool):
    """
    Extracts the single most “image-worthy” scene from a block of story context,
    and formats it as a standalone description. Returns exactly one paragraph
    (string) that an image-generation model can consume.
    """

    name = "extract_scene"
    description = """
    Given a chunk of narrative (one or more paragraphs), pick out the single most
    visually striking or important scene and rewrite it as a standalone prompt.
    The output should be a concise, vivid paragraph describing that scene in
    “cartoonic” (or whatever style you request) detail, ready for an image model.
    No JSON—just return raw text.
    """

    inputs = {
        "context_text": {
            "type": "string",
            "description": "One or more paragraphs of story. The tool should identify the key moment and produce an image prompt.",
            "required": True,
        }
    }
    output_type = "string"

    _hf_client: InferenceClientModel = None

    def _get_hf_client(self) -> InferenceClientModel:
        """
        Lazy-instantiates a single InferenceClientModel for the HF chat endpoint.
        """
        if ExtractSceneTool._hf_client is None:
            # Save your token to disk if not already done:
            # HfFolder.save_token(HF_TOKEN)
            ExtractSceneTool._hf_client = InferenceClientModel(
                model_name=model_name,
                api_token=HF_TOKEN
            )
        return ExtractSceneTool._hf_client

    def forward(self, context_text: str) -> str:
        # 1) Build the instruction prompt
        prompt = f"""
You are a scene-extraction assistant for an image-generation pipeline. Below is a block of story text:

\"\"\"
{context_text}
\"\"\"

Identify the single most important, visually striking moment (or “scene”) from this excerpt that would best translate into a cartoon‐style illustration. Then, write that scene as one concise, vivid paragraph, focusing on exactly what should appear in the image (characters, setting, lighting, mood, and any key objects). Do NOT output JSON—just return a straightforward, self-contained description suitable for plugging into a Stable Diffusion prompt.
"""

        system_msg = {
            "role": "system",
            "content": "You extract the most “image-worthy” moment from the story context and return a single descriptive paragraph for illustration."
        }
        user_msg = {"role": "user", "content": prompt}

        # 2) Call the chat model
        hf_client = self._get_hf_client()
        resp = hf_client.chat(
            messages=[system_msg, user_msg],
            temperature=0.0,
            max_tokens=150
        )

        # 3) Return the raw text (no JSON parsing)
        return resp.strip()
