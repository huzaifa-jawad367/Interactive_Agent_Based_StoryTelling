# Tools/story_generator_tool.py
# Author: Huzaifa Jawad
# Date: 2025-06-05

from typing import Dict, Any, List
from smolagents import Tool, InferenceClientModel
from typing import Optional
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