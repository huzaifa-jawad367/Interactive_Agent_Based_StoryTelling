# story_generator.py

from typing import Optional
import os

from smolagents import Tool
from huggingface_hub import InferenceClient

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
torch.manual_seed(30)


HF_TOKEN   = os.getenv("HUGGINGFACE_API_TOKEN", "")
MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"

if not HF_TOKEN:
    raise RuntimeError("Please set HUGGINGFACE_API_TOKEN in your environment.")

class StoryGeneratorTool(Tool):
    name        = "story_generator"
    description = "Generates the next scene of an interactive story."

    inputs = {
        "context": {
            "type": "string",
            "description": "Concatenated last N scenes + facts.",
            "required": True
        },
        "initial_prompt": {
            "type": "string",
            "description": "The very first user prompt to start the story.",
            "required": False,
            "nullable": True
        },
        "last_choice": {
            "type": "string",
            "description": "The user’s choice from the previous step.",
            "required": False,
            "nullable": True
        },
    }

    output_type = "string"
    _client: Optional[InferenceClient] = None

    def _get_client(self) -> InferenceClient:
        if self._client is None:
            self._client = InferenceClient(token=HF_TOKEN)
        return self._client

    def forward(
        self,
        context: str,
        initial_prompt: Optional[str] = None,
        last_choice:  Optional[str] = None,
    ) -> str:
        if initial_prompt and last_choice:
            raise ValueError("Provide exactly one of `initial_prompt` or `last_choice`.")

        # Build prompt
        if initial_prompt:
            system_content = (
                "You are a children's-book style storyteller. Generate a vivid opening scene."
            )
            user_content = f"User seed prompt:\n\"{initial_prompt}\"\n\nGenerate the opening scene."

        elif last_choice:
            system_content = (
                "You are a children's-book style storyteller. Continue from the last choice."
            )
            user_content = (
                f"Context:\n{context}\n\n"
                f"Last choice: \"{last_choice}\"\n\nGenerate the next scene."
            )

        else:
            system_content = (
                "You are a children's-book style storyteller. Continue based on context alone."
            )
            user_content = f"Context:\n{context}\n\nGenerate the next scene."

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]

        # 1) Tokenize/conform the messages
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-0324")
        model     = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V3-0324", device_map="auto", torch_dtype=torch.bfloat16
        )

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # 2) Generate
        outputs = model.generate(**inputs, max_new_tokens=400)

        # 3) Extract only the generated portion (not the prompt)
        input_length = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_length:]

        # 4) Decode and return
        scene_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return scene_text

        # client = self._get_client()
        # # Non‐streaming chat call
        # resp = client.chat_completion(
        #     model=MODEL_NAME,
        #     messages=messages,
        #     temperature=0.7,
        #     max_tokens=400,
        #     stream=False
        # )

        # return 
