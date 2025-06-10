# Tools/scene_extractor_tool.py

from typing import Dict
from smolagents import tool
from llm_utils import tokenizer, generate_completion
import torch

@tool
def extract_scene(context: str) -> str:
    """
    Identify the key visual scene in one vivid paragraph (≤77 tokens).
    """
    prompt = f"""
You are a visual scene extractor. Given the text below,
produce one vivid paragraph (max 77 tokens) describing the key visual moment. Return only that paragraph.

Text:
\"\"\"
{context}
\"\"\"

Visual description:"""

    inputs = tokenizer.apply_chat_template(
        [{"role":"system","content":"Extract a single visual scene."},
         {"role":"user","content":prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(tokenizer.device)

    with torch.no_grad():
        outputs = generate_completion(inputs,
                                      max_new_tokens=100,
                                      temperature=0.0,
                                      do_sample=False)

    plen = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0][plen:]
    raw     = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    # enforce token limit
    words = raw.split()
    if len(words) > 77:
        raw = " ".join(words[:77])
    return raw
