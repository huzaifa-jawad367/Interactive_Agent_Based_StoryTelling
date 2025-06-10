# scene_extractor_tool.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from smolagents import tool
import warnings
warnings.filterwarnings("ignore")

# ——————————————————————————————
# Global Mistral-7B-Instruct model for scene extraction
CHOICE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
_tokenizer_ch = AutoTokenizer.from_pretrained(CHOICE_MODEL)
_model_ch = AutoModelForCausalLM.from_pretrained(
    CHOICE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
_model_ch.eval()
# ——————————————————————————————

@tool
def extract_scene(context_text: str) -> str:
    """
    Identify and return one vivid paragraph describing the key visual scene (<=77 tokens).
    
    Args:
        context_text (str): The input text to extract a visual scene from
    
    Returns:
        str: A vivid paragraph describing the key visual scene
    """
    prompt = f"""
You are a visual scene extractor. Given the text below, identify the key visual moment and describe it in one vivid, concise paragraph (maximum 77 tokens). Focus on the most striking visual elements and atmosphere. Return only the scene description.

Text:
\"\"\"
{context_text}
\"\"\"

Visual description:"""

    # wrap in a chat template
    messages = [
        {"role": "system", "content": "You extract vivid visual scenes from text."},
        {"role": "user", "content": prompt}
    ]

    # tokenize & move to device
    inputs = _tokenizer_ch.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(_model_ch.device)

    # generate
    with torch.no_grad():
        outputs = _model_ch.generate(
            **inputs, 
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False,
            pad_token_id=_tokenizer_ch.eos_token_id
        )

    # slice off prompt
    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0][prompt_len:]

    # decode response
    raw = _tokenizer_ch.decode(gen_ids, skip_special_tokens=True)
    response = raw.strip()

    # ensure within token limit and not empty
    if response:
        words = response.split()
        if len(words) > 77:
            response = ' '.join(words[:77])
        return response
    else:
        # fallback to first sentence if no response
        return context_text.split('.')[0] + '.'

if __name__ == "__main__":
    # --- Test Cases ---
    contexts = [
        "The wizard stood atop the crystal tower as lightning crackled around his staff. Below, the armies of darkness gathered in the shadowy valley.",
        "Captain Sarah Martinez floated weightlessly in the observation deck of the starship Enterprise. Through the massive viewport, she watched as the alien mothership approached.",
        "The old lighthouse keeper climbed the spiral stairs one last time. Outside, the storm raged against the rocky cliffs."
    ]
    
    for i, ctx in enumerate(contexts, 1):
        scene = extract_scene(context_text=ctx)
        tokens = len(scene.split())
        print(f"Test {i}: Extracted scene ({tokens} tokens): {scene}")