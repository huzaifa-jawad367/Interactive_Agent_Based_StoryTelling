import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_utils import tokenizer, model, generate_completion
from smolagents import tool
import warnings
warnings.filterwarnings("ignore")

@tool
def check_significant_change(previous_context: str, current_context: str) -> int:
    """
    Compare previous and current context; return 1 if major change (new scene/env), else 0.
    
    Args:
        previous_context (str): The previous context text
        current_context (str): The current context text
    
    Returns:
        int: 1 if major significant change detected, 0 otherwise
    """
    prompt = f"""
Compare these two contexts and determine if there is a major significant change (like a new scene, environment, or dramatic shift in situation). Reply with only "change" for a major significant change, or "unchange" if the contexts are similar or show minor differences.

Previous: {previous_context}
Current: {current_context}

Answer (change or unchange):"""

    # wrap in a chat template
    messages = [
        {"role": "system", "content": "You detect significant changes between contexts. Reply only with 'change' or 'unchange'."},
        {"role": "user", "content": prompt}
    ]

    # tokenize & move to device
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    # generate
    with torch.no_grad():
        outputs = generate_completion(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # slice off prompt
    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0][prompt_len:]

    # decode response
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    response = raw.strip().lower()

    # check for change indicators
    if "change" in response and "unchange" not in response:
        return 1
    else:
        return 0

if __name__ == "__main__":
    # --- Test Cases ---
    tests = [
        ("John types at his desk in the morning light.", "John now types with a cup of coffee beside him."),
        ("Sarah walks through the quiet library browsing books.", "She stands on a cliff overlooking crashing waves."),
        ("Morning vendors set up at the market.", "The empty market is silent under the moonlight.")
    ]
    
    for i, (prev, curr) in enumerate(tests, 1):
        result = check_significant_change(previous_context=prev, current_context=curr)
        print(f"Test {i}: Prev='{prev}' | Curr='{curr}' -> Change Detected: {result}")