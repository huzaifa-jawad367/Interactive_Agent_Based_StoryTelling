# llm_utils.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Load **once** at import time
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.eval()

# Optionally wrap generation logic here
def generate_completion(inputs_tensor, **gen_kwargs):
    with torch.no_grad():
        return model.generate(inputs_tensor, **gen_kwargs)