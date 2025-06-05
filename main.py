from huggingface_hub import login

# Replace this with your actual HF token:
HF_TOKEN = "Enter your HUGGING_FACE_API_TOKEN"

# This will write the token into ~/.huggingface/token automatically
login(token=HF_TOKEN)



