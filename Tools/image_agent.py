import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import warnings
from smolagents import tool

warnings.filterwarnings("ignore")

# Global pipeline variable for reuse
_pipeline = None

def get_pipeline():
    """Initialize and return the Stable Diffusion pipeline."""
    global _pipeline
    if _pipeline is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            _pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(device)
            
            if hasattr(_pipeline, 'enable_attention_slicing'):
                _pipeline.enable_attention_slicing()
                
        except Exception as e:
            print(f"Failed to load pipeline: {e}")
            _pipeline = "mock"
    
    return _pipeline

@tool
def generate_image(scene_prompt: str) -> Image.Image:
    """
    Generates a cartoon-style image from a scene prompt using Stable Diffusion v1.5.
    Falls back to a placeholder if loading fails.
    
    Args:
        scene_prompt (str): Description of the scene to generate
        
    Returns:
        PIL.Image.Image: Generated cartoon-style image
    """
    pipe = get_pipeline()
    
    # Fallback to placeholder if pipeline loading failed
    if pipe == "mock":
        return Image.new('RGB', (512, 512), color='lightblue')
    
    # Enhance prompt for cartoon style
    prompt = f"cartoon style, {scene_prompt}, colorful, animated"
    
    # Generate image with fixed seed for reproducibility
    gen = torch.Generator(device=pipe.device).manual_seed(42)
    
    result = pipe(
        prompt,
        guidance_scale=7.5,
        num_inference_steps=20,
        height=512,
        width=512,
        generator=gen
    )
    
    return result.images[0]

if __name__ == "__main__":
    # Test the function
    test_prompts = [
        "Cartoon cat wearing a wizard hat in a magical forest",
        "Cartoon robot dancing in a disco with neon lights", 
        "Cartoon dragon flying over a rainbow castle"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Generating image {i}: '{prompt}'")
        img = generate_image(prompt)
        print(f"Result: Image size={img.size}, mode={img.mode}")
        
        # Optionally save the image
        # img.save(f"test_image_{i}.png")