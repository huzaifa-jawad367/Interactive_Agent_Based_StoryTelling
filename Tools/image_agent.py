import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from your_tool_base import Tool   # Replace with wherever your base Tool class is defined

model_id = "stabilityai/stable-diffusion-2"

class GenerateImageTool(Tool):
    """
    Generates a cartoon-style illustration given a scene prompt string.
    Uses Stable Diffusion v2 with the Euler scheduler on GPU.
    """

    name = "generate_image"
    description = """
    Given a scene description (string), produce a single image (PIL.Image) by running
    Stable Diffusion v2 with an Euler scheduler. The prompt should already be formatted
    to convey character, setting, lighting, and mood for a cartoon‐style illustration.
    """

    inputs = {
        "scene_prompt": {
            "type": "string",
            "description": "A concise, vivid description of the scene to visualize.",
            "required": True,
        }
    }
    output_type = "image"  # Indicates the tool returns a PIL.Image

    _pipe: StableDiffusionPipeline = None

    @classmethod
    def _get_pipeline(cls) -> StableDiffusionPipeline:
        """
        Lazy‐load the Stable Diffusion v2 pipeline with EulerDiscreteScheduler on GPU.
        """
        if GenerateImageTool._pipe is None:
            # 1) Load the Euler scheduler
            scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            # 2) Load the pipeline (fp16) and move to CUDA
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                scheduler=scheduler,
                torch_dtype=torch.float16
            )
            pipe = pipe.to("cuda")
            # 3) Store for reuse
            GenerateImageTool._pipe = pipe
        return GenerateImageTool._pipe

    def forward(self, scene_prompt: str) -> Any:
        """
        Generates and returns one PIL.Image for the given scene_prompt.
        """
        # 1) Ensure we have the pipeline
        pipe = self._get_pipeline()

        # 2) Run inference (you can tweak num_inference_steps, guidance_scale, etc.)
        image = pipe(
            scene_prompt,
            guidance_scale=7.5,
            num_inference_steps=50,
            height=768,
            width=768,
            generator=torch.Generator("cuda").manual_seed(42)
        ).images[0]

        # 3) Return the generated PIL.Image
        return image
