
# Simplified Hume TTS tool for storybook narration in Kaggle
#-------------libraries to install----------#
# !pip install hume smolagents nest-asyncio
import os
import asyncio
import base64
from smolagents import tool
from hume import AsyncHumeClient
import warnings
warnings.filterwarnings("ignore")

# ——————————————————————————————
# Hume API Configuration
HUME_API_KEY = "enter ur hume api "
NARRATOR_DESCRIPTION = "A warm, engaging storybook narrator with a clear, expressive voice that brings stories to life for listeners of all ages"

# Initialize Hume client
client = AsyncHumeClient(api_key=HUME_API_KEY)
# ——————————————————————————————

async def write_audio_to_file(audio_data: str, filename: str) -> str:
    """Helper function to write base64 audio data to file."""
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        # Write to file
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        
        return f"Story narration saved to: {filename} (Size: {len(audio_bytes)} bytes)"
    except Exception as e:
        return f"Error saving audio: {str(e)}"

def run_async_safely(async_func):
    """Helper to run async functions safely in Kaggle/Jupyter environments."""
    try:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(async_func)
    except ImportError:
        try:
            return asyncio.run(async_func)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_func)
            finally:
                loop.close()

@tool
def narrate_story(story_text: str, output_filename: str = "story_narration.wav") -> str:
    """
    Generate storybook narration from text using Hume's TTS with narrator voice.
    
    Args:
        story_text (str): The story text to narrate (paragraph form)
        output_filename (str): Output audio filename (default: "story_narration.wav")
    
    Returns:
        str: Path to the generated audio file or error message
    """
    async def _narrate():
        try:
            # Clean and prepare text
            clean_text = story_text.strip()
            if not clean_text:
                return "Error: Empty story text provided"
            
            # Check if text is too long (Hume has limits)
            if len(clean_text) > 3000:
                return f"Error: Story text too long ({len(clean_text)} characters). Please use text under 3000 characters."
            
            # Prepare utterance with narrator description
            utterance = {
                "text": clean_text,
                "description": NARRATOR_DESCRIPTION
            }
            
            # Call TTS API
            response = await client.tts.synthesize_json(
                utterances=[utterance]
            )
            
            # Extract audio data
            if response.generations and len(response.generations) > 0:
                audio_data = response.generations[0].audio
                generation_id = response.generations[0].generation_id
                
                # Save audio to file
                result = await write_audio_to_file(audio_data, output_filename)
                return f"{result} | Generation ID: {generation_id}"
            else:
                return "Error: No audio generated"
                
        except Exception as e:
            return f"Error generating story narration: {str(e)}"
    
    return run_async_safely(_narrate())


if __name__ == "__main__":
    # --- Test Connection First ---
    print("Testing Hume Storybook Narrator...")
#    connection_result = test_narrator_connection()
   # print(f"Connection Result: {connection_result}")
    
    if "successful" in connection_result:
        # --- Story Test Cases ---
        story_samples = [
            {
                "text": """In the heart of the Whispering Woods, where sunlight danced through ancient oak leaves, lived a small fox named Rusty. Unlike other foxes, Rusty had a peculiar habit of collecting shiny stones that seemed to glow with their own inner light. Each stone held a memory, a story, a piece of magic that only Rusty could understand.""",
                "filename": "rusty_fox_story.wav"
            },
            {
                "text": """The lighthouse keeper's daughter, Emma, had always wondered what lay beyond the endless ocean horizon. Every night, she would climb to the top of the lighthouse and watch the stars reflect on the water like scattered diamonds. Tonight was different though - tonight, she saw something extraordinary swimming in the moonlit waves.""",
                "filename": "lighthouse_mystery.wav"
            },
            {
                "text": """Deep in the castle's forgotten library, between dusty shelves of ancient books, Professor Willowby discovered a book that seemed to write itself. The pages turned on their own, revealing stories of brave knights, wise dragons, and magical kingdoms that existed just beyond the edge of reality.""",
                "filename": "magical_library.wav"
            },
            {
                "text": """In a small village where everyone knew each other's names, there was a bakery that smelled of cinnamon and dreams. The baker, Mrs. Honeysuckle, had a secret ingredient that made her pastries special - she baked them with wishes and hopes, and somehow, they always came true.""",
                "filename": "magic_bakery.wav"
            }
        ]
        
        print("\nGenerating story narrations...")
        
        for i, story in enumerate(story_samples, 1):
            print(f"\nNarrating Story {i}: {story['text'][:80]}...")
            result = narrate_story(
                story_text=story['text'],
                output_filename=story['filename']
            )
            print(f"Result: {result}")

