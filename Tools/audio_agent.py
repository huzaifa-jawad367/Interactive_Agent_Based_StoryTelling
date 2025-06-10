# voice_narration_tool.py
#-------------libraries to install----------#
#pip install g2p-en, nltk, smolagents
import torch
import soundfile as sf
import nltk
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerWithHifiGan
from smolagents import tool
import os
import warnings
warnings.filterwarnings("ignore")
nltk.download('all')

# ——————————————————————————————
# Global FastSpeech2 model for voice narration
VOICE_MODEL = "espnet/fastspeech2_conformer_with_hifigan"
_tokenizer_voice = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
_model_voice = FastSpeech2ConformerWithHifiGan.from_pretrained(VOICE_MODEL)
_model_voice.eval()
# ——————————————————————————————

@tool
def generate_voice_narration(text: str, output_filename: str = "narration.wav") -> str:
    """
    Generate voice narration from text using FastSpeech2 TTS model.
    
    Args:
        text (str): The text to convert to speech
        output_filename (str): Output audio filename (default: "narration.wav")
    
    Returns:
        str: Path to the generated audio file
    """
    try:
        # Clean and prepare text
        clean_text = text.strip()
        if not clean_text:
            return "Error: Empty text provided"
        
        # Tokenize the input text
        inputs = _tokenizer_voice(clean_text, return_tensors="pt")
        
        # Generate audio
        with torch.no_grad():
            # Move inputs to same device as model if needed
            if torch.cuda.is_available() and next(_model_voice.parameters()).is_cuda:
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate speech
            output = _model_voice(**inputs)
            audio = output.waveform.squeeze().cpu().numpy()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else ".", exist_ok=True)
        
        # Save audio file
        sample_rate = 22050  # FastSpeech2 default sample rate
        sf.write(output_filename, audio, samplerate=sample_rate)
        
        return f"Voice narration saved to: {output_filename}"
        
    except Exception as e:
        return f"Error generating voice narration: {str(e)}"

@tool
def generate_story_narration(story_text: str, chapter_name: str = "chapter") -> str:
    """
    Generate voice narration for story text, handling longer texts by splitting into sentences.
    
    Args:
        story_text (str): The story text to narrate
        chapter_name (str): Name prefix for the output file
    
    Returns:
        str: Status message with output file path
    """
    try:
        # Clean text
        clean_text = story_text.strip()
        if not clean_text:
            return "Error: Empty story text provided"
        
        # For longer texts, we might want to split into sentences and combine
        # or just process the whole text at once (FastSpeech2 can handle reasonably long texts)
        
        output_filename = f"{chapter_name}_narration.wav"
        
        # Tokenize and generate
        inputs = _tokenizer_voice(clean_text, return_tensors="pt")
        
        with torch.no_grad():
            # Move to appropriate device
            if torch.cuda.is_available() and next(_model_voice.parameters()).is_cuda:
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate speech
            output = _model_voice(**inputs)
            audio = output.waveform.squeeze().cpu().numpy()
        
        # Save the narration
        sample_rate = 22050
        sf.write(output_filename, audio, samplerate=sample_rate)
        
        # Calculate duration for user feedback
        duration = len(audio) / sample_rate
        
        return f"Story narration completed! Saved to: {output_filename} (Duration: {duration:.2f} seconds)"
        
    except Exception as e:
        return f"Error generating story narration: {str(e)}"

if __name__ == "__main__":
    # --- Test Cases ---
    test_texts = [
        "The warrior stepped into the shadowy forest, his sword gleaming under the moonlight.....there appeared a monster and he said WHAT ARE YOU DOING I WILL KILL YOU ",
        "Captain Sarah Martinez floated weightlessly in the observation deck of the starship Enterprise.",
        "The old lighthouse keeper climbed the spiral stairs one last time, as the storm raged outside."
    ]
    
    print("Testing voice narration tool...")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: Generating narration for: '{text[:50]}...'")
        result = generate_voice_narration(text=text, output_filename=f"test_narration_{i}.wav")
        print(f"Result: {result}")
    
    # Test story narration
    story = " ".join(test_texts)
    print(f"\nTesting story narration with combined text...")
    story_result = generate_story_narration(story_text=story, chapter_name="test_story")
    print(f"Story Result: {story_result}")