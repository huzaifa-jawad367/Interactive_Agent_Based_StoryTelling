# agents.py

import os
from smolagents import CodeAgent, InferenceClientModel

from Tools.build_world import build_world
from Tools.extract_facts import ExtractFactsTool
from Tools.story_generator import StoryGeneratorTool
from Tools.validate_consistency import validate_consistency
from Tools.generate_choices import generate_choices

# Ensure your HF token is set in the environment:
# export HUGGINGFACE_API_TOKEN="hf_YourTokenHere"
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
if not HF_TOKEN:
    raise RuntimeError("Please set HUGGINGFACE_API_TOKEN in your environment.")

# Instantiate a shared Qwen 2.5 InferenceClientModel
_qwen_model = InferenceClientModel(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    api_token=HF_TOKEN
)

# Create agent instances
_story_agent = CodeAgent(
    tools=[StoryGeneratorTool],
    model=_qwen_model
)

_fact_agent = CodeAgent(
    tools=[ExtractFactsTool],
    model=_qwen_model
)

_choices_agent = CodeAgent(
    tools=[generate_choices],
    model=_qwen_model
)

_world_agent = CodeAgent(
    tools=[build_world],
    model=_qwen_model
)

# Consistency agent does not require an LLM
_consistency_agent = CodeAgent(
    tools=[validate_consistency],
    model=None
)


# Getter functions for each agent:

def get_story_agent() -> CodeAgent:
    """
    Returns the singleton Story Generator agent instance.
    """
    return _story_agent


def get_fact_agent() -> CodeAgent:
    """
    Returns the singleton Fact Extraction agent instance.
    """
    return _fact_agent


def get_choices_agent() -> CodeAgent:
    """
    Returns the singleton Choice Generation agent instance.
    """
    return _choices_agent


def get_world_agent() -> CodeAgent:
    """
    Returns the singleton World-Building agent instance.
    """
    return _world_agent


def get_consistency_agent() -> CodeAgent:
    """
    Returns the singleton Consistency Validation agent instance.
    """
    return _consistency_agent
