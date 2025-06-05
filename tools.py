# fact_agent.py

from typing import Dict, Any
from smolagents import tool

@tool
def generate_scene(text_prompt: str, context: str) -> str:
    """
    Build a prompt like:
      SYSTEM: “You are a children’s‐book style storyteller…”
      USER: “Context: {context}\nLast choice: {text_prompt}\nGenerate the next scene.”
    Then call your LLM and return the raw text.
    """
    # TODO: call LLM, return its response
    return "Ali carefully raised her lantern, the shadows dancing as she pressed forward..."



@tool
def extract_facts(text: str) -> Dict[str, Any]:
    """
    Stub for fact extraction. In a real implementation, this would:
      1. Call an LLM or use a rule-based parser to identify new entities/attributes.
      2. Return a dictionary of extracted facts, e.g.:
           { "Ali": { "location": "rainy_forest" }, "weather": "rainy", ... }

    For now, it returns an empty dict.
    """
    # TODO: replace with actual LLM-based or rule-based extraction logic
    return {}

