# main.py

import os
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from store_facts import StoryState
from agents import (
    get_story_agent,
    get_fact_agent,
    get_consistency_agent,
    get_choices_agent,
    get_world_agent,
)

app = FastAPI(title="Interactive Storyteller API")

# ---------------------------
# Pydantic Models for Requests
# ---------------------------

class GenerateSceneRequest(BaseModel):
    """
    The front end sends:
      - initial_prompt: used only on the very first call (string or null)
      - last_choice: used on every subsequent call (string or null)
      - state: the full StoryState.to_dict() from the last turn
    Exactly one of (initial_prompt, last_choice) must be non-null.
    """
    initial_prompt: Optional[str]
    last_choice: Optional[str]
    state: Dict[str, Any]


# ---------------------------
# Pydantic Models for Responses
# ---------------------------

class GenerateSceneResponse(BaseModel):
    """
    We return:
      - scene_text: newly generated paragraph
      - new_facts: the dict returned by the fact extractor
      - is_consistent: whether new_facts contradicted old_facts
      - choices: a list of 2–4 next‐step options (empty if inconsistent)
      - world: the dict returned by the world‐builder
      - updated_state: StoryState.to_dict() after merging new_facts & appending scene
    """
    scene_text: str
    new_facts: Dict[str, Any]
    is_consistent: bool
    choices: List[str]
    world: Dict[str, Any]
    updated_state: Dict[str, Any]


# ---------------------------
# Orchestrator Endpoint
# ---------------------------

@app.post("/generate_scene", response_model=GenerateSceneResponse)
def generate_scene_endpoint(req: GenerateSceneRequest):
    # 1) Reconstruct the StoryState from the incoming dict
    try:
        state_obj = StoryState.from_dict(req.state)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid state payload: {e}")

    # 2) Build context string (last N scenes + facts). Use N=3 for example.
    context_str = state_obj.get_context_window(n=3)

    # 3) Call the Story Generator agent (LLM) to get a new scene
    story_agent = get_story_agent()
    try:
        scene_text = story_agent.run(
            initial_prompt=req.initial_prompt,
            last_choice=req.last_choice,
            context=context_str
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Story generation failed: {e}")

    # 4) Call the Fact Extraction agent on the new scene_text
    fact_agent = get_fact_agent()
    try:
        new_facts = fact_agent.run(scene_text=scene_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fact extraction failed: {e}")

    # 5) Validate consistency against old facts
    consistency_agent = get_consistency_agent()
    try:
        is_consistent = consistency_agent.run(
            old_facts=state_obj.facts,
            new_facts=new_facts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consistency check failed: {e}")

    # 6) If inconsistent, return early without updating state
    if not is_consistent:
        return GenerateSceneResponse(
            scene_text=scene_text,
            new_facts=new_facts,
            is_consistent=False,
            choices=[],
            world={},
            updated_state=state_obj.to_dict()
        )

    # 7) Merge new_facts into state and append the scene
    state_obj.update_facts(new_facts)
    state_obj.append_scene(scene_text)

    # 8) Generate next‐step choices
    choices_agent = get_choices_agent()
    try:
        choices = choices_agent.run(
            scene_text=scene_text,
            facts=state_obj.facts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Choice generation failed: {e}")

    # 9) Build world description/details
    world_agent = get_world_agent()
    try:
        world = world_agent.run(facts=state_obj.facts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"World‐building failed: {e}")

    # 10) Return everything, including the updated StoryState
    return GenerateSceneResponse(
        scene_text=scene_text,
        new_facts=new_facts,
        is_consistent=True,
        choices=choices,
        world=world,
        updated_state=state_obj.to_dict()
    )


# ---------------------------
# (Optional) Individual-Agent Endpoints
# ---------------------------

@app.post("/extract_facts")
def extract_facts_endpoint(payload: Dict[str, str]):
    """
    Exposes the Fact Extraction tool in isolation, for debugging.
    Request schema: {"scene_text": "<text>"}
    Response: {"facts": {...}}
    """
    scene_text = payload.get("scene_text", "")
    if not scene_text:
        raise HTTPException(status_code=400, detail="Missing 'scene_text'.")
    fact_agent = get_fact_agent()
    try:
        facts = fact_agent.run(scene_text=scene_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fact extraction error: {e}")
    return {"facts": facts}


@app.post("/generate_choices")
def generate_choices_endpoint(payload: Dict[str, Any]):
    """
    Exposes the Choice Generation tool in isolation.
    Request schema: {"scene_text": "<text>", "facts": {...}}
    Response: {"choices": ["...", "..."]}
    """
    scene_text = payload.get("scene_text", "")
    facts = payload.get("facts", {})
    if not scene_text or not isinstance(facts, dict):
        raise HTTPException(status_code=400, detail="Missing or invalid 'scene_text' or 'facts'.")
    choices_agent = get_choices_agent()
    try:
        choices = choices_agent.run(scene_text=scene_text, facts=facts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Choice generation error: {e}")
    return {"choices": choices}


@app.post("/build_world")
def build_world_endpoint(payload: Dict[str, Any]):
    """
    Exposes the World-Building tool in isolation.
    Request schema: {"facts": {...}}
    Response: {"world": {...}}
    """
    facts = payload.get("facts", {})
    if not isinstance(facts, dict):
        raise HTTPException(status_code=400, detail="Missing or invalid 'facts'.")
    world_agent = get_world_agent()
    try:
        world = world_agent.run(facts=facts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"World‐building error: {e}")
    return {"world": world}


@app.post("/validate_consistency")
def validate_consistency_endpoint(payload: Dict[str, Any]):
    """
    Exposes the Consistency Validation tool in isolation.
    Request schema: {"old_facts": {...}, "new_facts": {...}}
    Response: {"is_consistent": true/false}
    """
    old_facts = payload.get("old_facts", {})
    new_facts = payload.get("new_facts", {})
    if not isinstance(old_facts, dict) or not isinstance(new_facts, dict):
        raise HTTPException(status_code=400, detail="Missing or invalid 'old_facts' or 'new_facts'.")
    consistency_agent = get_consistency_agent()
    try:
        is_consistent = consistency_agent.run(old_facts=old_facts, new_facts=new_facts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consistency check error: {e}")
    return {"is_consistent": is_consistent}


# ---------------------------
# Start the app
# ---------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
