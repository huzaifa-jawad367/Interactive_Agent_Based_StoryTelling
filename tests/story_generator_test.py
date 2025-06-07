# 2. Write test code to file
import pytest
from Tools.story_generator import StoryGeneratorTool

class DummyClient:
    def __init__(self, response):
        self.response = response
        self.last_messages = None
    def chat(self, messages, temperature, max_tokens):
        self.last_messages = messages
        return self.response

@pytest.fixture(autouse=True)
def patch_hf_client(monkeypatch):
    dummy = DummyClient("dummy response")
    monkeypatch.setattr(StoryGeneratorTool, "_get_hf_client", lambda self: dummy)
    return dummy

def test_both_initial_and_last_choice_raises():
    tool = StoryGeneratorTool()
    with pytest.raises(ValueError):
        tool.forward(initial_prompt="start", last_choice="choice", context="")

def test_initial_prompt_only(patch_hf_client):
    dummy = patch_hf_client
    tool = StoryGeneratorTool()
    result = tool.forward(initial_prompt="Once upon a time", context="")
    assert result == "dummy response"

def test_last_choice_only(patch_hf_client):
    dummy = patch_hf_client
    tool = StoryGeneratorTool()
    result = tool.forward(last_choice="Follow the path", context="Context text")
    assert result == "dummy response"

def test_context_only(patch_hf_client):
    dummy = patch_hf_client
    tool = StoryGeneratorTool()
    result = tool.forward(context="Only context here")
    assert result == "dummy response"