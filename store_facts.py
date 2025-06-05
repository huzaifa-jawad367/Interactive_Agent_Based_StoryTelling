# story_state.py

from typing import List, Dict, Any
import json


class StoryState:
    """
    Holds the evolving state of the story, including:
      - scene_history: list of scene texts in order
      - facts: dictionary of structured facts extracted so far
      - npc_states: dictionary tracking NPC-specific state (e.g., status, location)
    """

    def __init__(self):
        self.scene_history: List[str] = []
        self.facts: Dict[str, Any] = {}
        self.npc_states: Dict[str, Any] = {}

    def update_facts(self, new_facts: Dict[str, Any]) -> None:
        """
        Merge new_facts into the existing facts dictionary.
        If a key already exists, it will be overwritten by new_facts[key].
        """
        for key, value in new_facts.items():
            # If the fact is nested (e.g., an NPC state), you can choose to merge deeper.
            # For now, we do a shallow update:
            self.facts[key] = value

    def get_context_window(self, n: int) -> str:
        """
        Return a string containing the last `n` scenes plus the current facts as JSON.
        This is useful for building prompts that need recent context and facts.
        """
        last_scenes = self.scene_history[-n:]
        scenes_text = "\n".join(last_scenes)
        facts_json = json.dumps(self.facts, indent=2)

        context_parts = []
        if scenes_text:
            context_parts.append(f"Recent scenes:\n{scenes_text}")
        if self.facts:
            context_parts.append(f"Current facts:\n{facts_json}")

        return "\n\n".join(context_parts)

    def append_scene(self, scene_text: str) -> None:
        """
        Add a newly generated scene to the history.
        """
        self.scene_history.append(scene_text)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the StoryState to a dict (for sending over HTTP, JSON, etc.).
        """
        return {
            "scene_history": self.scene_history,
            "facts": self.facts,
            "npc_states": self.npc_states,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoryState":
        """
        Reconstruct a StoryState from a dict (e.g., from a JSON payload).
        """
        obj = cls()
        obj.scene_history = data.get("scene_history", [])
        obj.facts = data.get("facts", {})
        obj.npc_states = data.get("npc_states", {})
        return obj
