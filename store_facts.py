# story_state.py

from typing import List, Dict, Any
import json


from typing import List, Dict, Any
import json

class StoryState:
    """
    Holds the evolving state of the story, including:
      - scene_history: list of scene texts in order
      - facts: dictionary of structured facts extracted so far
      - npc_states: dictionary tracking NPC-specific state (e.g., status, location)
      - world_meta: list of world-building metadata per scene
      - audio_paths: list of file paths for generated TTS audio
      - image_paths: list of file paths for generated images
    """

    def __init__(self):
        self.scene_history: List[str] = []
        self.facts: Dict[str, Any] = {}
        self.npc_states: Dict[str, Any] = {}
        self.world_meta: List[Dict[str, Any]] = []
        self.audio_paths: List[str] = []
        self.image_paths: List[str] = []

    def update_facts(self, new_facts: Dict[str, Any]) -> None:
        """
        Merge new_facts into the existing facts dictionary.
        """
        for key, value in new_facts.items():
            self.facts[key] = value

    def append_scene(self, scene_text: str) -> None:
        """
        Add a newly generated scene to the history.
        """
        self.scene_history.append(scene_text)

    def update_world_meta(self, world: Dict[str, Any]) -> None:
        """
        Store the world-building metadata for the latest scene.
        """
        self.world_meta.append(world)

    def record_audio(self, audio_path: str) -> None:
        """
        Store the file path of the generated TTS audio for the latest scene.
        """
        self.audio_paths.append(audio_path)

    def record_image(self, image_path: str) -> None:
        """
        Store the file path of the generated image for the latest scene.
        """
        self.image_paths.append(image_path)

    def get_context_window(self, n: int) -> str:
        """
        Return a string containing the last `n` scenes plus the current facts as JSON.
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the StoryState to a dict for HTTP/JSON.
        """
        return {
            "scene_history": self.scene_history,
            "facts": self.facts,
            "npc_states": self.npc_states,
            "world_meta": self.world_meta,
            "audio_paths": self.audio_paths,
            "image_paths": self.image_paths,
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
        obj.world_meta = data.get("world_meta", [])
        obj.audio_paths = data.get("audio_paths", [])
        obj.image_paths = data.get("image_paths", [])
        return obj