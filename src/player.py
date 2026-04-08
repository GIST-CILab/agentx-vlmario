from abc import ABC, abstractmethod
import uvicorn
import logging
from typing import Optional
from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from a2a.types import AgentCapabilities, AgentCard

class BasePlayer(ABC):
    def __init__(self, name: str):
        self.name = name
        self.video_path = None

    @abstractmethod
    def play(self, text_map: str, output_dir: str, output_name: str, idx: int) -> Optional[str]:
        """Plays the map locally and returns the local video path."""
        pass

    def run_a2a_server(self, host: str = "127.0.0.1", port: int = 9120, card_url: str = None):
        logging.basicConfig(level=logging.INFO)
        
        def play_tool(text_map: str, idx: int) -> str:
            """Simulates the map game and outputs the string file path to the mp4 video."""
            output_name = f"eval_map_{idx}_video.mp4"
            out_dir = "outputs"
            return str(self.play(text_map, out_dir, output_name, idx) or "")

        instruction = (
            "You are a Player Agent. The user provides a text map and an index (idx). "
            "Use the `play_tool` to simulate the game. You MUST return ONLY the string path returned by `play_tool`, and nothing else."
        )

        root_agent = Agent(
            name=self.name,
            model="gemini-2.5-flash",
            description="Player A2A Agent",
            instruction=instruction,
            tools=[play_tool]
        )
        
        agent_card = AgentCard(
            name=self.name,
            description="Map Player A2A Agent",
            url=card_url or f"http://{host}:{port}/",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[],
        )
        
        a2a_app = to_a2a(root_agent, agent_card=agent_card)
        logging.info(f"Starting A2A Player {self.name} on {host}:{port}")
        uvicorn.run(a2a_app, host=host, port=port)
