from abc import ABC, abstractmethod
import os
import logging
import uvicorn
from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from a2a.types import AgentCapabilities, AgentCard

class BaseDesigner(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model_name = os.getenv("MODEL", "gemini-2.5-pro")

    @abstractmethod
    def get_instruction(self) -> str:
        """Returns the system instruction for the LLM Designer"""
        pass

    def generate_map(self, map_num: int, total: int) -> str:
        pass

    def run_a2a_server(self, host: str = "127.0.0.1", port: int = 9110, card_url: str = None):
        logging.basicConfig(level=logging.INFO)
        root_agent = Agent(
            name=self.name,
            model=self.model_name,
            description="A2A Agent that generates maps.",
            instruction=self.get_instruction().strip(),
        )
        agent_card = AgentCard(
            name=self.name,
            description="Map Designer A2A Agent",
            url=card_url or f"http://{host}:{port}/",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[],
        )
        a2a_app = to_a2a(root_agent, agent_card=agent_card)
        logging.info(f"Starting A2A Designer {self.name} on {host}:{port}")
        uvicorn.run(a2a_app, host=host, port=port)
