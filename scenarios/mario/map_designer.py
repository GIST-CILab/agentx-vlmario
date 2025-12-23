import argparse
import asyncio
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    TaskState,
    InvalidParamsError,
    UnsupportedOperationError,
)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError


load_dotenv()

ASCII_GUIDELINES = """
You are a cooperative Mario map designer.
- Wait for external instructions (from the judge) and follow them exactly.
- Do not assume any default format beyond what the caller requests.
- If the caller asks for a fenced code block or specific tiles, comply precisely.
"""

MANUAL_MODE = True

# Manual mode: Pre-made map files
LEVELS_DIR = Path(__file__).parent / "levels"
MANUAL_MAPS = [
    # "test_level_1.txt",
    # "text_level_0.txt",
    "test_level_2.txt",
    "test_level_3.txt",
    "test_level_4.txt",
    "test_level_5.txt",
]


class ManualMapExecutor(AgentExecutor):
    """Custom executor that returns pre-made maps sequentially."""
    
    def __init__(self):
        self.session_map_counts = {}  # Track maps per context
    
    def get_next_map(self, context_id: str | None) -> str:
        """Get the next map for this context."""
        # Use context_id for tracking, fallback to "default" if None
        session_key = context_id or "default"
        
        if session_key not in self.session_map_counts:
            self.session_map_counts[session_key] = 0
        
        map_idx = self.session_map_counts[session_key]
        
        # Cycle through available maps
        file_idx = map_idx % len(MANUAL_MAPS)
        map_file = LEVELS_DIR / MANUAL_MAPS[file_idx]
        
        if not map_file.exists():
            return f"```ascii\n(Map file not found: {map_file})\n```"
        
        try:
            map_content = map_file.read_text(encoding="utf-8").strip()
            self.session_map_counts[session_key] += 1
            print(f"[Manual Mode] Sending map {file_idx + 1}/{len(MANUAL_MAPS)}: {MANUAL_MAPS[file_idx]} (session: {session_key})")
            return f"```ascii\n{map_content}\n```"
        except Exception as e:
            return f"```ascii\n(Error reading map: {e})\n```"
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute task by returning the next pre-made map."""
        # Get the next map using context ID (consistent across conversation)
        map_response = self.get_next_map(context.context_id)
        
        # Use incoming task ID if available, otherwise create new task
        msg = context.message
        if not msg:
            msg = new_agent_text_message("Map request", context_id=context.context_id)
        
        # Handle empty message text
        user_input = context.get_user_input()
        if not user_input or not user_input.strip():
            # Empty message - create placeholder for task creation
            msg = new_agent_text_message("continue", context_id=msg.context_id)
        
        # Check if there's an incoming task ID to reuse
        task_id = getattr(context, 'task_id', None)
        
        if task_id:
            # Continuing conversation - use existing task ID
            updater = TaskUpdater(event_queue, task_id, context.context_id)
        else:
            # New conversation - create new task
            task = new_task(msg)
            await event_queue.enqueue_event(task)
            updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        # Send responses
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Retrieving map...", context_id=context.context_id),
        )
        
        await updater.update_status(
            TaskState.completed,
            new_agent_text_message(map_response, context_id=context.context_id),
        )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancel is not supported for manual map delivery."""
        raise ServerError(error=UnsupportedOperationError())

def main():
    parser = argparse.ArgumentParser(
        description="Run the Mario map designer agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9110,
                        help="Port to bind the server")
    parser.add_argument("--card-url", type=str,
                        help="External URL to provide in the agent card")
    parser.add_argument("--name", type=str,
                        default="MapDesigner", help="Agent display name")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro",
                        help="Model identifier for the ADK agent")
    args = parser.parse_args()

    agent_card = AgentCard(
        name=args.name,
        description="Generates Mario-like ASCII levels within a fenced code block.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )

    if MANUAL_MODE:
        # Manual mode: Use pre-made maps
        print(f"[Manual Mode] Loading maps from: {LEVELS_DIR}")
        print(f"[Manual Mode] Available maps: {len(MANUAL_MAPS)}")
        
        executor = ManualMapExecutor()
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )
        app = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        ).build()
    else:
        # LLM mode: Use ADK Agent
        root_agent = Agent(
            name=args.name,
            model=args.model,
            description="Designs Mario-style ASCII platformer maps.",
            instruction=ASCII_GUIDELINES.strip(),
        )
        app = to_a2a(root_agent, agent_card=agent_card)
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
