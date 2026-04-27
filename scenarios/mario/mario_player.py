import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, Part, Task, TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.player import BasePlayer

class MarioPlayer(BasePlayer):
    _tile_replacements = str.maketrans({
        "H": "D",
        "{": "M",
        "}": "F",
    })

    def _sanitize_map_for_player(self, text_map: str) -> str:
        replaced = text_map.translate(self._tile_replacements)
        lines = [line.rstrip("\r") for line in replaced.splitlines() if line.strip()]
        return "\n".join(lines)

    def play(self, text_map: str, output_dir: str, output_name: str, idx: int) -> Optional[str]:
        jar_path = Path("scenarios/mario/PlayAstar.jar")
        if not jar_path.exists():
            print(f"PlayAstar.jar not found at {jar_path.absolute()}")
            return None

        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        map_filename = output_name.replace("_video.mp4", "_map.txt")
        map_path = out_path / map_filename
        sanitized_map = self._sanitize_map_for_player(text_map)
        map_path.write_text(sanitized_map, encoding="utf-8")

        video_file = out_path / output_name

        assets_path = Path("scenarios/mario/img/")
        assets_arg = str(assets_path.resolve()).rstrip("/\\") + "/"

        cmd = [
            "java", "-Djava.awt.headless=true", "-jar", jar_path.name,
            str(map_path), "human", assets_arg, str(out_path), output_name
        ]

        try:
            subprocess.run(cmd, check=True, timeout=120, cwd=jar_path.parent)
            if video_file.exists() and video_file.stat().st_size > 0:
                # Return the absolute or relative path to the generated map video
                return str(video_file.absolute())
        except Exception as e:
            print(f"PlayAstar execution failed: {e}")
            
        return None

    def _parse_prompt(self, prompt: str) -> tuple[str, int]:
        match = re.search(r"Here is the map:\s*(.*?)\s*Index:\s*(\d+)\s*$", prompt, re.DOTALL)
        if not match:
            raise ValueError("Invalid player request format.")
        text_map = match.group(1).strip()
        idx = int(match.group(2))
        return text_map, idx

    def run_a2a_server(self, host: str = "127.0.0.1", port: int = 9120, card_url: str = None):
        logging.basicConfig(level=logging.INFO)
        player = self

        class LocalPlayerExecutor(AgentExecutor):
            async def execute(self, context, event_queue) -> None:
                msg = context.message
                if not msg:
                    raise ValueError("Missing message.")

                request_text = ""
                for part in msg.parts:
                    root = getattr(part, "root", None)
                    if isinstance(root, TextPart):
                        request_text += root.text

                text_map, idx = player._parse_prompt(request_text)
                output_name = f"eval_map_{idx}_video.mp4"

                task = new_task(msg)
                await event_queue.enqueue_event(task)
                updater = TaskUpdater(event_queue, task.id, task.context_id)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Simulating Mario map {idx}.", context_id=context.context_id),
                )

                video_path = player.play(text_map, "outputs/mario", output_name, idx) or ""
                await updater.add_artifact(
                    parts=[Part(root=TextPart(kind="text", text=video_path))],
                    name="video_path",
                )
                await updater.complete()

            async def cancel(self, request, event_queue) -> Task | None:
                raise ServerError(error=UnsupportedOperationError())

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

        request_handler = DefaultRequestHandler(LocalPlayerExecutor(), InMemoryTaskStore())
        server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
        logging.info(f"Starting A2A Player {self.name} on {host}:{port}")
        uvicorn.run(server.build(), host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Mario Map Player Agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9120)
    parser.add_argument("--card-url", type=str)
    args = parser.parse_args()

    player = MarioPlayer("MarioPlayer")
    player.run_a2a_server(host=args.host, port=args.port, card_url=args.card_url)
