import argparse
import json
import logging
import os
import sys
from threading import Lock
from pathlib import Path

import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, Part, Task, TaskState, TextPart, UnsupportedOperationError
from a2a.utils.errors import ServerError
from a2a.utils import new_agent_text_message, new_task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.designer import BaseDesigner
class MarioDesigner(BaseDesigner):
    def __init__(self, name: str):
        super().__init__(name)
        maps_path = Path("scenarios/mario/tools/maps.json")
        raw_maps = json.loads(maps_path.read_text(encoding="utf-8"))
        self.maps = list(raw_maps.values())
        if not self.maps:
            raise ValueError(f"No maps found in {maps_path}")
        self._map_index = 0
        self._lock = Lock()
        self._tile_replacements = str.maketrans({
            "{": "M",
            "}": "F",
            "?": "Q",
            "H": "#",
        })
        self._allowed_tiles = set("MF- X#SCLUD%|?@Q!12otT<>[]*BbEgGkKrRyY")

    def get_instruction(self) -> str:
        return """
You are a Mario map provider.
When a request arrives, call `next_map_tool` exactly once and return only the tool result.
"""

    def _next_map(self) -> str:
        with self._lock:
            map_text = self.maps[self._map_index]
            self._map_index = (self._map_index + 1) % len(self.maps)
        normalized = self._normalize_map(map_text)
        return f"```ascii\n{normalized}\n```"

    def _normalize_map(self, map_text: str) -> str:
        # Remove empty lines and keep only tiles supported by Mario-AI parser.
        replaced_map = map_text.translate(self._tile_replacements)
        raw_lines = [line.rstrip("\r") for line in replaced_map.splitlines() if line.strip()]
        if not raw_lines:
            raw_lines = ["-" * 64 for _ in range(14)]

        width = max(len(line) for line in raw_lines)
        grid = [list(line.ljust(width, "-")) for line in raw_lines]
        self._normalize_enemies(grid)

        lines = []
        for row in grid:
            cleaned = "".join(ch if ch in self._allowed_tiles else "-" for ch in row)
            lines.append(cleaned)

        # Ensure at least one start and one flag exist.
        has_start = any("M" in line for line in lines)
        has_flag = any("F" in line for line in lines)
        floor_idx = len(lines) - 2 if len(lines) >= 2 else len(lines) - 1
        floor_idx = max(0, floor_idx)

        if not has_start and width > 1:
            row = list(lines[floor_idx])
            row[0] = "M"
            lines[floor_idx] = "".join(row)

        if not has_flag and width > 1:
            row = list(lines[floor_idx])
            row[-1] = "F"
            lines[floor_idx] = "".join(row)

        return "\n".join(lines)

    def _normalize_enemies(self, grid: list[list[str]]) -> None:
        if not grid:
            return

        height = len(grid)
        width = len(grid[0])
        for y in range(height):
            for x in range(width):
                if grid[y][x] != "E":
                    continue

                below = grid[y + 1][x] if y + 1 < height else "-"
                if below in {"<", ">"}:
                    self._convert_pipe_to_flower_pipe(grid, y + 1, x)
                    grid[y][x] = "-"
                elif below == "-":
                    grid[y][x] = "K"
                else:
                    grid[y][x] = "g"

    def _convert_pipe_to_flower_pipe(self, grid: list[list[str]], pipe_top_y: int, x: int) -> None:
        width = len(grid[0])
        top_char = grid[pipe_top_y][x]

        if top_char == "<":
            left_x = x
        elif top_char == ">":
            left_x = x - 1
        else:
            return

        right_x = left_x + 1
        if left_x < 0 or right_x >= width:
            return

        if grid[pipe_top_y][left_x] != "<" or grid[pipe_top_y][right_x] != ">":
            return

        y = pipe_top_y
        while y < len(grid):
            left_char = grid[y][left_x]
            right_char = grid[y][right_x]
            if y == pipe_top_y:
                if left_char != "<" or right_char != ">":
                    break
            elif left_char != "[" or right_char != "]":
                break

            grid[y][left_x] = "T"
            grid[y][right_x] = "T"
            y += 1

    def run_a2a_server(self, host: str = "127.0.0.1", port: int = 9110, card_url: str = None):
        logging.basicConfig(level=logging.INFO)
        agent_card = AgentCard(
            name=self.name,
            description="Preset Mario Map Designer A2A Agent",
            url=card_url or f"http://{host}:{port}/",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[],
        )

        class PresetDesignerExecutor(AgentExecutor):
            def __init__(self, outer):
                self.outer = outer

            async def execute(self, context, event_queue) -> None:
                msg = context.message
                if not msg:
                    raise ValueError("Missing message.")

                task = new_task(msg)
                await event_queue.enqueue_event(task)
                updater = TaskUpdater(event_queue, task.id, task.context_id)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message("Providing preset Mario map.", context_id=context.context_id),
                )
                await updater.add_artifact(
                    parts=[Part(root=TextPart(kind="text", text=self.outer._next_map()))],
                    name="map",
                )
                await updater.complete()

            async def cancel(self, request, event_queue) -> Task | None:
                raise ServerError(error=UnsupportedOperationError())

        request_handler = DefaultRequestHandler(PresetDesignerExecutor(self), InMemoryTaskStore())
        a2a_app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
        logging.info(f"Starting A2A Designer {self.name} on {host}:{port}")
        uvicorn.run(a2a_app.build(), host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Mario Map Designer Agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9110)
    parser.add_argument("--card-url", type=str)
    args = parser.parse_args()

    designer = MarioDesigner("MarioDesigner")
    designer.run_a2a_server(host=args.host, port=args.port, card_url=args.card_url)
