import argparse
import asyncio
import os
import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import uvicorn
from dotenv import load_dotenv

from google import genai
from google.genai import types as genai_types

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, Part, TaskState, DataPart
from a2a.utils import new_agent_text_message
from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest
from agentbeats.tool_provider import ToolProvider

logger = logging.getLogger("evaluator")

class BaseEvaluator(GreenAgent):
    def __init__(self, game_name: str, eval_config: dict, player=None):
        super().__init__()
        self._client = genai.Client()
        self._tool_provider = ToolProvider()
        self.game_name = game_name
        self.eval_config = eval_config
        self.player = player
        self._base_parts = []

    def validate_request(self, request: EvalRequest) -> Tuple[bool, str]:
        if "designer" not in request.participants:
            return False, "Participant 'designer' is required."
        return True, "ok"

    def _init_base_parts(self):
        self._base_parts = []
        criterion_text_path = Path(f"scenarios/{self.game_name}/prompts/initial_criterion_prompt.md")
        criterion_video_path = Path(f"scenarios/{self.game_name}/prompts/initial_criterion_video.mp4")
        if criterion_text_path.exists():
            self._base_parts.append(genai_types.Part(text=criterion_text_path.read_text(encoding="utf-8")))
            if criterion_video_path.exists():
                try:
                    self._base_parts.append(genai_types.Part.from_bytes(data=criterion_video_path.read_bytes(), mime_type="video/mp4"))
                except Exception:
                    pass
            self._base_parts.append(genai_types.Part(text="Previous Model Response: OK"))

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        def status_message(text: str):
            return new_agent_text_message(text)

        num_maps = int(req.config.get("num_maps", 1))
        video_only = bool(req.config.get("video_only", False))
        output_dir = req.config.get("jar_output_dir", req.config.get("output_dir", f"outputs/{self.game_name}"))
        os.makedirs(output_dir, exist_ok=True)
        
        designer_endpoint = str(req.participants["designer"])
        
        mode_text = "video generation only" if video_only else "full evaluation"
        await updater.update_status(TaskState.working, status_message(f"Starting A2A evaluation for {self.game_name} ({num_maps} iterations, mode: {mode_text})"))
        if not video_only:
            self._init_base_parts()

        results = []
        for i in range(1, num_maps + 1):
            await updater.update_status(TaskState.working, status_message(f"Requesting map {i}/{num_maps} from Designer..."))
            prompt = self.get_designer_prompt(i, num_maps)
            designer_resp = await self._tool_provider.talk_to_agent(prompt, designer_endpoint, new_conversation=(i==1))
            
            ascii_map = self.extract_map(designer_resp)
            if not ascii_map:
                logger.error(f"Map {i}: Failed to extract map.")
                result = {
                    "map_index": i,
                    "status": "failed",
                    "stage": "extract_map",
                    "reason": "Failed to extract map from designer response.",
                }
                results.append(result)
                Path(output_dir, f"result_{i}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                continue

            await updater.update_status(TaskState.working, status_message(f"Map {i}: Simulation started..."))
            video_path = None
            if "player" in req.participants:
                player_endpoint = str(req.participants["player"])
                player_prompt = f"Here is the map:\n{ascii_map}\n\nIndex: {i}"
                player_resp = await self._tool_provider.talk_to_agent(player_prompt, player_endpoint, new_conversation=True)
                video_path = self.extract_video_path(player_resp)
            elif self.player:
                video_path = self.player.play(ascii_map, output_dir, f"eval_map_{i}.mp4", i)
                
            if not video_path:
                logger.error(f"Map {i}: Gameplay simulation failed.")
                result = {
                    "map_index": i,
                    "status": "failed",
                    "stage": "simulate_video",
                    "reason": "Player did not return a video path.",
                    "map": ascii_map,
                }
                results.append(result)
                Path(output_dir, f"result_{i}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                continue
                
            # Need strict strip for checking if file exists
            clean_video_path = video_path.replace("```", "").strip()
            if not Path(clean_video_path).exists():
                logger.error(f"Map {i}: Video path invalid or file does not exist ({clean_video_path}).")
                result = {
                    "map_index": i,
                    "status": "failed",
                    "stage": "validate_video",
                    "reason": f"Video path invalid or file does not exist ({clean_video_path}).",
                    "map": ascii_map,
                    "video_path": clean_video_path,
                }
                results.append(result)
                Path(output_dir, f"result_{i}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                continue

            if video_only:
                result = {
                    "map_index": i,
                    "status": "video_generated",
                    "map": ascii_map,
                    "video_path": clean_video_path,
                }
            else:
                await updater.update_status(TaskState.working, status_message(f"Map {i}: Evaluating video..."))
                result = self.evaluate_video(clean_video_path)
                result["status"] = "evaluated"
                result["video_path"] = clean_video_path
                result["map_index"] = i
                result["map"] = ascii_map
            results.append(result)
            
            Path(output_dir, f"result_{i}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            
        await updater.add_artifact(parts=[Part(root=DataPart(data={"history": results}))], name="Results")
        success_count = sum(1 for item in results if item.get("status") in {"video_generated", "evaluated"})
        await updater.update_status(TaskState.working, status_message(f"Evaluation complete. Success: {success_count}/{num_maps}"))

    def get_designer_prompt(self, idx: int, total: int) -> str:
        prompt_path = Path(f"scenarios/{self.game_name}/prompts/map_request.md")
        guide_path = Path(f"scenarios/{self.game_name}/prompts/map_ascii_guide.md")
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
            guide = guide_path.read_text(encoding="utf-8") if guide_path.exists() else ""
            return prompt.format(map_num=idx, total=total, map_ascii_guide=guide)
        return f"Please generate map {idx} of {total}."

    def extract_map(self, response: str) -> str:
        match = re.search(r"```(?:\w+)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        content = match.group(1).strip() if match else response.strip()
        lines = [line.rstrip() for line in content.splitlines() if line.strip()]
        if not lines: return ""
        width = max(len(line) for line in lines)
        return "\n".join([line.ljust(width, "-") for line in lines])

    def extract_video_path(self, response: Optional[str]) -> Optional[str]:
        if not response:
            return None

        text = response.strip()
        if not text:
            return None

        # First try plain path extraction (fast path).
        path_match = re.search(r"([A-Za-z]:\\[^\n\"']+\.mp4|/[^\\n\"']+\.mp4|[^\\n\"']+\.mp4)", text)
        if path_match:
            return path_match.group(1).strip()

        # If the response is wrapped in JSON or markdown code fences, parse it.
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        json_text = fenced.group(1).strip() if fenced else text
        try:
            data = json.loads(json_text)
            if isinstance(data, dict):
                # ADK tool response shape: {"response": {"result": "..."}}
                nested = data.get("response")
                if isinstance(nested, dict):
                    result = nested.get("result")
                    if isinstance(result, str) and result.strip():
                        return result.strip()

                # Generic result field fallback.
                result = data.get("result")
                if isinstance(result, str) and result.strip():
                    return result.strip()
        except Exception:
            return None

        return None

    def evaluate_video(self, video_path: str) -> dict:
        prompt = f"You are a professional game level evaluator. {self.eval_config.get('description', '')}\n"
        prompt += 'When a map video is given, respond strictly in JSON: {"explain": "...", "result": {"<category>": {"reason": "...", "score": 1}}, "score": 20}\n'
        for axis in self.eval_config.get("eval_axes", []):
            scale_data = axis.get("scale")
            if isinstance(scale_data, list):
                prompt += f"- {axis['name']}: {axis['description']}\n"
                for scale_item in scale_data:
                    r = scale_item.get("range", [])
                    desc = scale_item.get("description", "")
                    if len(r) == 2:
                        if r[0] == r[1]:
                            prompt += f"  - Score {r[0]}: {desc}\n"
                        else:
                            prompt += f"  - Score {r[0]}-{r[1]}: {desc}\n"
            else:
                prompt += f"- {axis['name']} ({scale_data}-point scale): {axis['description']}\n"
            
        video_bytes = Path(video_path).read_bytes()
        contents = self._base_parts + [genai_types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")]
        
        try:
            resp = self._client.models.generate_content(
                model=os.getenv("MODEL", "gemini-2.5-pro"),
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=prompt,
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            match = re.search(r"```(?:json)?\s*(.*?)```", resp.text, re.DOTALL | re.IGNORECASE)
            raw_text = match.group(1).strip() if match else resp.text.strip()
            return json.loads(raw_text)
        except Exception as e:
            text = getattr(resp, "text", "") if 'resp' in locals() else ""
            return {"error": str(e), "raw": text}

    def run_a2a_server(self, host: str = "127.0.0.1", port: int = 9100, card_url: str = None):
        logging.basicConfig(level=logging.INFO)
        executor = GreenExecutor(self)
        card = AgentCard(
            name=f"{self.game_name.capitalize()}Evaluator",
            description=f"A2A Evaluator for {self.game_name}",
            url=card_url or f"http://{host}:{port}/",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[],
        )
        request_handler = DefaultRequestHandler(executor, InMemoryTaskStore())
        server = A2AStarletteApplication(agent_card=card, http_handler=request_handler)
        uvicorn_server = uvicorn.Server(uvicorn.Config(server.build(), host=host, port=port))
        
        # We manually manage the event loop via main since GreenAgent execution might have conflicts.
        logger.info(f"Starting A2A Evaluator on {host}:{port}")
        asyncio.run(uvicorn_server.serve())
