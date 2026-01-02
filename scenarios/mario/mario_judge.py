import argparse
import asyncio
import contextlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import subprocess
import base64

import uvicorn
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, Part, TaskState, TextPart, DataPart
from a2a.utils import new_agent_text_message

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest
from agentbeats.tool_provider import ToolProvider

try:
    from agentbeats.cloudflare import quick_tunnel
except ImportError:  # pragma: no cover
    quick_tunnel = None


load_dotenv()

logger = logging.getLogger("mario_judge")
logging.basicConfig(level=logging.INFO)

CODE_BLOCK_RE = re.compile(
    r"```(?:ascii|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Load prompt templates
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.md"
REFERENCE_CONTEXT_PATH = PROMPTS_DIR / "reference_context.md"
MAP_ASCII_GUIDE_PATH = PROMPTS_DIR / "map_ascii_guide.md"

def load_text_resource(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default

SYSTEM_PROMPT = load_text_resource(SYSTEM_PROMPT_PATH, "You are an expert Mario level judge.")
REFERENCE_CONTEXT = load_text_resource(REFERENCE_CONTEXT_PATH)
MAP_ASCII_GUIDE = load_text_resource(MAP_ASCII_GUIDE_PATH)


def mario_judge_agent_card(name: str, url: str) -> AgentCard:
    return AgentCard(
        name=name,
        description="Evaluates ASCII Mario platformer levels submitted by participants.",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )


class MarioJudge(GreenAgent):
    """
    Green agent that evaluates Mario maps submitted by purple agents.
    """
    
    NUM_MAPS = 25
    REFERENCE_VIDEO_PATH = PROMPTS_DIR / "initial_criterion_video.mp4"
    REFERENCE_SCORE = 12.0

    def __init__(self):
        self._tool_provider = ToolProvider()
        self._client = genai.Client()
        self._cached_prompt: dict[str, Any] | None = None
        self._log_file_path: Path | None = None

    def _write_log(self, content: str) -> None:
        if self._log_file_path:
            try:
                with open(self._log_file_path, "a", encoding="utf-8") as f:
                    f.write(content + "\n")
            except Exception as exc:
                logger.error(f"Failed to write to log file: {exc}")

    async def _update_status(self, updater: TaskUpdater, message: str) -> None:
        """Update task status and log the message."""
        logger.info(message)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(message),
        )

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        if not request.participants:
            return False, "At least one map designer participant is required."
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        """Main evaluation entry point."""
        self._setup_logging()
        
        # Load configuration
        num_maps = int(req.config.get("num_maps", self.NUM_MAPS))
        jar_output_dir = str(req.config.get("jar_output_dir", "./"))
        jar_output_template = req.config.get("jar_output_name_template", "{role}_gameplay_{ts}_{map_idx}.mp4")
        ref_video_path = req.config.get("reference_video_path", self.REFERENCE_VIDEO_PATH)
        ref_score = float(req.config.get("reference_score", self.REFERENCE_SCORE))

        try:
            # Initialize Prompt Cache
            await self._update_status(updater, "Initializing cached prompt...")
            self._initialize_cached_prompt(str(ref_video_path), ref_score)
            
            all_results: list[dict[str, Any]] = []

            # Evaluate each participant
            for idx, (role, endpoint) in enumerate(req.participants.items(), start=1):
                await self._update_status(updater, f"Starting evaluation for {role} ({idx}/{len(req.participants)})...")
                
                participant_results = await self._evaluate_participant(
                    role=role,
                    endpoint=str(endpoint),
                    num_maps=num_maps,
                    jar_output_dir=jar_output_dir,
                    jar_template=jar_output_template,
                    updater=updater
                )
                all_results.extend(participant_results)

            # Send Final Consolidated Results for Leaderboard
            await self._send_leaderboard_artifact(updater, all_results, req.participants)

        finally:
            self._tool_provider.reset()
            self._write_log(f"\n=== Session Ended at {datetime.now().isoformat()} ===")
            await self._update_status(updater, "All evaluations completed.")

    async def _evaluate_participant(
        self, 
        role: str, 
        endpoint: str, 
        num_maps: int, 
        jar_output_dir: str, 
        jar_template: str,
        updater: TaskUpdater
    ) -> list[dict[str, Any]]:
        """Evaluate a single participant for N maps."""
        results = []
        
        for map_idx in range(1, num_maps + 1):
            # 1. Request Map
            if map_idx == 1:
                # First map: Send full instructions
                prompt = self._build_map_request_prompt(map_idx, num_maps)
                await self._update_status(updater, f"Requesting map {map_idx}/{num_maps} from {role}...")
                response = await self._tool_provider.talk_to_agent(prompt, endpoint, new_conversation=True)
            else:
                # Subsequent maps: Just receive (requested in previous feedback)
                await self._update_status(updater, f"Receiving map {map_idx}/{num_maps} from {role}...")
                response = await self._tool_provider.talk_to_agent("", endpoint, new_conversation=False)

            # 2. Extract & Validate
            ascii_map = self._extract_ascii_map(response)
            if not ascii_map:
                logger.warning(f"Failed to extract map {map_idx} from {role}")
                continue

            # 3. Simulation & Recording
            await self._update_status(updater, f"Simulating map {map_idx}/{num_maps} from {role}...")
            output_name = jar_template.format(role=role, ts=int(time.time()), map_idx=map_idx)
            video_path = self._run_astar_and_record(ascii_map, jar_output_dir, output_name, map_idx)

            # 4. LLM Evaluation
            score_data = self._evaluate_map(ascii_map, video_path)
            
            # 5. Collect Data
            video_base64 = None
            if video_path and Path(video_path).exists():
                try:
                    video_base64 = base64.b64encode(Path(video_path).read_bytes()).decode("utf-8")
                except Exception as e:
                    logger.error(f"Failed to encode video base64: {e}")

            result_entry = {
                "role": role,
                "map_index": map_idx,
                "map": ascii_map,
                "video": video_path,
                "video_base64": video_base64,
                **score_data  # Unpack exact scores from LLM
            }
            results.append(result_entry)

            # 6. Send Feedback & Artifacts
            feedback = f"Map {map_idx} score: {score_data.get('score', 0)}/20"
            if map_idx < num_maps:
                feedback += f"\n\nPlease generate map {map_idx + 1}/{num_maps}."
            
            await self._tool_provider.talk_to_agent(feedback, endpoint, new_conversation=False)
            await self._send_map_artifacts(updater, result_entry)

        # Send completion message to agent
        avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0
        await self._tool_provider.talk_to_agent(f"Evaluation complete. Avg: {avg_score:.1f}", endpoint, new_conversation=False)
        return results

    async def _send_map_artifacts(self, updater: TaskUpdater, entry: dict[str, Any]) -> None:
        """Send individual artifacts for human review."""
        role = entry["role"]
        idx = entry["map_index"]
        
        # Text Artifact (Map + Score)
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=f"{role} map {idx}:\n```ascii\n{entry['map']}\n```")),
                Part(root=TextPart(text=f"Score Data: {json.dumps(entry, indent=2)}")),
            ],
            name=f"{role}-map-{idx}",
        )
        
        # Video Artifact
        if entry.get("video_base64"):
            await updater.add_artifact(
                parts=[Part(root=DataPart(mime_type="video/mp4", data={"base64": entry["video_base64"]}))],
                name=f"{role}-video-{idx}",
            )

    async def _send_leaderboard_artifact(self, updater: TaskUpdater, all_results: list[dict[str, Any]], participants: dict[str, Any]) -> None:
        """Send the unified JSON artifact for the leaderboard."""
        valid_participants = {k: str(v) for k, v in participants.items()}
        await updater.add_artifact(
            parts=[Part(root=DataPart(mime_type="application/json", data={
                "participants": valid_participants,
                "results": all_results
            }))],
            name="results"
        )

    # --- Helper Methods ---

    def _setup_logging(self) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file_path = log_dir / f"mario_eval_log_{timestamp}.txt"
        self._write_log(f"=== Session Started at {datetime.now().isoformat()} ===")

    def _build_map_request_prompt(self, map_num: int, total: int) -> str:
        return f"""# Map Generation Request
Task: Generate map {map_num}/{total}
## Instructions
1. Return ONLY the ASCII map in a ```ascii code block
2. Include 'M' (start) and 'F' (exit flag)
3. Make each row 70-90 characters wide
4. All rows must be exactly the same length (pad with '-')
5. Make the map playable within 60 seconds

## ASCII Tile Reference
{MAP_ASCII_GUIDE}

## Response Format
```ascii
[Your map here]
```"""

    def _extract_ascii_map(self, response: str) -> str | None:
        match = CODE_BLOCK_RE.search(response)
        if not match:
            return None
        content = match.group(1).strip("\n")
        lines = [line.rstrip() for line in content.splitlines() if line.strip()]
        if not lines:
            return None
        width = max(len(line) for line in lines)
        return "\n".join([line.ljust(width, ".") for line in lines])

    def _run_astar_and_record(self, ascii_map: str, output_dir: str, output_name: str, idx: int) -> str | None:
        jar_path = Path("scenarios/mario/PlayAstar.jar")
        if not jar_path.exists():
            jar_path = Path("PlayAstar.jar")
        
        if not jar_path.exists():
            return None

        # Save map file
        map_filename = f"map{datetime.now().strftime('%Y%m%d')}_{idx}.txt"
        map_path = jar_path.parent / map_filename
        map_path.write_text(ascii_map, encoding="utf-8")

        # Config paths
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        video_file = out_path / output_name
        
        assets_path = Path("scenarios/mario/img/")
        if not assets_path.exists(): assets_path = Path("img")
        assets_arg = str(assets_path.resolve()).rstrip("/\\") + "/"

        cmd = [
            "java", "-Djava.awt.headless=true", "-jar", jar_path.name,
            map_filename, "human", assets_arg, str(out_path), output_name
        ]

        try:
            subprocess.run(cmd, check=True, timeout=120, cwd=jar_path.parent)
            if video_file.exists() and video_file.stat().st_size > 0:
                return str(video_file)
        except Exception as e:
            logger.error(f"PlayAstar failed: {e}")
        return None

    def _initialize_cached_prompt(self, video_path_str: str, score: float) -> None:
        """Initialize prompt cache with reference video."""
        ref_text = REFERENCE_CONTEXT.replace("[REFERENCE_SCORE]", str(int(score)))
        video_path = Path(video_path_str)
        
        if not video_path.exists():
            logger.warning("Reference video not found.")
            # Create a fallback cache without video bytes
            self._cached_prompt = {
                "system_prompt": SYSTEM_PROMPT,
                "reference_context": ref_text,
                "reference_video_path": video_path_str,
                "reference_video_bytes": None,
                "initial_response": "OK (fallback)",
                "reference_score": score,
            }
            return

        try:
            # Send initial warmup request
            video_bytes = video_path.read_bytes()
            content = [
                genai_types.Part(text=ref_text),
                genai_types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
            ]
            
            resp = self._client.models.generate_content(
                model="gemini-2.5-pro",
                contents=content,
                config=genai_types.GenerateContentConfig(
                    system_instruction="If you understand the scoring format, respond with: OK",
                    temperature=0.0
                )
            )
            
            self._cached_prompt = {
                "system_prompt": SYSTEM_PROMPT,
                "reference_context": ref_text,
                "reference_video_path": video_path_str,
                "reference_video_bytes": video_bytes,
                "initial_response": resp.text or "OK",
                "reference_score": score,
            }
            logger.info("Prompt cache initialized.")
            
        except Exception as e:
            logger.error(f"Failed to init prompt cache: {e}")

    def _evaluate_map(self, ascii_map: str, video_path: str | None) -> dict[str, Any]:
        """Evaluate using cached prompt."""
        if not self._cached_prompt:
            logger.warning("Cache missing, using default score.")
            return {"score": 1, "explain": "System error: Cache not initialized"}

        if not video_path:
            return {"score": 1, "explain": "Evaluation failed: No video"}

        try:
            content = [
                genai_types.Part(text=self._cached_prompt["reference_context"]),
            ]
            if self._cached_prompt["reference_video_bytes"]:
                content.append(genai_types.Part.from_bytes(
                    data=self._cached_prompt["reference_video_bytes"], mime_type="video/mp4"))
            
            content.append(genai_types.Part(text=f"Previous: {self._cached_prompt['initial_response']}"))
            
            # Current map data
            content.append(genai_types.Part.from_bytes(
                data=Path(video_path).read_bytes(), mime_type="video/mp4"))
            
            resp = self._client.models.generate_content(
                model="gemini-2.5-pro",
                contents=content,
                config=genai_types.GenerateContentConfig(
                    system_instruction=self._cached_prompt["system_prompt"],
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            return json.loads(resp.text) if resp.text else {"score": 1}
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"score": 1, "explain": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Mario Judge Agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--card-url", type=str)
    parser.add_argument("--cloudflare-quick-tunnel", action="store_true")
    parser.add_argument("--name", default="MarioJudge")
    args = parser.parse_args()

    if args.cloudflare_quick_tunnel:
        if quick_tunnel is None: raise RuntimeError("cloudflared missing")
        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(args.card_url or f"http://{args.host}:{args.port}/")

    async with agent_url_cm as agent_url:
        agent = MarioJudge()
        executor = GreenExecutor(agent)
        card = mario_judge_agent_card(args.name, agent_url)
        
        server = uvicorn.Server(uvicorn.Config(
            A2AStarletteApplication(
                agent_card=card, 
                http_handler=DefaultRequestHandler(executor, InMemoryTaskStore())
            ).build(), 
            host=args.host, port=args.port
        ))
        await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
