import argparse
import asyncio
import os
import contextlib
import json
import logging
import re
import time
import subprocess
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Import OpenRouter utils
import sys
# Add the current script's directory to sys.path to allow importing local utils
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from util_openrouter import call_openrouter, encode_video_to_base64

try:
    from agentbeats.cloudflare import quick_tunnel
except ImportError:  # pragma: no cover
    quick_tunnel = None


load_dotenv()

logger = logging.getLogger("mario_map_evaluator")
logging.basicConfig(level=logging.INFO)

CODE_BLOCK_RE = re.compile(
    r"```(?:\w+)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# --- Configuration & Resources ---
PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.md"
MAP_ASCII_GUIDE_PATH = PROMPTS_DIR / "map_ascii_guide.md"
MAP_REQUEST_PROMPT_PATH = PROMPTS_DIR / "map_request.md"

INITIAL_CRITERION_PROMPT_PATH = PROMPTS_DIR / "initial_criterion_prompt.md"
INITIAL_CRITERION_VIDEO_PATH = PROMPTS_DIR / "initial_criterion_video.mp4"


def load_text_resource(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default


SYSTEM_PROMPT = load_text_resource(SYSTEM_PROMPT_PATH, "You are an expert Mario level judge.")
MAP_ASCII_GUIDE = load_text_resource(MAP_ASCII_GUIDE_PATH)
MAP_REQUEST_PROMPT = load_text_resource(MAP_REQUEST_PROMPT_PATH)
INITIAL_CRITERION_PROMPT = load_text_resource(INITIAL_CRITERION_PROMPT_PATH)


def mario_map_evaluator_agent_card(name: str, url: str) -> AgentCard:
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


class MarioMapEvaluator(GreenAgent):
    """
    Green agent that evaluates Mario maps submitted by purple agents.
    Orchestrates the loop of: Request -> Extract -> Simulate -> Evaluate -> Trace.
    """
    DEFAULT_NUM_MAPS = 25
    MAX_SCORE = 20.0
    TOP_K = 5
    CATEGORIES = [
        "composition", "probability", "completeness", "aesthetics",
        "originality", "fairness", "fun", "difficulty"
    ]

    def __init__(self):
        self._client = genai.Client()
        self._tool_provider = ToolProvider()
        self._base_parts: List[genai_types.Part] = []

    # --- Core Lifecycle Methods ---

    def validate_request(self, request: EvalRequest) -> Tuple[bool, str]:
        if "agent" not in request.participants:
            return False, "Participant 'agent' is required."
        return True, "ok"

    def _get_failure_score_data(self, reason: str) -> Dict[str, Any]:
        """Returns a standardized 1-point score result for failures."""
        result_details = {
            cat: {"score": 1, "reason": reason} for cat in self.CATEGORIES
        }
        return {
            "score": 1,
            "explain": reason,
            "result": result_details
        }

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        """Main execution entry point."""

        # 1. Parse Config
        num_maps = int(req.config.get("num_maps", self.DEFAULT_NUM_MAPS))
        top_k = int(req.config.get("top_k", self.TOP_K))

        # Initialize session timestamp for consistent naming
        self._session_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        jar_output_dir = req.config.get("jar_output_dir", "outputs")
        ref_video_path = INITIAL_CRITERION_VIDEO_PATH

        try:
            # 2. Init Resources
            await self._update_status(updater, "Initializing base prompt parts...")
            self._initialize_base_parts(str(ref_video_path))

            # 3. Execution Loop
            role = "agent"
            endpoint = str(req.participants["agent"])
            await self._update_status(updater, f"Starting evaluation for {role} ({num_maps} maps)...")

            all_results = []
            participant_results = await self._evaluate_participant_loop(
                role=role,
                endpoint=endpoint,
                num_maps=num_maps,
                jar_output_dir=jar_output_dir,
                updater=updater
            )

            # 4. Aggregation
            aggregated_result = self._aggregate_results(participant_results, k=top_k)

            # 5. Reporting
            await self._send_leaderboard_artifact(updater, aggregated_result)

            # Save final aggregate results to JSON
            out_path = Path(jar_output_dir).resolve()
            out_path.mkdir(parents=True, exist_ok=True)
            results_file = out_path / f"{self._session_ts}_total_result.json"
            results_file.write_text(json.dumps(aggregated_result, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Final aggregated results saved to {results_file}")

        finally:
            self._tool_provider.reset()
            await self._update_status(updater, "Evaluation session completed.")

    async def _evaluate_participant_loop(
        self,
        role: str,
        endpoint: str,
        num_maps: int,
        jar_output_dir: str,
        updater: TaskUpdater
    ) -> List[Dict[str, Any]]:
        """Iterates through N maps for a single participant."""
        results = []
        next_prompt = self._build_map_request_prompt(1, num_maps)

        for map_idx in range(1, num_maps + 1):
            result_entry = await self._process_single_map(
                role=role,
                endpoint=endpoint,
                map_idx=map_idx,
                total_maps=num_maps,
                prompt=next_prompt,
                jar_output_dir=jar_output_dir,
                updater=updater
            )
            if result_entry:
                results.append(result_entry)
                # Prepare next prompt for the next iteration
                if map_idx < num_maps:
                    feedback = (
                        f"### Evaluation for Map {map_idx}/{num_maps}\n"
                        f"- **Score**: {result_entry['score']}/20\n"
                        f"- **Feedback**: {result_entry['explain']}\n\n"
                    )
                    # For subsequent maps, we don't need to resend the full ASCII guide
                    next_prompt = feedback + self._build_map_request_prompt(map_idx + 1, num_maps, include_guide=False)

        # Final session summary to agent
        avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
        await self._tool_provider.talk_to_agent(
            f"Evaluation complete. Average Score: {avg_score:.1f}/20",
            endpoint,
            new_conversation=False
        )
        return results

    async def _process_single_map(
        self,
        role: str,
        endpoint: str,
        map_idx: int,
        total_maps: int,
        prompt: str,
        jar_output_dir: str,
        updater: TaskUpdater
    ) -> Optional[Dict[str, Any]]:
        """Handles the lifecycle of a single map evaluation."""

        # 1. Communication (Request/Receive Map)
        is_new_conv = (map_idx == 1)

        action_verb = "Requesting" if map_idx == 1 else "Receiving"
        await self._update_status(updater, f"{action_verb} map {map_idx}/{total_maps} from {role}...")

        ascii_map = ""
        video_path = None
        score_data = {}

        try:
            response = await self._tool_provider.talk_to_agent(prompt, endpoint, new_conversation=is_new_conv)
            # 2. Extraction
            ascii_map = self._extract_ascii_map(response)
            if not ascii_map:
                logger.warning(f"Map {map_idx}: Failed to extract ASCII map from response.")
                score_data = self._get_failure_score_data("Failed to extract ASCII map from agent response.")
                ascii_map = "" # Placeholder
                video_path = None
            else:
                # 3. Simulation
                await self._update_status(updater, f"Simulating map {map_idx}/{total_maps}...")
                output_name = f"{self._session_ts}_{map_idx}_video.mp4"
                video_path = self._run_astar_and_record(ascii_map, jar_output_dir, output_name, map_idx)
                
                # 4. Evaluation (LLM)
                score_data = self._evaluate_map_content(ascii_map, video_path)

        except Exception as e:
            logger.error(f"Failed to process map {map_idx}: {e}")
            score_data = self._get_failure_score_data(f"Request failed or timed out: {str(e)}")
            ascii_map = ""
            video_path = None

        # 5. Data Assembly
        # We no longer include base64 video in the JSON to keep it lightweight.
        video_base64 = None

        # Parse task rewards
        llm_result = score_data.get("result", {})
        task_rewards = {}
        for k, v in llm_result.items():
            if isinstance(v, dict):
                task_rewards[k] = v.get("score")
                task_rewards[f"{k}_reason"] = v.get("reason", "")
            else:
                task_rewards[k] = v

        current_score = float(score_data.get("score", 0))

        result_entry = {
            "domain": "mario",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "score": current_score,
            "max_score": self.MAX_SCORE,
            "task_rewards": task_rewards,
            "time_used": 0.0,
            "role": role,
            "map_index": map_idx,
            "map": ascii_map,
            "video_base64": video_base64,
            "explain": score_data.get("explain", "")
        }

        # Save individual result to JSON (incremental)
        try:
            out_path = Path(jar_output_dir).resolve()
            out_path.mkdir(parents=True, exist_ok=True)
            indiv_file = out_path / f"{self._session_ts}_{map_idx}_result.json"
            # We exclude video_base64 for individual JSON to keep them lightweight, 
            # but keep it if you need complete individual records.
            indiv_file.write_text(json.dumps(result_entry, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Map {map_idx} result saved to {indiv_file}")
        except Exception as e:
            logger.warning(f"Failed to save individual JSON for map {map_idx}: {e}")

        # 6. Feedback & Artifacts
        # Send artifacts to UI
        await self._send_map_artifacts(updater, result_entry)

        return result_entry

    # --- Aggregation Logic ---

    def _aggregate_results(self, results: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
        """Aggregate results using Top-K averaging."""
        if not results:
            return {
                "domain": "mario",
                "eval_temperature": float(os.getenv("TEMPERATURE", "0.0")),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "map_count": 0,
                "k": k,
                "score": 0.0,
                "max_score": self.MAX_SCORE,
                "task_rewards": {},
                "history": []
            }

        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

        # Top-K
        actual_k = min(k, len(sorted_results))
        top_k_items = sorted_results[:actual_k]

        # Compute Averages
        avg_score = sum(r.get("score", 0) for r in top_k_items) / actual_k

        avg_task_rewards = {}
        if top_k_items:
            # We assume all results obey the same schema
            first_rewards = top_k_items[0].get("task_rewards", {})
            for key in first_rewards:
                try:
                    total_val = sum(float(item.get("task_rewards", {}).get(key, 0)) for item in top_k_items)
                    avg_task_rewards[key] = total_val / actual_k
                except (ValueError, TypeError):
                    # Skip non-numeric fields like 'fun_reason'
                    continue

        return {
            "domain": "mario",
            "eval_temperature": float(os.getenv("TEMPERATURE", "0.0")),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "map_count": len(results),
            "k": actual_k,
            "score": avg_score,
            "max_score": self.MAX_SCORE,
            "task_rewards": avg_task_rewards,
            "history": results
        }

    # --- Helpers: Logging & Artifacts ---

    async def _update_status(self, updater: TaskUpdater, message: str) -> None:
        logger.info(message)
        await updater.update_status(TaskState.working, new_agent_text_message(message))

    async def _send_map_artifacts(self, updater: TaskUpdater, entry: Dict[str, Any]) -> None:
        role = entry["role"]
        idx = entry["map_index"]
        # Exclude bulky base64 from logs/UI
        display_data = {k: v for k, v in entry.items() if k != "video_base64"}

        # GitHub Actions log grouping
        print(f"\n::group::{role} Map {idx} Detailed Data")
        print(json.dumps(display_data, indent=2))
        print("::endgroup::")

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(
                    text=f"{role} map {idx}:\n```ascii\n{entry['map']}\n```")),
                Part(root=TextPart(
                    text=f"Score Data: {json.dumps(display_data, indent=2)}")),
            ],
            name=f"{role}-map-{idx}",
        )

    async def _send_leaderboard_artifact(self, updater: TaskUpdater, result: Dict[str, Any]) -> None:
        await updater.add_artifact(
            parts=[Part(root=DataPart(data=result))],
            name="Results"
        )

    # --- Helpers: Core Logic ---

    def _build_map_request_prompt(self, map_num: int, total: int, include_guide: bool = True) -> str:
        guide = MAP_ASCII_GUIDE if include_guide else "*(Refer to the ASCII tile reference provided in the first message)*"
        return MAP_REQUEST_PROMPT.format(
            map_num=map_num,
            total=total,
            map_ascii_guide=guide
        )

    def _extract_code_block(self, text: str) -> str:
        """Extracts content from a markdown code block if present, otherwise returns original text."""
        text = text.strip()
        match = CODE_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()
        return text

    def _extract_ascii_map(self, response: str) -> Optional[str]:
        content = self._extract_code_block(response)
        if content == response and not response.startswith("```"):
            # If no code block found and it didn't look like one, 
            # we might still want to try parsing if the whole response is the map,
            # but usually we expect a block. For Mario maps, we'll be strict.
            if not any(char in response for char in "X#SF-"): 
                return None
        
        lines = [line.rstrip() for line in content.splitlines() if line.strip()]
        if not lines:
            return None

        width = max(len(line) for line in lines)
        return "\n".join([line.ljust(width, "-") for line in lines])

    def _run_astar_and_record(self, ascii_map: str, output_dir: str, output_name: str, idx: int) -> Optional[str]:
        # Locate JAR
        jar_path = Path("scenarios/mario/PlayAstar.jar")
        if not jar_path.exists():
            jar_path = Path("PlayAstar.jar")
        if not jar_path.exists():
            logger.error("PlayAstar.jar not found.")
            return None

        # Prepare output directory
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        # Write temp map file with requested suffix
        map_filename = output_name.replace("_video.mp4", "_map.txt")
        map_path = out_path / map_filename
        map_path.write_text(ascii_map, encoding="utf-8")

        # Prepare video path
        video_file = out_path / output_name

        assets_path = Path("scenarios/mario/img/")
        if not assets_path.exists():
            assets_path = Path("img")
        assets_arg = str(assets_path.resolve()).rstrip("/\\") + "/"

        cmd = [
            "java", "-Djava.awt.headless=true", "-jar", jar_path.name,
            str(map_path), "human", assets_arg, str(out_path), output_name
        ]

        # Execute
        try:
            subprocess.run(cmd, check=True, timeout=120, cwd=jar_path.parent)
            if video_file.exists() and video_file.stat().st_size > 0:
                return str(video_file)
        except Exception as e:
            logger.error(f"PlayAstar execution failed: {e}")
        return None

    def _initialize_base_parts(self, video_path_str: str) -> None:
        """Initializes the base prompt parts that define evaluation criteria and reference examples."""
        video_path = Path(video_path_str)
        self._base_parts = [genai_types.Part(text=INITIAL_CRITERION_PROMPT)]

        if not video_path.exists():
            logger.error(f"Reference video not found at: {video_path_str}")
        else:
            try:
                self._base_parts.append(genai_types.Part.from_bytes(
                    data=video_path.read_bytes(), 
                    mime_type="video/mp4"
                ))
            except Exception as e:
                logger.error(f"Failed to read reference video: {e}")

        self._base_parts.append(genai_types.Part(text="Previous Model Response: OK"))
        logger.info("Base prompt parts initialized.")

    def _evaluate_map_content(self, ascii_map: str, video_path: Optional[str]) -> Dict[str, Any]:
        """Calls LLM (Google or OpenRouter) to evaluate the generated map video against the reference."""
        if not self._base_parts:
            logger.error("Base prompt parts not initialized.")
            return {"score": 1, "explain": "System error: Base parts not initialized"}

        if not video_path:
            logger.warning("Evaluation failed: No video generated (simulation likely failed).")
            return self._get_failure_score_data("Simulation failed: No gameplay video generated.")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                if os.getenv("USE_OPEN_ROUTER", "").lower() == "true":
                    return self._evaluate_with_openrouter(video_path)
                else:
                    return self._evaluate_with_google(video_path)

            except Exception as e:
                logger.warning(f"Evaluation attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"LLM Evaluation failed after {max_retries} attempts.")
                    return self._get_failure_score_data(f"LLM Error after max retries: {str(e)}")
                time.sleep(2)  # Brief wait before retry

    def _evaluate_with_openrouter(self, video_path: str) -> Dict[str, Any]:
        """Evaluation path using OpenRouter API."""
        # Construct messages for OpenRouter
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # User Message Parts
        user_content_parts = []
        
        # Add base parts (text history)
        for part in self._base_parts:
            if part.text:
                user_content_parts.append({"type": "text", "text": part.text})
        
        # Add Video (Base64 encoded)
        video_data_url = encode_video_to_base64(video_path)
        user_content_parts.append({
            "type": "video_url",
            "video_url": {"url": video_data_url}
        })

        # Assemble User Message
        messages.append({"role": "user", "content": user_content_parts})

        # Get temperature from env, default to 0.0
        try:
            temp = float(os.getenv("TEMPERATURE", "0.0"))
        except ValueError:
            temp = 0.0

        logger.info(f"Calling OpenRouter with model: google/gemini-2.5-pro with {temp} temperature")

        resp_json = call_openrouter(
             model="google/gemini-2.5-pro",
             messages=messages,
             temperature=temp,
             response_format={"type": "json_object"},
             seed=0,
             top_p=0.,
             top_k=1,
        )
        
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
             raw_text = resp_json["choices"][0]["message"]["content"]
             return json.loads(self._extract_code_block(raw_text))
        else:
             raise ValueError(f"Invalid OpenRouter response: {resp_json}")

    def _evaluate_with_google(self, video_path: str) -> Dict[str, Any]:
        """Evaluation path using Google GenAI SDK directly."""
        # Simple assembly: Base content + Current video
        content = self._base_parts + [genai_types.Part.from_bytes(data=Path(video_path).read_bytes(), mime_type="video/mp4")]

        # Get temperature from env, default to 0.0
        try:
            temp = float(os.getenv("TEMPERATURE", "0.0"))
        except ValueError:
            temp = 0.0

        logger.info(f"Calling Google with model: gemini-2.5-pro with {temp} temperature")

        resp = self._client.models.generate_content(
            model="gemini-2.5-pro",
            contents=content,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=temp,
                seed=0,
                top_p=0.,
                top_k=1,

            )
        )

        if not resp.text:
            raise ValueError("Empty response from model.")

        raw_text = self._extract_code_block(resp.text)
        return json.loads(raw_text)


async def main():
    parser = argparse.ArgumentParser(description="Mario Map Evaluator Agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--card-url", type=str)
    parser.add_argument("--cloudflare-quick-tunnel", action="store_true")
    parser.add_argument("--name", default="MarioMapEvaluator")
    args = parser.parse_args()

    if args.cloudflare_quick_tunnel:
        if quick_tunnel is None:
            raise RuntimeError("cloudflared missing")
        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(
            args.card_url or f"http://{args.host}:{args.port}/")

    async with agent_url_cm as agent_url:
        agent = MarioMapEvaluator()
        executor = GreenExecutor(agent)
        card = mario_map_evaluator_agent_card(args.name, agent_url)

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=card,
            http_handler=request_handler,
        )

        uvicorn_config = uvicorn.Config(
            server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()

if __name__ == "__main__":
    asyncio.run(main())
