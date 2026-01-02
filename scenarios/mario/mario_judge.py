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

try:
    SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    SYSTEM_PROMPT = "You are an expert Mario level judge."

try:
    REFERENCE_CONTEXT = REFERENCE_CONTEXT_PATH.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    REFERENCE_CONTEXT = ""

try:
    MAP_ASCII_GUIDE = MAP_ASCII_GUIDE_PATH.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    MAP_ASCII_GUIDE = ""


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
    
    Workflow:
    1. Initialize reference cache with a reference video and score
    2. Request maps from purple agents one at a time
    3. For each map:
       - Validate map structure
       - Run PlayAstar.jar to generate gameplay video
       - Send map ASCII + video to LLM for evaluation
    4. Aggregate and report results
    """
    
    # Number of maps to request from each purple agent
    NUM_MAPS = 25
    
    # Reference video for establishing evaluation baseline
    REFERENCE_VIDEO_PATH = PROMPTS_DIR / "initial_criterion_video.mp4"
    
    # Reference score (1-20 scale as per prompt format)
    REFERENCE_SCORE = 12.0

    def __init__(self):
        self._tool_provider = ToolProvider()
        self._client = genai.Client()
        # Cached prompt: system_prompt + reference_context + video + LLM's OK response
        self._cached_prompt: str | None = None
        # Log file for recording all transactions in this session
        self._log_file_path: Path | None = None

    def _write_log(self, content: str) -> None:
        """Write content to session log file."""
        if self._log_file_path:
            try:
                with open(self._log_file_path, "a", encoding="utf-8") as f:
                    f.write(content)
                    f.write("\n")
            except Exception as exc:
                logger.error(f"Failed to write to log file: {exc}")

    async def _update_status_with_log(self, updater: TaskUpdater, message: str) -> None:
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
        """
        Main evaluation entry point.
        
        Args:
            req: Contains participant info and configuration
            updater: TaskUpdater for sending status updates to platform
        """
        # Initialize log file for this evaluation session
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file_path = log_dir / f"mario_eval_log_{timestamp}.txt"
        self._write_log(f"=== Mario Evaluation Session Started at {datetime.now().isoformat()} ===\n")
        
        # Extract configuration
        num_maps = int(req.config.get("num_maps", self.NUM_MAPS))
        jar_output_dir = str(req.config.get("jar_output_dir", "./"))
        jar_output_template = req.config.get(
            "jar_output_name_template", "{role}_gameplay_{ts}_{map_idx}.mp4")

        # Reference map video and score paths from config or defaults
        reference_video_path = req.config.get(
            "reference_video_path", self.REFERENCE_VIDEO_PATH)
        reference_score = float(req.config.get("reference_score", self.REFERENCE_SCORE))
        try:
            # Step 0: Initialize cached prompt with reference video
            await self._update_status_with_log(updater, "Initializing cached prompt...")
            self._initialize_cached_prompt(reference_video_path, reference_score)
            
            # To store ALL results for the final leaderboard artifact
            all_leaderboard_results: list[dict[str, Any]] = []

            # Process each participant
            for idx, (role, endpoint) in enumerate(req.participants.items(), start=1):
                map_evaluations: list[dict[str, Any]] = []

                for map_idx in range(1, num_maps + 1):
                    if map_idx == 1:
                        await self._update_status_with_log(updater, f"Requesting map {map_idx}/{num_maps} from {role}...")
                        prompt = self._build_map_request_prompt(map_idx, num_maps)
                        response = await self._tool_provider.talk_to_agent(prompt, str(endpoint), new_conversation=True)
                    else:
                        await self._update_status_with_log(updater, f"Receiving map {map_idx}/{num_maps} from {role}...")
                        response = await self._tool_provider.talk_to_agent("", str(endpoint), new_conversation=False)

                    ascii_map = self._extract_ascii_map(response)
                    if not ascii_map: continue

                    # Run simulation and evaluation
                    output_name = jar_output_template.format(role=role, ts=int(time.time()), map_idx=map_idx)
                    video_path = self._run_astar_and_record(ascii_map, jar_output_dir, output_name, map_idx)
                    score_data = self._evaluate_map(ascii_map, video_path)
                    
                    map_evaluations.append({
                        "role": role, "map_index": map_idx, "map": ascii_map,
                        "score_data": score_data, "video": video_path,
                    })
                    
                    all_leaderboard_results.append({
                        "role": role,
                        "map_index": map_idx,
                        **score_data
                    })

                    # Feedback
                    feedback = f"Map {map_idx} score: {score_data.get('score', 0)}/20"
                    if map_idx < num_maps: feedback += f"\n\nPlease generate map {map_idx+1}/{num_maps}."
                    await self._tool_provider.talk_to_agent(feedback, str(endpoint), new_conversation=False)

                # Send final summary to agent
                avg_score = sum(e["score_data"].get("score", 0) for e in map_evaluations) / len(map_evaluations) if map_evaluations else 0
                await self._tool_provider.talk_to_agent(f"Evaluation finished. Avg: {avg_score:.1f}", str(endpoint), new_conversation=False)

                # Add artifacts for this participant
                for e in map_evaluations:
                    await updater.add_artifact(
                        parts=[
                            Part(root=TextPart(text=f"{role} map {e['map_index']}:\n```ascii\n{e['map']}\n```")),
                            Part(root=TextPart(text=f"Score: {e['score_data']}")),
                        ],
                        name=f"{role}-map-{e['map_index']}",
                    )
                    if e["video"] and Path(e["video"]).exists():
                        video_bytes = Path(e["video"]).read_bytes()
                        await updater.add_artifact(
                            parts=[Part(root=DataPart(mime_type="video/mp4", data={"base64": base64.b64encode(video_bytes).decode("utf-8")}))],
                            name=f"{role}-video-{e['map_index']}",
                        )

            # --- FINAL STEP: Prepare and send the unified results artifact for the Leaderboard ---
            # Group results by role to calculate averages for all numeric fields
            final_submission_data = {
                "participants": {role: str(url) for role, url in req.participants.items()},
                "per_agent_results": {}
            }

            for role in req.participants.keys():
                role_evals = [e for e in all_leaderboard_results if e["role"] == role]
                if not role_evals:
                    continue
                
                # Identify all numeric keys (excluding map_index)
                numeric_keys = set()
                for e in role_evals:
                    for k, v in e.items():
                        if isinstance(v, (int, float)) and k != "map_index":
                            numeric_keys.add(k)
                
                # Calculate averages for each numeric key
                averages = {}
                for k in numeric_keys:
                    values = [e[k] for e in role_evals if k in e]
                    if values:
                        averages[f"avg_{k}"] = sum(values) / len(values)
                
                # Store both detailed results and pre-calculated averages
                final_submission_data["per_agent_results"][role] = {
                    "averages": averages,
                    "map_details": role_evals
                }

            # Also provide a flat list of all evaluations for the standard unnest query
            # (Adding a 'results' key at the top level for backward compatibility with the query)
            final_submission_data["results"] = all_leaderboard_results

            await updater.add_artifact(
                parts=[
                    Part(root=DataPart(
                        mime_type="application/json",
                        data=final_submission_data
                    ))
                ],
                name="results"
            )
        finally:
            self._tool_provider.reset()

        await self._update_status_with_log(
            updater, "All map evaluations completed.")
        
        # Finalize log
        self._write_log(f"\n=== Mario Evaluation Session Ended at {datetime.now().isoformat()} ===")

    def _build_map_request_prompt(self, map_number: int, total_maps: int) -> str:
        """
        Build map request prompt with ASCII guide.
        
        Args:
            map_number: Current map number (1-indexed)
            total_maps: Total number of maps to generate
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""# Map Generation Request

Task: Generate map {map_number}/{total_maps}

## Instructions

Generate a single Mario ASCII map in a fenced code block. Follow these guidelines:

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
```
"""
        return prompt

    def _extract_ascii_map(self, response: str) -> str:
        """
        Extract ASCII map from response text.
        
        Looks for code blocks (```ascii or ```text) and normalizes row lengths.
        
        Args:
            response: Agent response text containing map
            
        Returns:
            Normalized ASCII map string
            
        Raises:
            ValueError: If no code block found or map is empty
        """
        match = CODE_BLOCK_RE.search(response)
        if not match:
            raise ValueError("Map response missing ascii code block.")
        content = match.group(1).strip("\n")
        lines = [line.rstrip()
                 for line in content.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Empty map submitted.")
        # Normalize all rows to same width by padding with '.'
        width = max(len(line) for line in lines)
        normalized = [line.ljust(width, ".") for line in lines]
        return "\n".join(normalized)

    def _validate_map_row_lengths(self, ascii_map: str) -> bool:
        """
        Validate that all rows in the map have the same length.
        
        Args:
            ascii_map: ASCII map string
            
        Returns:
            True if all rows have equal length, False otherwise
        """
        lines = ascii_map.splitlines()
        if not lines:
            return False
        first_length = len(lines[0])
        return all(len(line) == first_length for line in lines)

    def _evaluate_map(
        self,
        ascii_map: str,
        video_path: str | None,
    ) -> dict[str, Any]:
        """
        Evaluate a map using cached prompt + gameplay video.
        
        Args:
            ascii_map: ASCII representation of the map
            video_path: Path to gameplay video
            
        Returns:
            Dictionary with evaluation results matching prompt format
        """
        return self._llm_judge_map(ascii_map, video_path)

    def _initialize_cached_prompt(
        self,
        reference_video_path: str,
        reference_score: float,
    ) -> None:
        """
        Initialize cached prompt with reference evaluation.
        
        Step 1: Send system_prompt + reference_context + reference_video to LLM
        Step 2: LLM responds with "OK" (initial response)
        Step 3: Cache = system_prompt + reference_context + reference_video + LLM's OK response
        
        Args:
            reference_video_path: Path to reference gameplay video
            reference_score: Score (1-20) for the reference map
        """
        # Build initial prompt with reference context
        reference_text = REFERENCE_CONTEXT.replace("[REFERENCE_SCORE]", str(int(reference_score)))
        
        content_parts: list[genai_types.Part] = [
            genai_types.Part(text=reference_text),
        ]

        # Attach reference video
        try:
            video_path = Path(reference_video_path)
            if video_path.exists():
                video_bytes = video_path.read_bytes()
                content_parts.append(
                    genai_types.Part.from_bytes(
                        data=video_bytes, mime_type="video/mp4")
                )
            else:
                logger.warning(f"Reference video not found: {reference_video_path}")
                return
        except Exception as exc:
            logger.warning(f"Failed to load reference video: {exc}")
            return

        # Log LLM request
        self._write_log(f"\n--- LLM Request (Cached Prompt Init) at {datetime.now().isoformat()} ---")
        self._write_log(f"System Prompt:\n{SYSTEM_PROMPT}")
        self._write_log(f"Reference Context:\n{reference_text}")
        self._write_log(f"Reference Video: {reference_video_path}")

        # Send to LLM to get "OK" response
        # Use a simple instruction that just asks for acknowledgment
        acknowledgment_instruction = (
            "You have been shown a reference evaluation example. "
            "If you understand the scoring format and criteria, respond with only: OK"
        )
        try:
            response = self._client.models.generate_content(
                model="gemini-2.5-pro",
                contents=content_parts,
                config=genai_types.GenerateContentConfig(
                    system_instruction=acknowledgment_instruction,
                    temperature=0.0,
                    # No response_mime_type to allow plain text "OK"
                ),
            )
            
            if response.text:
                initial_response = response.text
                
                # Log LLM response
                self._write_log(f"\n--- LLM Response (Cached Prompt Init) at {datetime.now().isoformat()} ---")
                self._write_log(f"Initial Response:\n{initial_response}")
                
                # Store cached prompt components
                # Cached prompt = system + reference + video + initial_response
                self._cached_prompt = {
                    "system_prompt": SYSTEM_PROMPT,
                    "reference_context": reference_text,
                    "reference_video_path": reference_video_path,
                    "reference_video_bytes": video_bytes,
                    "initial_response": initial_response,
                    "reference_score": reference_score,
                }
                
                logger.info(f"Cached prompt initialized successfully")
        except Exception as exc:
            logger.warning(f"Failed to initialize cached prompt: {exc}")
            self._write_log(f"\n--- LLM Error (Cached Prompt Init) at {datetime.now().isoformat()} ---")
            self._write_log(f"Error: {exc}")
            self._cached_prompt = None

    def _llm_judge_map(
        self,
        ascii_map: str,
        video_path: str | None,
    ) -> dict[str, Any]:
        """
        Evaluate a map using cached prompt + gameplay video.
        
        IMPORTANT: Only send cached_prompt + current_video to LLM.
        Do NOT include previous evaluation results.
        
        Args:
            ascii_map: ASCII representation of the map
            video_path: Path to gameplay video
            
        Returns:
            Dictionary matching prompt format with scores and feedback
        """
        # Fallback: if cached prompt has not been initialized, populate it
        # on-the-fly with reasonable defaults and emit a warning instead of failing.
        if not self._cached_prompt:
            logger.warning(
                "Cached prompt was not initialized via _initialize_cached_prompt; "
                "using fallback cached prompt without prior LLM warmup."
            )
            self._write_log(
                "\n--- Cached Prompt Fallback ---\n"
                "Warning: Cached prompt was missing. "
                "Using REFERENCE_CONTEXT and REFERENCE_VIDEO_PATH directly without OK warmup.\n"
            )

            reference_score = getattr(self, "REFERENCE_SCORE", 12.0)
            reference_video_path = str(
                getattr(self, "REFERENCE_VIDEO_PATH", type(self).REFERENCE_VIDEO_PATH)
            )

            # Build reference text from REFERENCE_CONTEXT
            reference_text = REFERENCE_CONTEXT.replace(
                "[REFERENCE_SCORE]", str(int(reference_score))
            )

            # Try to load reference video bytes if available
            video_bytes = None
            try:
                ref_video_path_obj = Path(reference_video_path)
                if ref_video_path_obj.exists():
                    video_bytes = ref_video_path_obj.read_bytes()
                else:
                    logger.warning(
                        f"Fallback cached prompt: reference video not found at {reference_video_path}"
                    )
                    self._write_log(
                        f"Fallback cached prompt: reference video not found at {reference_video_path}"
                    )
            except Exception as exc:
                logger.warning(
                    f"Fallback cached prompt: failed to load reference video: {exc}"
                )
                self._write_log(
                    f"Fallback cached prompt: failed to load reference video: {exc}"
                )

            # Manually construct a minimal cached prompt
            self._cached_prompt = {
                "system_prompt": SYSTEM_PROMPT,
                "reference_context": reference_text,
                "reference_video_path": reference_video_path,
                "reference_video_bytes": video_bytes,
                "initial_response": "OK (fallback: no warmup)",
                "reference_score": reference_score,
            }
        
        if not video_path or not Path(video_path).exists():
            logger.error(f"Video not found: {video_path}")
            return {"explain": "Evaluation failed: No video", "result": {}, "score": 1}

        # Build content with cached context + current video
        # Include: reference_context + reference_video + LLM's OK + current_video
        content_parts: list[genai_types.Part] = []
        
        # 1. Add reference context text
        content_parts.append(
            genai_types.Part(text=self._cached_prompt["reference_context"])
        )
        
        # 2. Add reference video if available
        ref_video_bytes = self._cached_prompt.get("reference_video_bytes")
        if ref_video_bytes:
            content_parts.append(
                genai_types.Part.from_bytes(
                    data=ref_video_bytes,
                    mime_type="video/mp4",
                )
            )
        else:
            logger.warning(
                "Cached prompt has no reference_video_bytes; proceeding without reference video."
            )
            self._write_log(
                "Cached prompt has no reference_video_bytes; proceeding without reference video."
            )
        
        # 3. Add LLM's initial OK response
        content_parts.append(
            genai_types.Part(text=f"Previous response: {self._cached_prompt['initial_response']}")
        )
        
        # 4. Add current map video
        try:
            video_bytes = Path(video_path).read_bytes()
            content_parts.append(
                genai_types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
            )
        except Exception as exc:
            logger.error(f"Failed to load video: {exc}")
            return {"explain": "Evaluation failed: Video load error", "result": {}, "score": 1}

        # Log LLM request
        self._write_log(f"\n--- LLM Request (Map Evaluation) at {datetime.now().isoformat()} ---")
        self._write_log(f"Contents:")
        self._write_log(f"  1. Reference context (text)")
        self._write_log(f"  2. Reference video: {self._cached_prompt['reference_video_path']}")
        self._write_log(f"  3. Previous LLM response: {self._cached_prompt['initial_response'][:50]}...")
        self._write_log(f"  4. Current video: {video_path}")
        self._write_log(f"Map ASCII:\n{ascii_map}")

        # Call LLM with cached prompt (system + reference + initial response)
        try:
            response = self._client.models.generate_content(
                model="gemini-2.5-pro",
                contents=content_parts,
                config=genai_types.GenerateContentConfig(
                    system_instruction=self._cached_prompt["system_prompt"],
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            
            if response.text:
                parsed = json.loads(response.text)
                if isinstance(parsed, dict):
                    final_score = parsed.get("score", 0)
                    
                    # Log LLM response
                    self._write_log(f"\n--- LLM Response (Map Evaluation) at {datetime.now().isoformat()} ---")
                    self._write_log(f"Score: {final_score}/20")
                    self._write_log(f"Full Response:\n{json.dumps(parsed, indent=2)}")
                    
                    logger.info(f"LLM evaluation completed: score={final_score}/20")
                    return parsed
        except Exception as exc:
            logger.error(f"LLM evaluation failed: {exc}")
            self._write_log(f"\n--- LLM Error (Map Evaluation) at {datetime.now().isoformat()} ---")
            self._write_log(f"Error: {exc}")
        
        return {"explain": "Evaluation failed", "result": {}, "score": 1}

    def _run_astar_and_record(
        self,
        ascii_map: str,
        jar_output_dir: str,
        output_name: str,
        map_index: int = 1,
    ) -> str | None:
        """
        Save map to file and run PlayAstar.jar to generate gameplay video.
        
        PlayAstar.jar simulates an AI Mario playing the map and records
        a video of the gameplay for up to 60 seconds.
        
        Args:
            ascii_map: ASCII representation of the map
            jar_output_dir: Directory to save output video
            output_name: Name of output video file
            map_index: Index of the map (for unique filenames)
            
        Returns:
            Path to generated video file, or None if execution failed
        """
        # Locate JAR file
        jar_path = Path("scenarios/mario/PlayAstar.jar")
        if not jar_path.exists():
            jar_path = Path("PlayAstar.jar")

        if not jar_path.exists():
            logger.warning(
                "PlayAstar.jar not found. Skipping execution and recording.")
            return None

        # Save map to text file
        map_filename = f"map{datetime.now().strftime('%Y%m%d')}_{map_index}.txt"
        map_path = jar_path.parent / map_filename
        map_path.write_text(ascii_map, encoding="utf-8")
        logger.info("Saved map to %s", map_path)

        # Prepare output directory
        output_dir = Path(jar_output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        gameplay_file = output_dir / output_name

        # Locate assets directory (contains Mario sprites)
        assets_path = Path("scenarios/mario/img/")
        if not assets_path.exists():
            assets_path = Path("img")
        assets_arg = assets_path.resolve().as_posix()
        if not assets_arg.endswith("/"):
            assets_arg += "/"

        # Build Java command
        # Usage: java -jar PlayAstar.jar <map_file> human <img_folder> <output_dir> <video_name>
        cmd = [
            "java",
            "-Djava.awt.headless=true",
            "-jar",
            jar_path.name,
            map_filename,
            "human",  # Visualization mode
            assets_arg,
            str(output_dir),
            output_name,
        ]

        logger.info("Running PlayAstar: %s", " ".join(cmd))
        try:
            # Run JAR with 120 second timeout
            subprocess.run(cmd, check=True, timeout=120, cwd=jar_path.parent)
        except FileNotFoundError as exc:
            logger.error("Error running PlayAstar (Java not found?): %s", exc)
            return None
        except subprocess.CalledProcessError as exc:
            logger.error("PlayAstar exited with error: %s", exc)
            return None
        except subprocess.TimeoutExpired:
            logger.error("PlayAstar timed out after 120 seconds")
            return None

        # Verify video was created
        if gameplay_file.exists() and gameplay_file.stat().st_size > 0:
            logger.info(f"Gameplay video created: {gameplay_file}")
            return str(gameplay_file)
        logger.warning("Gameplay file not found after run: %s", gameplay_file)
        return None


async def main():
    parser = argparse.ArgumentParser(
        description="Run the Mario map evaluation green agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9100,
                        help="Port to bind the server")
    parser.add_argument("--card-url", type=str,
                        help="External URL to provide in the agent card")
    parser.add_argument(
        "--cloudflare-quick-tunnel",
        action="store_true",
        help="Use a Cloudflare quick tunnel. Requires cloudflared. Overrides --card-url.",
    )
    parser.add_argument("--name", type=str,
                        default="MarioJudge", help="Agent display name")
    args = parser.parse_args()

    if args.cloudflare_quick_tunnel:
        if quick_tunnel is None:
            raise RuntimeError(
                "cloudflared helpers are unavailable in this environment.")
        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(
            args.card_url or f"http://{args.host}:{args.port}/")

    async with agent_url_cm as agent_url:
        agent = MarioJudge()
        executor = GreenExecutor(agent)
        card = mario_judge_agent_card(args.name, agent_url)

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )
        app = A2AStarletteApplication(
            agent_card=card, http_handler=request_handler)
        server = uvicorn.Server(uvicorn.Config(
            app.build(), host=args.host, port=args.port))
        await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
