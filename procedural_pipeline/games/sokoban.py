import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PCG_ROOT = PROJECT_ROOT / "third_party" / "pcg_benchmark"
if str(PCG_ROOT) not in sys.path:
    sys.path.insert(0, str(PCG_ROOT))

from pcg_benchmark.probs.sokoban.problem import SokobanProblem  # type: ignore[reportMissingImports]  # noqa: E402


KEY = "sokoban"

DEFAULT_SOLVER_POWER = 500_000_000

DEFAULTS = {
    "maps_file": "scenarios/sokoban/map.json",
    "criteria_file": "scenarios/sokoban/evaluation.json",
    "output_dir": "outputs/procedural_sokoban",
    "profile_key": "sokoban",
}


ALLOWED_CHARS = {"#", "-", " ", "@", "+", "$", "*", "."}
CHAR_MAP = {
    "#": 0,
    "-": 1,
    " ": 1,
    "@": 2,
    "+": 2,
    "$": 3,
    "*": 3,
    ".": 4,
}


def add_arguments(parser) -> None:
    parser.add_argument("--solver-power", type=int, default=DEFAULT_SOLVER_POWER)


def load_maps(path: str) -> list[tuple[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.items())


def normalize_map(map_text: str) -> str:
    raw_lines = [line.rstrip("\r") for line in map_text.splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError("Sokoban map is empty.")

    width = max(len(line) for line in raw_lines)
    lines = []
    for line in raw_lines:
        normalized = []
        for char in line.ljust(width, "-"):
            if char not in ALLOWED_CHARS:
                normalized.append("-")
            elif char == " ":
                normalized.append("-")
            else:
                normalized.append(char)
        lines.append("".join(normalized))

    return "\n".join(lines)


def render_map(map_text: str, map_id: str, output_dir: str, args) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_path = out_dir / f"{map_id}.txt"
    video_path = out_dir / f"{map_id}.mp4"
    map_path.write_text(map_text, encoding="utf-8")

    solver_power = getattr(args, "solver_power", DEFAULT_SOLVER_POWER)
    frames = _render_solution_frames(map_text, map_id, solver_power)
    _write_video(frames, video_path)
    return map_path, video_path


def _render_solution_frames(map_text: str, map_id: str, solver_power: int) -> list[Image.Image]:
    content = _map_text_to_content(map_text)
    problem = SokobanProblem(
        width=content.shape[1],
        height=content.shape[0],
        difficulty=1,
        solver=solver_power,
    )
    info = problem.info(content)
    print(f"[sokoban] render_solution solver_power={solver_power}")
    frames, _ = problem.render_solution(info, map_name=map_id)
    if frames:
        return frames

    fallback = problem.render(content)
    return [fallback] * 12


def _map_text_to_content(map_text: str) -> np.ndarray:
    lines = [line.rstrip("\r") for line in map_text.splitlines() if line.strip()]
    height = len(lines)
    width = max(len(line) for line in lines)

    content = np.ones((height, width), dtype=int)
    for y, line in enumerate(lines):
        for x, char in enumerate(line.ljust(width, "-")):
            content[y][x] = CHAR_MAP.get(char, 1)
    return content


def _write_video(frames: list[Image.Image], video_path: Path, fps: int = 3) -> None:
    with imageio.get_writer(video_path, fps=fps, codec="libx264", macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))

    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Video was not created: {video_path}")
