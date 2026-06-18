import json
import subprocess
from pathlib import Path

from procedural_pipeline.paths import package_resource


KEY = "mario"

DEFAULTS = {
    "maps_file": str(package_resource("mario", "maps.json")),
    "criteria_file": str(package_resource("mario", "evaluation.json")),
    "output_dir": "outputs/procedural_mario",
    "profile_key": "mario",
}


TILE_REPLACEMENTS = str.maketrans({
    "{": "M",
    "}": "F",
    "?": "Q",
    "H": "X",
})

ALLOWED_TILES = set("MF- X#SCLUD%|?@Q!12otT<>[]*BbEgGkKrRyY")
MIN_ROWS = 16
DEFAULT_WIDTH = 64


def add_arguments(parser) -> None:
    return None


def load_maps(path: str) -> list[tuple[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.items())


def normalize_map(map_text: str) -> str:
    replaced = map_text.translate(TILE_REPLACEMENTS)
    raw_lines = [line.rstrip("\r") for line in replaced.splitlines() if line.strip()]
    if not raw_lines:
        raw_lines = ["-" * DEFAULT_WIDTH for _ in range(MIN_ROWS)]

    width = max(len(line) for line in raw_lines)
    raw_lines = _pad_top_to_min_rows(raw_lines, width)
    grid = [list(line.ljust(width, "-")) for line in raw_lines]
    _normalize_enemies(grid)

    lines = []
    for row in grid:
        lines.append("".join(ch if ch in ALLOWED_TILES else "-" for ch in row))

    _ensure_start_and_flag(lines)
    return "\n".join(lines)


def _pad_top_to_min_rows(lines: list[str], width: int) -> list[str]:
    if len(lines) >= MIN_ROWS:
        return lines
    top_padding = ["-" * width for _ in range(MIN_ROWS - len(lines))]
    return top_padding + lines


def render_map(map_text: str, map_id: str, output_dir: str, args) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_path = out_dir / f"{map_id}.txt"
    video_path = out_dir / f"{map_id}.mp4"
    map_path.write_text(map_text, encoding="utf-8")

    jar_path = package_resource("mario", "PlayAstar.jar")
    assets_path = package_resource("mario", "img").resolve()
    cmd = [
        "java",
        "-Djava.awt.headless=true",
        "-jar",
        jar_path.name,
        str(map_path.resolve()),
        "human",
        str(assets_path).rstrip("/\\") + "/",
        str(out_dir.resolve()),
        video_path.name,
    ]

    subprocess.run(cmd, cwd=jar_path.parent, check=True, timeout=180)
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Video was not created: {video_path}")

    return map_path, video_path


def _normalize_enemies(grid: list[list[str]]) -> None:
    for y, row in enumerate(grid):
        for x, tile in enumerate(row):
            if tile != "E":
                continue

            below = grid[y + 1][x] if y + 1 < len(grid) else "-"
            if below in {"<", ">"}:
                _convert_pipe_to_flower_pipe(grid, y + 1, x)
                grid[y][x] = "-"
            elif below == "-":
                grid[y][x] = "K"
            else:
                grid[y][x] = "g"


def _convert_pipe_to_flower_pipe(grid: list[list[str]], pipe_top_y: int, x: int) -> None:
    top = grid[pipe_top_y][x]
    left_x = x if top == "<" else x - 1
    right_x = left_x + 1
    if left_x < 0 or right_x >= len(grid[0]):
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


def _ensure_start_and_flag(lines: list[str]) -> None:
    width = max(len(line) for line in lines)
    floor_idx = max(0, len(lines) - 2 if len(lines) >= 2 else len(lines) - 1)

    if not any("M" in line for line in lines) and width > 1:
        row = list(lines[floor_idx])
        row[0] = "M"
        lines[floor_idx] = "".join(row)

    if not any("F" in line for line in lines) and width > 1:
        row = list(lines[floor_idx])
        row[-1] = "F"
        lines[floor_idx] = "".join(row)
