"""Render pipeline maps and compare them with the original map_images PNGs.

Examples:
    uv run python -m procedural_pipeline.compare_original_maps --game both --limit 30
    uv run python -m procedural_pipeline.compare_original_maps --game mario --limit 5
    uv run python -m procedural_pipeline.compare_original_maps --game sokoban --ids 8Ua6YU1D,HjLVlKx4
"""

from __future__ import annotations

import argparse
import html
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from procedural_pipeline.games import mario, sokoban
from procedural_pipeline.paths import PROJECT_ROOT


@dataclass(frozen=True)
class CompareRow:
    game: str
    map_id: str
    original_path: Path | None
    rendered_path: Path | None
    combined_path: Path | None
    diff_path: Path | None
    status: str
    note: str = ""


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[CompareRow] = []
    games = ["mario", "sokoban"] if args.game == "both" else [args.game]
    for game_key in games:
        rows.extend(compare_game(game_key, args, out_dir))

    html_path = write_index(rows, out_dir)
    print(f"\n[compare] done -> {html_path}")
    print(f"[compare] open this file in a browser to view side-by-side comparisons.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render Mario/Sokoban maps from procedural_pipeline inputs and place "
            "the rendered first screen next to map_images originals."
        )
    )
    parser.add_argument("--game", choices=["mario", "sokoban", "both"], default="both")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated map IDs to compare. If set, --limit is applied after filtering.",
    )
    parser.add_argument("--output-dir", default="outputs/map_compare")
    parser.add_argument("--map-images-dir", default="map_images")
    parser.add_argument("--mario-maps-file", default=mario.DEFAULTS["maps_file"])
    parser.add_argument("--sokoban-maps-file", default=sokoban.DEFAULTS["maps_file"])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate rendered PNG/MP4 files even if cached outputs already exist.",
    )
    return parser.parse_args()


def compare_game(game_key: str, args: argparse.Namespace, out_dir: Path) -> list[CompareRow]:
    maps_file = args.mario_maps_file if game_key == "mario" else args.sokoban_maps_file
    loader = mario.load_maps if game_key == "mario" else sokoban.load_maps
    normalizer = mario.normalize_map if game_key == "mario" else sokoban.normalize_map
    original_subdir = "smb levels" if game_key == "mario" else "soko levels"

    wanted_ids = {item.strip() for item in args.ids.split(",") if item.strip()}
    maps = [(map_id, raw) for map_id, raw in loader(maps_file) if not wanted_ids or map_id in wanted_ids]
    maps = maps[: args.limit]

    print(f"[compare:{game_key}] maps={len(maps)}  source={maps_file}")
    rows: list[CompareRow] = []
    for index, (map_id, raw_map) in enumerate(maps, start=1):
        print(f"  [{index}/{len(maps)}] {map_id}")
        original_path = Path(args.map_images_dir) / original_subdir / f"{map_id}.png"
        rendered_path: Path | None = None
        combined_path: Path | None = None
        diff_path: Path | None = None

        try:
            normalized_map = normalizer(raw_map)
            if game_key == "mario":
                rendered_path = render_mario_full_map(
                    normalized_map,
                    map_id,
                    out_dir / "mario",
                    force=args.force,
                )
            else:
                rendered_path = render_sokoban_initial_screen(
                    normalized_map,
                    map_id,
                    out_dir / "sokoban",
                    force=args.force,
                )

            if not original_path.exists():
                rows.append(
                    CompareRow(
                        game=game_key,
                        map_id=map_id,
                        original_path=None,
                        rendered_path=rendered_path,
                        combined_path=None,
                        diff_path=None,
                        status="missing-original",
                        note=f"Missing {original_path}",
                    )
                )
                continue

            copied_original_path = out_dir / game_key / "originals" / f"{map_id}.png"
            copied_original_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original_path, copied_original_path)

            combined_path = out_dir / game_key / "combined" / f"{map_id}.png"
            diff_path = out_dir / game_key / "diff" / f"{map_id}.png"
            make_comparison_image(original_path, rendered_path, combined_path, diff_path)
            rows.append(
                CompareRow(
                    game=game_key,
                    map_id=map_id,
                    original_path=copied_original_path,
                    rendered_path=rendered_path,
                    combined_path=combined_path,
                    diff_path=diff_path,
                    status="ok",
                )
            )
        except Exception as exc:
            rows.append(
                CompareRow(
                    game=game_key,
                    map_id=map_id,
                    original_path=original_path if original_path.exists() else None,
                    rendered_path=rendered_path,
                    combined_path=combined_path,
                    diff_path=diff_path,
                    status="failed",
                    note=str(exc),
                )
            )
            print(f"    error: {exc}")

    return rows


def render_mario_full_map(map_text: str, map_id: str, out_dir: Path, *, force: bool) -> Path:
    frames_dir = out_dir / "rendered"
    map_dir = out_dir / "maps"
    frames_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    rendered_path = frames_dir / f"{map_id}.png"
    map_path = map_dir / f"{map_id}.txt"
    if rendered_path.exists() and not force:
        return rendered_path

    map_path.write_text(map_text, encoding="utf-8")
    image = render_mario_text_map(map_text)
    image.save(rendered_path)
    return rendered_path


def render_mario_text_map(map_text: str) -> Image.Image:
    scale = 16
    lines = [line.rstrip("\r") for line in map_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Mario map is empty.")

    width = max(len(line) for line in lines)
    height = len(lines)
    padded = [line.ljust(width, "-") for line in lines]
    sprites = load_mario_sprites()
    empty = sprites["-"]

    image = Image.new("RGBA", (width * scale, height * scale), (109, 143, 252, 255))
    for y, line in enumerate(padded):
        for x, char in enumerate(line):
            image.alpha_composite(empty, (x * scale, y * scale))
            sprite = sprites.get(mario_sprite_key(char))
            if sprite is not None:
                image.alpha_composite(sprite, (x * scale, y * scale))
    return image.convert("RGB")


def load_mario_sprites() -> dict[str, Image.Image]:
    image_dir = PROJECT_ROOT / "third_party" / "pcg_benchmark" / "pcg_benchmark" / "probs" / "smb" / "images"
    files = {
        "-": "empty.png",
        "M": "mario.png",
        "F": "flag_top.png",
        "I": "flag_middle.png",
        "f": "flag_white.png",
        "g": "gomba.png",
        "k": "greenkoopa.png",
        "r": "redkoopa.png",
        "y": "spiky.png",
        "X": "floor.png",
        "#": "solid.png",
        "Q": "question_coin.png",
        "S": "brick.png",
        "o": "coin.png",
        "<": "tubetop_left.png",
        ">": "tubetop_right.png",
        "[": "tube_left.png",
        "]": "tube_right.png",
        "H": "tube.png",
        "O": "tubetop.png",
    }
    return {key: Image.open(image_dir / filename).convert("RGBA") for key, filename in files.items()}


def mario_sprite_key(char: str) -> str | None:
    if char in {" ", "-"}:
        return None
    if char in {"?", "Q"}:
        return "Q"
    if char in {"B", "b", "S"}:
        return "S"
    if char in {"X", "#", "M", "F", "o", "<", ">", "[", "]"}:
        return char
    if char in {"t", "T", "H", "|", "%"}:
        return "H"
    if char in {"E", "g", "G"}:
        return "g"
    if char in {"k", "K"}:
        return "k"
    if char in {"r", "R"}:
        return "r"
    if char in {"y", "Y", "*"}:
        return "y"
    if char in {"C", "L", "U", "D", "@", "!", "1", "2"}:
        return "S"
    return None


def render_sokoban_initial_screen(map_text: str, map_id: str, out_dir: Path, *, force: bool) -> Path:
    frames_dir = out_dir / "rendered"
    map_dir = out_dir / "maps"
    frames_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    rendered_path = frames_dir / f"{map_id}.png"
    if rendered_path.exists() and not force:
        return rendered_path

    (map_dir / f"{map_id}.txt").write_text(map_text, encoding="utf-8")
    content = sokoban._map_text_to_content(map_text)
    problem = sokoban.SokobanProblem(
        width=content.shape[1],
        height=content.shape[0],
        difficulty=1,
        solver=sokoban.DEFAULT_SOLVER_POWER,
    )
    image = problem.render(content)
    image.save(rendered_path)
    return rendered_path


def make_comparison_image(original_path: Path, rendered_path: Path, combined_path: Path, diff_path: Path) -> None:
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.parent.mkdir(parents=True, exist_ok=True)

    original = Image.open(original_path).convert("RGB")
    rendered = Image.open(rendered_path).convert("RGB")

    diff = make_diff(original, rendered)
    diff.save(diff_path)

    label_h = 30
    gap = 12
    cell_w = 520
    left = fit_image(original, cell_w, 360)
    right = fit_image(rendered, cell_w, 360)
    diff_fit = fit_image(diff, cell_w, 360)
    height = label_h + max(left.height, right.height, diff_fit.height)
    width = cell_w * 3 + gap * 2

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    labels = ["original map_images", "pipeline first screen", "pixel diff (resized)"]
    for i, label in enumerate(labels):
        x = i * (cell_w + gap)
        draw.text((x + 8, 9), label, fill=(20, 20, 20), font=font)

    canvas.paste(left, (0, label_h))
    canvas.paste(right, (cell_w + gap, label_h))
    canvas.paste(diff_fit, ((cell_w + gap) * 2, label_h))
    canvas.save(combined_path)


def make_diff(original: Image.Image, rendered: Image.Image) -> Image.Image:
    rendered_resized = rendered.resize(original.size, Image.Resampling.NEAREST)
    diff = ImageChops.difference(original, rendered_resized)
    diff_np = np.asarray(diff, dtype=np.uint8)
    mask = diff_np.max(axis=2)
    heat = np.zeros((*mask.shape, 3), dtype=np.uint8)
    heat[..., 0] = mask
    heat[..., 1] = 255 - mask
    heat[..., 2] = 255 - mask
    return Image.fromarray(heat, mode="RGB")


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((max_width, max_height), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (max_width, max_height), (245, 245, 245))
    x = (max_width - fitted.width) // 2
    y = (max_height - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def write_index(rows: list[CompareRow], out_dir: Path) -> Path:
    html_path = out_dir / "index.html"
    ok_count = sum(1 for row in rows if row.status == "ok")
    fail_count = len(rows) - ok_count
    cards = "\n".join(render_html_card(row, out_dir) for row in rows)
    doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Map Comparison</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #15171a; }}
    header {{ position: sticky; top: 0; z-index: 1; padding: 14px 20px; background: #ffffff; border-bottom: 1px solid #dfe3e8; }}
    h1 {{ margin: 0 0 4px; font-size: 20px; }}
    .summary {{ color: #59616d; font-size: 13px; }}
    main {{ padding: 18px 20px 32px; display: grid; gap: 18px; }}
    section {{ background: #ffffff; border: 1px solid #dfe3e8; border-radius: 8px; padding: 12px; }}
    h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .meta {{ margin-left: 8px; color: #59616d; font-weight: normal; }}
    .stack {{ display: grid; gap: 12px; }}
    .label {{ font-size: 13px; color: #374151; margin-bottom: 6px; }}
    .image-strip {{ overflow-x: auto; overflow-y: hidden; border: 1px solid #e5e8ec; background: #eef1f5; }}
    img {{ display: block; max-width: none; height: auto; }}
    .error {{ color: #a33; white-space: pre-wrap; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>Map Comparison</h1>
    <div class="summary">ok: {ok_count} / failed or missing: {fail_count}</div>
  </header>
  <main>
{cards}
  </main>
</body>
</html>
"""
    html_path.write_text(doc, encoding="utf-8")
    return html_path


def render_html_card(row: CompareRow, base_dir: Path) -> str:
    title = f"{row.game} / {row.map_id}"
    if row.original_path and row.rendered_path and row.original_path.exists() and row.rendered_path.exists():
        original_src = browser_src(row.original_path, base_dir)
        rendered_src = browser_src(row.rendered_path, base_dir)
        return f"""    <section>
      <h2>{html.escape(title)} <span class="meta">{html.escape(row.status)}</span></h2>
      <div class="stack">
        <div>
          <div class="label">original map_images</div>
          <div class="image-strip"><img src="{html.escape(original_src)}" alt="{html.escape(title)} original"></div>
        </div>
        <div>
          <div class="label">pipeline rendered</div>
          <div class="image-strip"><img src="{html.escape(rendered_src)}" alt="{html.escape(title)} rendered"></div>
        </div>
      </div>
    </section>"""

    note = row.note or row.status
    return f"""    <section>
      <h2>{html.escape(title)} <span class="meta">{html.escape(row.status)}</span></h2>
      <div class="error">{html.escape(note)}</div>
    </section>"""


def browser_src(path: Path, base_dir: Path) -> str:
    rel = os.path.relpath(Path(path).resolve(), Path(base_dir).resolve())
    return quote(Path(rel).as_posix(), safe="/._-")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
