#!/usr/bin/env python3
"""Sokoban 레벨 뷰어: map.json 맵을 ``third_party/pcg_benchmark`` 타일 PNG로 그립니다.

  ``procedural_pipeline.games.sokoban`` 과 동일하게 ``SokobanProblem.render()``를 사용합니다
  (solid / empty / player / crate / target 스프라이트).

  ← / ↑ / p : 이전 맵
  → / ↓ / n : 다음 맵
  Home / End : 첫·끝 맵
  q 또는 창 닫기 : 종료

실행 (저장소 루트 ``tutorial/`` 에서):

  uv run python scenarios/sokoban/visualize_maps.py

서브모듈이 비어 있으면:

  git submodule update --init third_party/pcg_benchmark
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 스크립트 직접 실행 시 ``procedural_pipeline``·``pcg_benchmark`` import 보장
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_PCG_ROOT = _PROJECT_ROOT / "third_party" / "pcg_benchmark"
if str(_PCG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PCG_ROOT))


def _ensure_pcg_assets() -> None:
    img_dir = _PCG_ROOT / "pcg_benchmark" / "probs" / "sokoban" / "images"
    need = ("solid.png", "empty.png", "player.png", "crate.png", "target.png")
    missing = [n for n in need if not (img_dir / n).is_file()]
    if missing:
        print(
            "pcg_benchmark Sokoban 에셋이 없습니다.\n"
            f"  기대 경로: {img_dir}\n"
            "  다음으로 서브모듈을 받으세요:\n"
            "    git submodule update --init third_party/pcg_benchmark\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _render_with_pcg(map_text: str):
    from pcg_benchmark.probs.sokoban.problem import SokobanProblem  # type: ignore[import-not-found]

    from procedural_pipeline.games import sokoban as sokoban_game

    normalized = sokoban_game.normalize_map(map_text)
    content = sokoban_game._map_text_to_content(normalized)
    problem = SokobanProblem(
        width=int(content.shape[1]),
        height=int(content.shape[0]),
        difficulty=1,
        solver=1,
    )
    pil = problem.render(content)
    return pil.convert("RGB")


def main() -> None:
    p = argparse.ArgumentParser(description="Sokoban map.json 브라우저 (pcg_benchmark 타일)")
    p.add_argument(
        "--map-json",
        type=Path,
        default=Path(__file__).resolve().parent / "map.json",
        help="map.json 경로",
    )
    p.add_argument("--start", type=int, default=0, help="시작 인덱스 (0-based)")
    args = p.parse_args()

    path: Path = args.map_json
    if not path.is_file():
        print(f"파일 없음: {path}", file=sys.stderr)
        sys.exit(1)

    ordered = list(json.loads(path.read_text(encoding="utf-8")).items())
    if not ordered:
        print("맵이 비어 있습니다.", file=sys.stderr)
        sys.exit(1)

    _ensure_pcg_assets()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib·numpy 필요: uv sync 등으로 설치하세요.", file=sys.stderr)
        sys.exit(1)

    n = len(ordered)
    idx = max(0, min(args.start, n - 1))

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.08)

    def draw(i: int) -> None:
        map_id, raw = ordered[i]
        pil = _render_with_pcg(raw)
        img = np.asarray(pil, dtype=np.uint8)
        ax.clear()
        ax.imshow(img, interpolation="nearest", origin="upper")
        ax.set_axis_off()
        ax.set_title(f"{map_id}  —  {i + 1} / {n}  (pcg_benchmark tiles)", fontsize=11)
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        nonlocal idx
        if event.key is None:
            return
        k = event.key.lower()
        if k in ("q", "escape"):
            plt.close(fig)
            return
        if k in ("right", "down", "n", " "):
            idx = (idx + 1) % n
            draw(idx)
        elif k in ("left", "up", "p", "backspace"):
            idx = (idx - 1) % n
            draw(idx)
        elif k == "home":
            idx = 0
            draw(idx)
        elif k == "end":
            idx = n - 1
            draw(idx)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw(idx)
    fig.text(
        0.5,
        0.02,
        "←/↑/p 이전  ·  →/↓/n 다음  ·  Home/End 첫·끝  ·  q 종료",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    plt.show()


if __name__ == "__main__":
    main()
