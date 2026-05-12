#!/usr/bin/env python3
"""Sokoban 맵만 풀고(솔버 실행) 해결 과정 영상(mp4)만 저장합니다.

LLM·평가 파이프라인 없이 ``SokobanProblem`` + ``render_solution`` 만 사용합니다.

실행 (저장소 루트 ``tutorial/`` 에서, 프로젝트 가상환경 권장):

  uv run python scenarios/sokoban/solve_video.py --map-json scenarios/sokoban/map.json --id QUT4qGl2 -o outputs/solve_QUT4qGl2.mp4

  map.json 안의 **모든** 맵을 한 번에 (출력 디렉터리에 ``<id>.mp4``):

  uv run python scenarios/sokoban/solve_video.py --map-json scenarios/sokoban/map.json --all -o outputs/solve_all/

  uv run python scenarios/sokoban/solve_video.py --map-file path/to/level.txt -o out.mp4

표준입력으로 아스키 맵:

  uv run python scenarios/sokoban/solve_video.py -o out.mp4 < level.txt

옵션: ``--solver-power`` (기본 procedural_pipeline.games.sokoban 과 동일),
``--fps`` 영상 초당 프레임.

참고: ``pcg_benchmark`` 의 ``render_solution`` 은 구현상 프레임 PNG를
``third_party/pcg_benchmark/videos/sokoban/<이름>/`` 에도 남깁니다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from procedural_pipeline.games import sokoban as sgame  # noqa: E402
from pcg_benchmark.probs.sokoban.problem import SokobanProblem  # type: ignore[import-not-found]  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sokoban 솔버만 실행해 해답 영상(mp4) 생성")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--map-json",
        type=Path,
        help="map.json 경로 (--id 단일 / 또는 --all 배치)",
    )
    src.add_argument(
        "--map-file",
        type=Path,
        help="아스키 맵 텍스트 파일",
    )
    src.add_argument(
        "--stdin",
        action="store_true",
        help="표준입력에서 맵 텍스트 읽기",
    )

    p.add_argument(
        "--all",
        action="store_true",
        help="--map-json 과 함께: JSON의 모든 키에 대해 영상 생성 (-o 는 디렉터리)",
    )
    p.add_argument(
        "--id",
        type=str,
        default=None,
        help="map.json 안의 맵 키 하나만 (--all 이 아닐 때 필수)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="출력 mp4 파일 경로이거나, --all 일 때 출력 디렉터리",
    )
    p.add_argument(
        "--solver-power",
        type=int,
        default=sgame.DEFAULT_SOLVER_POWER,
        help=f"솔버 노드 확장 상한 (기본 {sgame.DEFAULT_SOLVER_POWER})",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="영상 FPS (기본 3)",
    )
    return p


def _load_map_text(args: argparse.Namespace) -> tuple[str, str]:
    """반환: (맵 문자열, render_solution 에 넘길 이름)."""
    if args.stdin:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit("표준입력이 비어 있습니다.")
        text = sgame.normalize_map(raw)
        return text, "stdin"

    if args.map_file is not None:
        if not args.map_file.is_file():
            sys.exit(f"파일 없음: {args.map_file}")
        raw = args.map_file.read_text(encoding="utf-8")
        text = sgame.normalize_map(raw)
        return text, args.map_file.stem

    assert args.map_json is not None
    if not args.map_json.is_file():
        sys.exit(f"파일 없음: {args.map_json}")
    if not args.id:
        sys.exit("--map-json 에서 한 맵만 쓸 때는 --id 가 필요합니다. 전체는 --all -o <디렉터리>")

    data = json.loads(args.map_json.read_text(encoding="utf-8"))
    if args.id not in data:
        sys.exit(f"map.json 에 키 없음: {args.id!r}")
    raw = data[args.id]
    text = sgame.normalize_map(raw)
    return text, args.id


def _load_json_maps(path: Path) -> list[tuple[str, str]]:
    """map.json 전체: (map_id, normalized_text) 목록, 키 순서 유지."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        sys.exit("map.json 은 비어 있지 않은 객체여야 합니다.")
    out: list[tuple[str, str]] = []
    for map_id, raw in data.items():
        if not isinstance(raw, str):
            sys.exit(f"맵 {map_id!r} 값은 문자열이어야 합니다.")
        out.append((str(map_id), sgame.normalize_map(raw)))
    return out


def _solve_and_write_video(
    map_text: str,
    map_name: str,
    output_mp4: Path,
    solver_power: int,
    fps: int,
) -> None:
    content = sgame._map_text_to_content(map_text)
    problem = SokobanProblem(
        width=int(content.shape[1]),
        height=int(content.shape[0]),
        difficulty=1,
        solver=solver_power,
    )
    info = problem.info(content)
    sol_len = len(info.get("solution") or [])
    print(
        f"[solve_video] map={map_name}  players={info['players']}  "
        f"crates={info['crates']}  targets={info['targets']}  solution_steps={sol_len}",
        file=sys.stderr,
    )

    frames, _ = problem.render_solution(info, map_name=map_name)
    if not frames:
        frames = [problem.render(content)] * 12

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    sgame._write_video(frames, output_mp4, fps=fps)
    print(f"[solve_video] wrote {output_mp4}", file=sys.stderr)


def main() -> None:
    args = _build_parser().parse_args()

    if args.all and args.map_json is None:
        sys.exit("--all 은 --map-json 과 함께만 사용할 수 있습니다.")
    if args.all and args.id:
        sys.exit("--all 일 때는 --id 를 쓰지 마세요.")
    if args.all and args.map_file is not None:
        sys.exit("--all 은 map.json 전용입니다.")
    if args.all and args.stdin:
        sys.exit("--all 은 map.json 전용입니다.")

    fps = max(1, int(round(args.fps)))

    if args.all:
        assert args.map_json is not None
        if not args.map_json.is_file():
            sys.exit(f"파일 없음: {args.map_json}")
        out_dir = args.output
        pairs = _load_json_maps(args.map_json)
        ok, fail = 0, 0
        for map_id, map_text in pairs:
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in map_id)
            target = out_dir / f"{safe}.mp4"
            try:
                _solve_and_write_video(map_text, map_id, target, args.solver_power, fps)
                ok += 1
            except Exception as e:
                print(f"[solve_video] FAIL {map_id!r}: {e}", file=sys.stderr)
                fail += 1
        print(f"[solve_video] done  ok={ok}  fail={fail}  dir={out_dir}", file=sys.stderr)
        if fail:
            sys.exit(1)
        return

    map_text, map_name = _load_map_text(args)
    _solve_and_write_video(map_text, map_name, args.output, args.solver_power, fps)


if __name__ == "__main__":
    main()
