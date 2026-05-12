#!/usr/bin/env python3
"""sturgeon scheme2output 를 시드별로 돌려 생성된 .lvl 과 map.json 의 타깃 맵들과 유사도를 비교합니다.

map.json 은 ``# - @ $ . * +`` 등을 쓰고, sturgeon ``work/*.lvl`` 은 보통 ``X - @ # o`` (벽/바닥/플레이어/상자/목표) 등이라
문자를 의미 계층(W/F/P/C/G/…)으로 통일한 뒤 문자 단위 일치율을 냅니다.

사전 준비 (sturgeon-pub 디렉터리에서):

  python input2tile.py --outfile work/soko.tile --textfile levels/mkiii/soko.lvl
  python tile2scheme.py --outfile work/soko.scheme --tilefile work/soko.tile

실행 (저장소 루트 ``tutorial/`` 에서):

  uv run python scenarios/sokoban/sturgeon_seed_match.py \\
    --sturgeon-root sokoban_ref/sturgeon-pub \\
    --map-json scenarios/sokoban/map.json \\
    --max-seed 500 --threshold 0.8 \\
    --start-seed 1 \\
    --log-file scenarios/sokoban/sturgeon_seed_match.log \\
    --hits-file scenarios/sokoban/sturgeon_hits.txt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, TextIO

# map.json (튜토리얼) 문자 → 의미 토큰
_TUTORIAL_MAP = {
    "#": "W",
    "-": "F",
    " ": "F",
    "@": "P",
    "+": "R",  # 플레이어 on goal
    "$": "C",
    "*": "K",  # 상자 on goal
    ".": "G",
}

# sturgeon .lvl (mkiii soko 예시 계열) 문자 → 의미 토큰
_STURGEON_MAP = {
    "X": "W",
    "x": "W",
    "-": "F",
    "@": "P",
    "#": "C",
    "o": "G",
    "O": "G",
    ".": "G",
}


def _strip_meta_and_lines(raw: str) -> list[str]:
    lines: list[str] = []
    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("META") or s.startswith("{"):
            break
        lines.append(s)
    return lines


def canonical_tutorial(map_text: str) -> str:
    lines = [ln.rstrip("\r") for ln in map_text.splitlines() if ln.strip()]
    out: list[str] = []
    for line in lines:
        row = []
        for ch in line:
            row.append(_TUTORIAL_MAP.get(ch, "?"))
        out.append("".join(row))
    return "".join(out)


def raw_grid_ascii(lvl_path: Path) -> str:
    """META 위쪽 격자 줄만 합친 문자열 (파일에 나온 아스키 그대로)."""
    raw = lvl_path.read_text(encoding="utf-8", errors="replace")
    lines = _strip_meta_and_lines(raw)
    return "\n".join(lines)


def canonical_sturgeon(lvl_path: Path) -> str:
    raw = lvl_path.read_text(encoding="utf-8", errors="replace")
    lines = _strip_meta_and_lines(raw)
    out: list[str] = []
    for line in lines:
        row = []
        for ch in line:
            row.append(_STURGEON_MAP.get(ch, "?"))
        out.append("".join(row))
    return "".join(out)


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) == len(b):
        return sum(x == y for x, y in zip(a, b)) / len(a)
    # 길이 다르면 편집 거리 기반 근사 (문자 스트림 정렬)
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


TARGET_KEYS_DEFAULT = (
    "UeeTjy5b",
    "oHSLE5UO",
    "79DSy6TF",
    "aE7sjnta",
    "wSCVt298",
)

# util_solvers.try_import_pysat 와 동일하게 검사
_PYSAT_IMPORT_SNIPPET = (
    "import pysat.card, pysat.formula, pysat.examples.fm, "
    "pysat.examples.rc2, pysat.solvers"
)


def check_pysat_available(py: str) -> None:
    r = subprocess.run(
        [py, "-c", _PYSAT_IMPORT_SNIPPET],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        print(
            "\n[오류] 이 Python 에 PySAT(pip 패키지 이름: python-sat) 가 없습니다.\n"
            f"  사용 중 interpreter: {py}\n"
            "  설치 예:\n"
            f'    "{py}" -m pip install python-sat\n'
            "  (sturgeon 쓰는 가상환경을 활성화한 뒤 같은 명령을 실행하세요.)\n"
            "  검사 생략: --skip-pysat-check\n",
            file=sys.stderr,
        )
        if err:
            print(err[:4000], file=sys.stderr)
        sys.exit(1)


class TeeStream(TextIO):
    """콘솔과 로그 파일에 동시에 씁니다."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self.streams:
            n = s.write(data)
        for s in self.streams:
            s.flush()
        return n if n else len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def fileno(self) -> int:
        return self.streams[0].fileno()

    def isatty(self) -> bool:
        return self.streams[0].isatty()


@contextmanager
def tee_stdio(log_f: IO[str]):
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = TeeStream(sys.__stdout__, log_f)  # type: ignore[assignment]
    sys.stderr = TeeStream(sys.__stderr__, log_f)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


def _write_hits_header(
    hits_f: IO[str],
    *,
    threshold: float,
    map_json: Path,
    keys: list[str],
    sturgeon_root: Path,
    start_seed: int,
    max_seed: int,
) -> None:
    hits_f.write(
        "# sturgeon_seed_match — 임계값 이상 유사도만 기록\n"
        f"# threshold={threshold}\n"
        f"# map_json={map_json.resolve()}\n"
        f"# keys={keys}\n"
        f"# sturgeon_root={sturgeon_root}\n"
        f"# seed_range={start_seed}..{max_seed}\n"
        f"# wrote_at_utc={datetime.now(timezone.utc).isoformat()}\n\n"
    )
    hits_f.flush()


def _write_hits_append_marker(
    hits_f: IO[str],
    *,
    threshold: float,
    map_json: Path,
    keys: list[str],
    sturgeon_root: Path,
    start_seed: int,
    max_seed: int,
) -> None:
    hits_f.write("\n")
    hits_f.write("# --- appended run (기존 파일에 이어 씀) ---\n")
    hits_f.write(f"# threshold={threshold}\n")
    hits_f.write(f"# map_json={map_json.resolve()}\n")
    hits_f.write(f"# keys={keys}\n")
    hits_f.write(f"# sturgeon_root={sturgeon_root}\n")
    hits_f.write(f"# seed_range={start_seed}..{max_seed}\n")
    hits_f.write(f"# wrote_at_utc={datetime.now(timezone.utc).isoformat()}\n\n")
    hits_f.flush()


def _append_hit_record(
    hits_f: IO[str],
    *,
    seed: int,
    map_id: str,
    sim: float,
    threshold: float,
    lvl_path: Path,
    ascii_grid: str,
    sims: dict[str, float],
    key_order: list[str],
) -> None:
    hits_f.write("=" * 80 + "\n")
    hits_f.write(f"seed\t{seed}\n")
    hits_f.write(f"map_id\t{map_id}\n")
    hits_f.write(f"similarity\t{sim:.6f}\n")
    hits_f.write(f"threshold\t{threshold}\n")
    hits_f.write(f"lvl_path\t{lvl_path.resolve()}\n")
    hits_f.write("\n--- 생성 맵 (.lvl 격자) ---\n")
    hits_f.write(ascii_grid.rstrip() + "\n")
    hits_f.write("\n--- 이 시드에서 각 map.json 키별 유사도 ---\n")
    for k in key_order:
        hits_f.write(f"  {k}\t{sims[k]:.6f}\n")
    hits_f.write("\n")
    hits_f.flush()


def main() -> None:
    p = argparse.ArgumentParser(description="sturgeon 시드 순회 → map.json 맵과 유사도 로그")
    p.add_argument(
        "--sturgeon-root",
        type=Path,
        required=True,
        help="sturgeon-pub 루트 (그 안에 scheme2output.py, work/soko.scheme)",
    )
    p.add_argument("--map-json", type=Path, required=True)
    p.add_argument(
        "--keys",
        nargs="+",
        default=list(TARGET_KEYS_DEFAULT),
        help="비교할 map.json 키 (기본: map.json 맨 아래 5개)",
    )
    p.add_argument("--max-seed", type=int, default=500)
    p.add_argument(
        "--start-seed",
        type=int,
        default=1,
        metavar="N",
        help="순회 시작 시드 (포함). 기본 1.",
    )
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument(
        "--scheme-file",
        type=str,
        default="work/soko.scheme",
        help="sturgeon-root 기준 scheme 상대 경로",
    )
    p.add_argument(
        "--no-print-map",
        action="store_true",
        help="생성 맵 본문·유사도 목록 출력 생략 (MATCH 줄만 threshold 넘을 때)",
    )
    p.add_argument(
        "--python",
        type=Path,
        default=None,
        help="scheme2output 를 돌릴 Python 실행 파일 (기본: 이 스크립트를 실행한 sys.executable). "
        "sturgeon venv 의 python 을 쓰려면 경로를 지정하세요.",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="scheme2output 가 한 번이라도 실패하면 자식 프로세스 출력을 보여 준 뒤 종료합니다.",
    )
    p.add_argument(
        "--skip-pysat-check",
        action="store_true",
        help="시작 시 PySAT 임포트 검사 생략 (이미 설치됐는데 검사만 실패할 때 등).",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="표준 출력·표준 오류 전체를 UTF-8 텍스트로 기록합니다 (화면에도 동일하게 출력).",
    )
    p.add_argument(
        "--no-hit-banner",
        action="store_true",
        help="유사도가 임계값 이상일 때 구분선·안내 문구 생략 (MATCH 줄만).",
    )
    p.add_argument(
        "--hits-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="유사도가 --threshold 이상인 경우만 기록. 파일이 이미 있으면 덮지 않고 끝에 추가(append)합니다.",
    )
    args = p.parse_args()

    if args.start_seed < 1:
        print("--start-seed 는 1 이상이어야 합니다.", file=sys.stderr)
        sys.exit(1)
    if args.start_seed > args.max_seed:
        print("--start-seed 는 --max-seed 이하여야 합니다.", file=sys.stderr)
        sys.exit(1)

    if args.python is not None and not args.python.is_file():
        print(f"--python 을 찾을 수 없음: {args.python}", file=sys.stderr)
        sys.exit(1)

    root = args.sturgeon_root.resolve()
    scheme = root / args.scheme_file
    if not scheme.is_file():
        print(f"scheme 없음: {scheme}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(args.map_json.read_text(encoding="utf-8"))
    targets: dict[str, str] = {}
    for k in args.keys:
        if k not in data:
            print(f"map.json 에 키 없음: {k}", file=sys.stderr)
            sys.exit(1)
        targets[k] = canonical_tutorial(data[k])

    py = str(args.python.resolve()) if args.python else sys.executable
    script = root / "scheme2output.py"
    if not script.is_file():
        print(f"scheme2output.py 없음: {script}", file=sys.stderr)
        sys.exit(1)

    log_ctx: Any
    if args.log_file is not None:
        log_path = args.log_file.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_ctx = open(log_path, "w", encoding="utf-8")
    else:
        log_ctx = nullcontext()

    hits_ctx: Any
    hits_append = False
    if args.hits_file is not None:
        hp = args.hits_file.expanduser().resolve()
        hp.parent.mkdir(parents=True, exist_ok=True)
        hits_append = hp.is_file()
        hits_ctx = open(hp, "a" if hits_append else "w", encoding="utf-8")
    else:
        hits_ctx = nullcontext()

    with log_ctx as log_f:
        with hits_ctx as hits_f:
            if log_f is not None:
                with tee_stdio(log_f):
                    _run_match(args, root, scheme, targets, py, hits_f, hits_append=hits_append)
            else:
                _run_match(args, root, scheme, targets, py, hits_f, hits_append=hits_append)


def _run_match(
    args: argparse.Namespace,
    root: Path,
    scheme: Path,
    targets: dict[str, str],
    py: str,
    hits_f: IO[str] | None,
    *,
    hits_append: bool = False,
) -> None:
    script = root / "scheme2output.py"

    if not args.skip_pysat_check:
        check_pysat_available(py)

    if hits_f is not None:
        if hits_append:
            _write_hits_append_marker(
                hits_f,
                threshold=args.threshold,
                map_json=args.map_json,
                keys=list(args.keys),
                sturgeon_root=root,
                start_seed=args.start_seed,
                max_seed=args.max_seed,
            )
        else:
            _write_hits_header(
                hits_f,
                threshold=args.threshold,
                map_json=args.map_json,
                keys=list(args.keys),
                sturgeon_root=root,
                start_seed=args.start_seed,
                max_seed=args.max_seed,
            )

    print(
        f"[match] root={root}\n"
        f"[match] scheme={scheme}\n"
        f"[match] python={py}\n"
        f"[match] keys={list(targets.keys())}\n"
        f"[match] start_seed={args.start_seed} max_seed={args.max_seed} threshold={args.threshold}",
        file=sys.stderr,
    )
    if args.log_file is not None:
        print(f"[match] log_file={args.log_file.expanduser().resolve()}", file=sys.stderr)
    if args.hits_file is not None:
        hp = args.hits_file.expanduser().resolve()
        print(
            f"[match] hits_file={hp} (append={hits_append})",
            file=sys.stderr,
        )

    for seed in range(args.start_seed, args.max_seed + 1):
        out_base = f"work/_seed_match_{seed}"
        cmd = [
            py,
            str(script),
            "--outfile",
            out_base,
            "--schemefile",
            args.scheme_file,
            "--mkiii-example",
            "soko2",
            "--size",
            "8",
            "8",
            "--mkiii-layers",
            "15",
            "--solver",
            "pysat-minicard",
            "--randomize",
            str(seed),
        ]
        try:
            r = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            print(f"[seed {seed}] TIMEOUT", file=sys.stderr)
            continue
        except OSError as exc:
            print(
                f"[seed {seed}] subprocess OSError (프로세스 생성 실패): {exc}\n"
                f"  python={py!r}\n"
                "  경로·권한·백신 차단을 확인하거나 --python 으로 다른 실행 파일을 지정해 보세요.",
                file=sys.stderr,
            )
            sys.exit(1)

        if r.returncode != 0:
            print(f"[seed {seed}] scheme2output 실패 rc={r.returncode}", file=sys.stderr)
            out = (r.stdout or "").strip()
            err = (r.stderr or "").strip()
            if out:
                print("[scheme2output stdout]\n" + out[:8000], file=sys.stderr)
            if err:
                print("[scheme2output stderr]\n" + err[:8000], file=sys.stderr)
            if not out and not err:
                print("(자식 stdout/stderr 비어 있음)", file=sys.stderr)
            if args.stop_on_error:
                sys.exit(1)
            continue

        lvl_path = root / "work" / f"_seed_match_{seed}.lvl"
        if not lvl_path.is_file():
            print(f"[seed {seed}] .lvl 없음: {lvl_path}", file=sys.stderr)
            continue

        gen = canonical_sturgeon(lvl_path)
        ascii_grid = raw_grid_ascii(lvl_path)

        sims: dict[str, float] = {}
        for map_id, ref in targets.items():
            sims[map_id] = similarity(gen, ref)

        if not args.no_print_map:
            print(f"===== seed {seed}  ({lvl_path.name}) =====")
            print(ascii_grid)
            print("--- similarity (canonical tokens, vs map.json targets) ---")
            for map_id in args.keys:
                print(f"  {map_id}: {sims[map_id]:.4f}")
            print()

        for map_id, sim in sims.items():
            if sim >= args.threshold:
                if hits_f is not None:
                    _append_hit_record(
                        hits_f,
                        seed=seed,
                        map_id=map_id,
                        sim=sim,
                        threshold=args.threshold,
                        lvl_path=lvl_path,
                        ascii_grid=ascii_grid,
                        sims=sims,
                        key_order=list(args.keys),
                    )
                if not args.no_hit_banner:
                    pct = 100.0 * args.threshold
                    print(
                        "\n"
                        "================================================================================\n"
                        f"  [유사도 충족] 목표 {pct:g}% 이상 — seed={seed}  map_id={map_id}  similarity={sim:.4f}\n"
                        "  (canonical 토큰 기준, threshold 와 동일 기준)\n"
                        "================================================================================"
                    )
                print(
                    f"MATCH seed={seed} map_id={map_id} similarity={sim:.4f} "
                    f"(threshold={args.threshold})"
                )


if __name__ == "__main__":
    main()
