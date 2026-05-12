#!/usr/bin/env python3
"""borgar.net Sokoban 레벨 팩과 ``map.json`` 동일 맵 여부 검사.

사이트는 ``levels/<팩이름>.txt`` 텍스트를 `` --- `` 구분으로 여러 방을 나눕니다
(페이지: https://borgar.net/programs/sokoban/ 의 select + ajax).

기본 비교:

1. ``normalize_map`` (파이프라인과 동일)
2. **바깥 바닥(``-``만 있는 줄·열)을 잘라** 직사각형 ``struct`` 문자열을 만든 뒤,
3. ``difflib.SequenceMatcher`` 비율이 ``--min-ratio`` 이상이면 같은 맵으로 본다 (기본 **0.93**).
   ``--min-ratio 1.0`` 이면 구조 문자열이 **완전히 같을 때만** 매칭.

``--strict`` 는 (1)만 적용한 **전체 정규화 문자열**에 대해 ``--min-ratio`` 로만 비교합니다.

실행 (저장소 루트):

  uv run python scenarios/sokoban/check_borgar_vs_json.py
  uv run python scenarios/sokoban/check_borgar_vs_json.py --min-ratio 1.0
  uv run python scenarios/sokoban/check_borgar_vs_json.py --packs Intro,Sokoban
  uv run python scenarios/sokoban/check_borgar_vs_json.py --map-json scenarios/sokoban/map.json
  uv run python scenarios/sokoban/check_borgar_vs_json.py --strict
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_BASE = "https://borgar.net/programs/sokoban/"
LEVELS_SUBPATH = "levels/"

OPTION_RE = re.compile(r'<option\s+value="([^"]+)"\s*>', re.IGNORECASE)
SEGMENT_SPLIT_RE = re.compile(r"\s---\s", re.MULTILINE)


def _fetch(url: str, timeout: int = 30) -> str:
    req = Request(url, headers={"User-Agent": "tutorial-check-borgar/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def discover_pack_keys(base_url: str) -> list[str]:
    base = base_url.rstrip("/") + "/"
    html = _fetch(base)
    keys = OPTION_RE.findall(html)
    if not keys:
        raise RuntimeError(
            "인덱스 HTML에서 <option value=\"...\"> 를 찾지 못했습니다. "
            "사이트 구조가 바뀌었을 수 있습니다."
        )
    return keys


def segment_to_raw_map(segment: str) -> str:
    """borgar 팩의 한 조각(주석/들여쓰기 포함) → normalize_map에 넣을 ASCII."""
    lines: list[str] = []
    for line in segment.splitlines():
        s = line.rstrip("\r")
        if re.match(r"^\s*;", s):
            continue
        lines.append(s)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return ""
    mindent = min(len(ln) - len(ln.lstrip(" \t")) for ln in nonempty)
    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            continue
        if len(ln) >= mindent:
            out.append(ln[mindent:])
        else:
            out.append(ln.lstrip(" \t"))
    return "\n".join(out)


def split_pack_levels(text: str) -> list[str]:
    parts = SEGMENT_SPLIT_RE.split(text)
    return [p for p in parts if p.strip()]


def crop_non_dash_bbox(normalized: str) -> str:
    """``-`` 가 아닌 타일이 있는 최소 직사각형만 남긴다 (벽·목표·상자·플레이어 상대 배치)."""
    lines = normalized.strip().splitlines()
    if not lines:
        return ""
    w = max(len(ln) for ln in lines)
    padded = [ln.ljust(w, "-") for ln in lines]
    h = len(padded)
    coords = [(y, x) for y in range(h) for x in range(w) if padded[y][x] != "-"]
    if not coords:
        return normalized.strip()
    y0 = min(y for y, _ in coords)
    y1 = max(y for y, _ in coords)
    x0 = min(x for _, x in coords)
    x1 = max(x for _, x in coords)
    return "\n".join(padded[y][x0 : x1 + 1] for y in range(y0, y1 + 1))


def comparison_key(normalized: str, *, strict: bool) -> str:
    if strict:
        return normalized
    return crop_non_dash_bbox(normalized)


def best_json_match(
    bkey: str,
    json_pairs: list[tuple[str, str]],
    *,
    min_ratio: float,
) -> tuple[str | None, float]:
    """struct(또는 strict 시 전체 정규화) 문자열 기준 최고 유사도 ``map.json`` id."""
    best_mid: str | None = None
    best_r = 0.0
    for mid, jkey in json_pairs:
        r = SequenceMatcher(None, bkey, jkey).ratio()
        if r > best_r or (abs(r - best_r) < 1e-12 and (best_mid is None or mid < best_mid)):
            best_r = r
            best_mid = mid
    if best_mid is not None and best_r + 1e-15 >= min_ratio:
        return best_mid, best_r
    return None, best_r


def main() -> None:
    from procedural_pipeline.games.sokoban import normalize_map

    p = argparse.ArgumentParser(description="borgar.net Sokoban 팩 vs map.json 동일 맵 검사")
    p.add_argument(
        "--map-json",
        type=Path,
        default=Path(__file__).resolve().parent / "map.json",
        help="비교할 map.json",
    )
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE,
        help="borgar Sokoban 페이지 베이스 URL (끝 슬래시 있어도 됨)",
    )
    p.add_argument(
        "--packs",
        default=None,
        help="검사할 팩 키 쉼표 목록 (기본: 인덱스에 있는 전부). 예: Intro,Sokoban",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP 타임아웃(초)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="normalize_map 전체 문자열만 비교 (--min-ratio 적용, 바깥 '-' bbox 자르기 없음)",
    )
    p.add_argument(
        "--min-ratio",
        type=float,
        default=0.93,
        metavar="R",
        help="SequenceMatcher 유사도가 R 이상이면 매칭 (기본 0.93). 1.0 = 구조 문자열 완전 일치",
    )
    args = p.parse_args()

    if not 0.0 <= args.min_ratio <= 1.0:
        print("--min-ratio 는 0~1 사이여야 합니다.", file=sys.stderr)
        sys.exit(2)

    map_path: Path = args.map_json
    if not map_path.is_file():
        print(f"map.json 없음: {map_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(map_path.read_text(encoding="utf-8"))
    if args.strict:
        mode = f"strict + min_ratio>={args.min_ratio}"
    else:
        mode = f"구조(bbox crop) + min_ratio>={args.min_ratio}"
    print(f"비교 모드: {mode}\n")

    json_pairs: list[tuple[str, str]] = []
    seen_struct: dict[str, str] = {}
    for mid, raw in data.items():
        try:
            norm = normalize_map(raw)
        except ValueError as e:
            print(f"[skip json] {mid}: {e}", file=sys.stderr)
            continue
        skey = comparison_key(norm, strict=args.strict)
        if skey in seen_struct:
            print(
                f"[warn] JSON 내 동일 비교키: {seen_struct[skey]} 와 {mid} (첫 항목을 우선 매칭에 사용)",
                file=sys.stderr,
            )
        else:
            seen_struct[skey] = mid
        json_pairs.append((mid, skey))

    base = args.base_url.rstrip("/") + "/"
    try:
        all_keys = discover_pack_keys(base)
    except (HTTPError, URLError, OSError, RuntimeError) as e:
        print(f"인덱스 가져오기 실패: {e}", file=sys.stderr)
        sys.exit(1)

    if args.packs:
        want = {x.strip() for x in args.packs.split(",") if x.strip()}
        pack_keys = [k for k in all_keys if k in want]
        missing = want - set(pack_keys)
        if missing:
            print(f"[warn] 요청한 팩이 인덱스에 없음: {sorted(missing)}", file=sys.stderr)
    else:
        pack_keys = all_keys

    total_borgar = 0
    matched = 0
    matched_ids: set[str] = set()
    unmatched: list[tuple[str, int, str, float]] = []

    for pack_key in pack_keys:
        enc = quote(pack_key, safe="")
        url = f"{base}{LEVELS_SUBPATH}{enc}.txt"
        try:
            body = _fetch(url, timeout=args.timeout)
        except HTTPError as e:
            print(f"[skip pack] {pack_key!r} HTTP {e.code}: {url}", file=sys.stderr)
            continue
        except (URLError, OSError) as e:
            print(f"[skip pack] {pack_key!r} {e}: {url}", file=sys.stderr)
            continue

        segments = split_pack_levels(body)
        if not segments:
            print(f"[warn] 팩에 조각 없음: {pack_key!r}", file=sys.stderr)
            continue

        for i, seg in enumerate(segments, start=1):
            raw = segment_to_raw_map(seg)
            if not raw.strip():
                continue
            total_borgar += 1
            label = f"{pack_key}#{i}"
            try:
                norm = normalize_map(raw)
            except ValueError as e:
                print(f"[skip borgar] {label}: {e}", file=sys.stderr)
                continue
            bkey = comparison_key(norm, strict=args.strict)
            mid, ratio = best_json_match(bkey, json_pairs, min_ratio=args.min_ratio)
            if mid:
                matched += 1
                matched_ids.add(mid)
                ratio_note = "" if ratio >= 1.0 - 1e-12 else f"\t(ratio={ratio:.4f})"
                print(f"MATCH\t{label}\t->\tmap.json[{mid}]{ratio_note}")
            else:
                preview = bkey[:80] + ("…" if len(bkey) > 80 else "")
                unmatched.append((label, len(bkey.splitlines()), preview, ratio))

    print()
    print(f"borgar 방(유효) 총 {total_borgar}개 중 map.json 과 동일 ({mode}): {matched}개")
    if unmatched:
        print(f"미매칭 {len(unmatched)}개 (best ratio / 미리보기):")
        for label, rows, preview, br in sorted(unmatched, key=lambda t: -t[3])[:50]:
            print(f"  {label}\trows={rows}\tbest={br:.3f}\t{preview!r}")
        if len(unmatched) > 50:
            print(f"  ... 외 {len(unmatched) - 50}개")

    never = [mid for mid in data if mid not in matched_ids]
    if never:
        print()
        print(f"map.json {len(never)}개 id는 이번에 검사한 borgar 팩 어디에도 없음 ({mode}):")
        for mid in never[:40]:
            print(f"  {mid}")
        if len(never) > 40:
            print(f"  ... 외 {len(never) - 40}개")


if __name__ == "__main__":
    main()
