#!/usr/bin/env python3
"""Sokoban 맵 두 개를 상·하로 나란히 시각화합니다 (pcg_benchmark 타일).

- 위 / 아래 텍스트 칸에 아스키 맵을 붙여넣고 「시각화」를 누르면 즉시 갱신됩니다.
- 줄바꿈: Enter(개행), ``\\n`` 이스케이프, 또는 **한 줄에서 행들만 스페이스로 구분**한 입력.
  (각 덩어리는 ``#@$.*+-`` 문자만으로 이루어지고 길이 2 이상일 때만 행으로 나눕니다.
  ``####@  #`` 처럼 행 안 공백은 나누지 않습니다.)
  JSON 따옴표 문자열이면 그대로 디코드합니다.
  행 **내부**의 공백만 있는 경우는 기존처럼 바닥(`-`)으로 정규화됩니다.

실행 (저장소 루트 ``tutorial/`` 에서):

  uv run python scenarios/sokoban/visualize_compare.py

CLI로 두 파일 비교:

  uv run python scenarios/sokoban/visualize_compare.py --top a.txt --bottom b.txt

서브모듈이 비어 있으면:

  git submodule update --init third_party/pcg_benchmark
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext

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
            "  git submodule update --init third_party/pcg_benchmark\n",
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


_ROW_CHARS = frozenset("#@$.*+-")


def _try_space_separated_rows_one_line(line: str) -> str | None:
    """한 줄이 ``######## ##--##`` 처럼 스페이스로 행만 구분된 경우 ``\\n`` 으로 펼칩니다."""
    if "\t" in line:
        line = line.replace("\t", " ")
    if " " not in line:
        return None
    parts = re.split(r"\s+", line.strip())
    if len(parts) < 2:
        return None
    if any(len(p) < 2 for p in parts):
        return None
    for p in parts:
        if not _ROW_CHARS.issuperset(p):
            return None
    return "\n".join(parts)


def _expand_space_separated_rows(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        expanded = _try_space_separated_rows_one_line(line)
        if expanded is not None:
            out.extend(expanded.split("\n"))
        else:
            out.append(line)
    return "\n".join(out)


def parse_pasted_ascii(raw: str) -> str:
    """붙여넣기/파일에서 온 문자열을 Sokoban 텍스트로 정리합니다."""
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    t = s.strip("\n")

    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        try:
            t = json.loads(t)
        except json.JSONDecodeError:
            t = t[1:-1]

    if "\\n" in t:
        t = t.replace("\\n", "\n")

    t = _expand_space_separated_rows(t)
    return t.strip("\n")


def run_gui() -> None:
    _ensure_pcg_assets()

    try:
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
    except ImportError:
        print("matplotlib·numpy 필요: uv sync 등으로 설치하세요.", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    root.title("Sokoban 상·하 비교")
    root.geometry("900x780")

    top_frame = tk.LabelFrame(root, text="위쪽 맵 (아스키)", padx=6, pady=4)
    top_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
    txt_top = scrolledtext.ScrolledText(top_frame, height=8, font=("Consolas", 10), wrap=tk.NONE)
    txt_top.pack(fill=tk.BOTH, expand=True)

    bot_frame = tk.LabelFrame(root, text="아래쪽 맵 (아스키)", padx=6, pady=4)
    bot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
    txt_bottom = scrolledtext.ScrolledText(bot_frame, height=8, font=("Consolas", 10), wrap=tk.NONE)
    txt_bottom.pack(fill=tk.BOTH, expand=True)

    btn_bar = tk.Frame(root)
    btn_bar.pack(fill=tk.X, padx=8, pady=6)
    tk.Button(btn_bar, text="시각화", command=lambda: draw()).pack(side=tk.LEFT, padx=(0, 8))
    tk.Button(btn_bar, text="칸 비우기", command=lambda: clear_all()).pack(side=tk.LEFT)

    fig = Figure(figsize=(9, 7), dpi=100)
    ax_top = fig.add_subplot(2, 1, 1)
    ax_bot = fig.add_subplot(2, 1, 2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, hspace=0.15)

    canvas_frame = tk.Frame(root)
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def clear_all() -> None:
        txt_top.delete("1.0", tk.END)
        txt_bottom.delete("1.0", tk.END)

    def draw() -> None:
        raw_a = txt_top.get("1.0", tk.END)
        raw_b = txt_bottom.get("1.0", tk.END)
        try:
            a = parse_pasted_ascii(raw_a)
            b = parse_pasted_ascii(raw_b)
            if not a.strip() or not b.strip():
                messagebox.showwarning("입력", "위·아래 모두 맵 문자열이 필요합니다.")
                return
            pil_a = _render_with_pcg(a)
            pil_b = _render_with_pcg(b)
            arr_a = np.asarray(pil_a, dtype=np.uint8)
            arr_b = np.asarray(pil_b, dtype=np.uint8)
        except Exception as e:
            messagebox.showerror("렌더 오류", str(e))
            return

        ax_top.clear()
        ax_bot.clear()
        ax_top.imshow(arr_a, interpolation="nearest", origin="upper")
        ax_top.set_axis_off()
        ax_top.set_title("위쪽 맵", fontsize=11)
        ax_bot.imshow(arr_b, interpolation="nearest", origin="upper")
        ax_bot.set_axis_off()
        ax_bot.set_title("아래쪽 맵", fontsize=11)
        canvas.draw_idle()

    root.bind("<Control-Return>", lambda e: draw())
    root.mainloop()


def run_cli(top_path: Path, bottom_path: Path) -> None:
    _ensure_pcg_assets()
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib·numpy 필요.", file=sys.stderr)
        sys.exit(1)

    a = parse_pasted_ascii(top_path.read_text(encoding="utf-8"))
    b = parse_pasted_ascii(bottom_path.read_text(encoding="utf-8"))
    pil_a = _render_with_pcg(a)
    pil_b = _render_with_pcg(b)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.imshow(np.asarray(pil_a.convert("RGB")), interpolation="nearest", origin="upper")
    ax1.set_axis_off()
    ax1.set_title(f"위: {top_path.name}", fontsize=11)
    ax2.imshow(np.asarray(pil_b.convert("RGB")), interpolation="nearest", origin="upper")
    ax2.set_axis_off()
    ax2.set_title(f"아래: {bottom_path.name}", fontsize=11)
    plt.subplots_adjust(hspace=0.15)
    plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Sokoban 맵 두 개 상·하 비교 (GUI 또는 두 파일)")
    p.add_argument("--top", type=Path, help="위쪽 맵 텍스트 파일")
    p.add_argument("--bottom", type=Path, help="아래쪽 맵 텍스트 파일")
    args = p.parse_args()

    if args.top is None and args.bottom is None:
        run_gui()
    elif args.top is not None and args.bottom is not None:
        if not args.top.is_file() or not args.bottom.is_file():
            print("--top / --bottom 파일을 확인하세요.", file=sys.stderr)
            sys.exit(1)
        run_cli(args.top, args.bottom)
    else:
        print("--top 과 --bottom 을 함께 지정하세요.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
