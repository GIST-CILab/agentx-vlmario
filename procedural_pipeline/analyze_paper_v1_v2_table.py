"""Table: paper (human study) vs v1 (full LLM) vs v2 (blind LLM) on five experience metrics.

Aligns with the paper's Table 1 framing: **perceived source of creation**
(Perceived as AI vs Perceived as Human), using 1–5 Likert means (and SD).

- **Paper**: fixed numbers from the published table (participant ratings).
  The paper pooled **Mario + Sokoban**; use ``--combined`` to pool our two games the same way.
- **v1**: each map's aggregated scores, grouped by that run's
  `creator_belief` → Human / AI (same as `analyze_paper_proxy` / RQ4 belief split).
- **v2**: blind prompt has no belief; each map is assigned the **v1 belief**
  for that map so v2 means are comparable under the same perceived-source bins.

Also prints/writes **ground-truth** Human vs AI means (same rule as ``load_truth_map``:
first 15 map ids in ``maps.json`` = Human, rest = AI): ``truth_rows`` in JSON,
second section in ``paper_v1_v2_table.md``, and ``paper_v1_v2_truth_compare.png``.

Usage:
  uv run python -m procedural_pipeline.analyze_paper_v1_v2_table --game mario
  uv run python -m procedural_pipeline.analyze_paper_v1_v2_table --combined
  uv run python -m procedural_pipeline.analyze_paper_v1_v2_table --game mario \\
    --v1-csv results/v1.3/procedural_mario/results_20260422-214139.csv \\
    --v2-csv outputs/procedural_mario_v2_blind/results_20260428-151808.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from procedural_pipeline.games import GAMES
from procedural_pipeline.analyze_results import (
    METRICS,
    load_results,
    load_truth_map,
    mann_whitney_u_two_sided,
    normalize_creator_label,
    to_number,
)

# Table 1 (paper): descriptive statistics by *perceived* AI vs Human.
# Keys match internal metric ids: fun, challenge, frustration, surprise, design.
PAPER_PERCEIVED_AI = {
    "fun": (2.92, 1.33),
    "challenge": (3.88, 1.27),
    "frustration": (3.60, 1.37),
    "surprise": (2.65, 1.35),
    "design": (2.70, 1.23),
}
PAPER_PERCEIVED_HUMAN = {
    "fun": (3.72, 1.20),
    "challenge": (3.65, 1.32),
    "frustration": (2.84, 1.43),
    "surprise": (2.62, 1.26),
    "design": (3.57, 1.17),
}

METRIC_TITLES = {
    "fun": "Enjoyment / Fun",
    "challenge": "Difficulty / Challenge",
    "frustration": "Frustration / Negative affect",
    "surprise": "Novelty / Surprise",
    "design": "Aesthetics / Design quality",
}

CSV_METRIC_COL = {
    "fun": "enjoyment",
    "challenge": "difficulty",
    "frustration": "frustration",
    "surprise": "novelty",
    "design": "aesthetics",
}


def main() -> None:
    args = parse_args()

    if args.combined:
        label = "combined (Mario + Sokoban)"
        mario_bundle = _run_for_game(
            "mario",
            v1_csv=None,
            v2_csv=None,
            v1_dir=args.mario_v1_dir,
            v2_dir=args.mario_v2_dir,
            maps_file=args.mario_maps_file,
        )
        soko_bundle = _run_for_game(
            "sokoban",
            v1_csv=None,
            v2_csv=None,
            v1_dir=args.sokoban_v1_dir,
            v2_dir=args.sokoban_v2_dir,
            maps_file=args.sokoban_maps_file,
        )
        v1_stats = _merge_group_stats(mario_bundle["v1_stats"], soko_bundle["v1_stats"])
        v2_stats = _merge_group_stats(mario_bundle["v2_stats"], soko_bundle["v2_stats"])
        v1_truth_stats = _merge_group_stats(mario_bundle["v1_truth_stats"], soko_bundle["v1_truth_stats"])
        v2_truth_stats = _merge_group_stats(mario_bundle["v2_truth_stats"], soko_bundle["v2_truth_stats"])
        p_interpretation = (
            "Mann–Whitney U (two-sided): Human-bin vs AI-bin using **pooled** per-map scores from "
            "Mario and Sokoban (same framing as the paper’s pooled Table 1). v1_p = full prompt; v2_p = blind "
            "scores in bins defined by each map’s v1 belief. Games differ in difficulty — pooling is for "
            "paper alignment, not a claim that maps are exchangeable across games."
        )
        summary = {
            "mode": "combined",
            "label": label,
            "mario": {
                **{k: mario_bundle[k] for k in ("v1_csv", "v2_csv", "n_maps_v1_ok", "n_maps_paired", "v1_group_counts", "v2_group_counts")},
                "v1_truth_group_counts": mario_bundle["v1_truth_stats"]["counts"],
                "v2_truth_group_counts": mario_bundle["v2_truth_stats"]["counts"],
            },
            "sokoban": {
                **{k: soko_bundle[k] for k in ("v1_csv", "v2_csv", "n_maps_v1_ok", "n_maps_paired", "v1_group_counts", "v2_group_counts")},
                "v1_truth_group_counts": soko_bundle["v1_truth_stats"]["counts"],
                "v2_truth_group_counts": soko_bundle["v2_truth_stats"]["counts"],
            },
            "pooled_v1_group_counts": v1_stats["counts"],
            "pooled_v2_group_counts": v2_stats["counts"],
            "pooled_v1_truth_group_counts": v1_truth_stats["counts"],
            "pooled_v2_truth_group_counts": v2_truth_stats["counts"],
            "n_maps_paired_total": mario_bundle["n_maps_paired"] + soko_bundle["n_maps_paired"],
            "mann_whitney_note": p_interpretation,
        }
    else:
        label = args.game
        bundle = _run_for_game(
            args.game,
            v1_csv=args.v1_csv,
            v2_csv=args.v2_csv,
            v1_dir=args.v1_dir,
            v2_dir=args.v2_dir,
            maps_file=args.maps_file,
        )
        v1_stats = bundle["v1_stats"]
        v2_stats = bundle["v2_stats"]
        v1_truth_stats = bundle["v1_truth_stats"]
        v2_truth_stats = bundle["v2_truth_stats"]
        p_interpretation = (
            "Mann–Whitney U (two-sided): p compares per-map scores in the v1-believed-Human bin vs "
            "v1-believed-AI bin. v1_p tests full-prompt scores; v2_p tests blind scores under the same bins. "
            "Interpret with care: small/imbalanced n, bins are defined by the same LLM that produced v1, "
            "and maps are not i.i.d. draws from the paper population."
        )
        summary = {
            "mode": "single",
            "game": args.game,
            "label": label,
            "v1_csv": bundle["v1_csv"],
            "v2_csv": bundle["v2_csv"],
            "n_maps_v1_ok": bundle["n_maps_v1_ok"],
            "n_maps_v2_ok": bundle["n_maps_v2_ok"],
            "n_maps_paired": bundle["n_maps_paired"],
            "v1_group_counts": v1_stats["counts"],
            "v2_group_counts": v2_stats["counts"],
            "v1_truth_group_counts": v1_truth_stats["counts"],
            "v2_truth_group_counts": v2_truth_stats["counts"],
            "mann_whitney_note": p_interpretation,
        }

    table = build_table_rows(v1_stats, v2_stats)
    summary["rows"] = table
    truth_table = build_truth_table_rows(v1_truth_stats, v2_truth_stats)
    summary["truth_rows"] = truth_table
    summary["truth_bins_note"] = (
        "Ground truth from maps.json key order: first 15 map ids = Human-made, remaining = AI-generated "
        "(same rule as load_truth_map)."
    )

    print_table(label, table, v1_stats["counts"], v2_stats["counts"])
    print_truth_table(label, truth_table, v1_truth_stats["counts"], v2_truth_stats["counts"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "paper_v1_v2_table.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "paper_v1_v2_table.md").write_text(
        "## Perceived-source bins (v1 belief; v2 blind scores in same bins)\n\n"
        + markdown_table(table)
        + "\n\n"
        + _markdown_p_note(args.combined)
        + "\n\n## Ground-truth bins (maps.json: first 15 Human, rest AI)\n\n"
        + markdown_truth_table(truth_table)
        + "\n\n"
        + _markdown_truth_p_note(args.combined)
        + "\n",
        encoding="utf-8",
    )
    if not args.no_plot:
        plot_path = out / "paper_v1_v2_compare.png"
        _save_comparison_plots(table, plot_path, pooled=args.combined)
        print(f"[{label}] wrote plot -> {plot_path}")
        truth_plot = out / "paper_v1_v2_truth_compare.png"
        _save_truth_comparison_plots(truth_table, truth_plot, pooled=args.combined)
        print(f"[{label}] wrote truth plot -> {truth_plot}")
    print(f"\n[{label}] wrote {out / 'paper_v1_v2_table.json'} and paper_v1_v2_table.md")
    print("\n" + p_interpretation)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare paper Table 1 style means with v1 and v2 (blind, belief from v1). "
        "Use --combined to pool Mario + Sokoban like the paper.",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Pool Mario and Sokoban maps into one Human-bin / AI-bin distribution (paper Table 1 is pooled).",
    )
    p.add_argument(
        "--game",
        default=None,
        choices=sorted(GAMES.keys()),
        help="Single game (omit when using --combined).",
    )

    p.add_argument("--v1-csv", default=None, help="(single-game) v1 results CSV.")
    p.add_argument(
        "--v1-dir",
        default=None,
        help="(single-game) Directory with latest results_*.csv if --v1-csv omitted.",
    )
    p.add_argument("--v2-csv", default=None, help="(single-game) v2 blind results CSV.")
    p.add_argument(
        "--v2-dir",
        default=None,
        help="(single-game) Directory with latest results_*.csv if --v2-csv omitted.",
    )
    p.add_argument("--maps-file", default=None, help="(single-game) Override maps.json path.")

    p.add_argument(
        "--mario-v1-dir",
        default=str(Path("results/v1.3") / "procedural_mario"),
        help="(--combined) Mario v1 directory.",
    )
    p.add_argument(
        "--mario-v2-dir",
        default=str(Path("outputs") / "procedural_mario_v2_blind"),
        help="(--combined) Mario v2 blind directory.",
    )
    p.add_argument("--mario-maps-file", default=None, help="(--combined) Override Mario maps.json.")
    p.add_argument(
        "--sokoban-v1-dir",
        default=str(Path("results/v1.3") / "procedural_sokoban"),
        help="(--combined) Sokoban v1 directory.",
    )
    p.add_argument(
        "--sokoban-v2-dir",
        default=str(Path("outputs") / "procedural_sokoban_v2_blind"),
        help="(--combined) Sokoban v2 blind directory.",
    )
    p.add_argument("--sokoban-maps-file", default=None, help="(--combined) Override Sokoban maps.json.")

    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: outputs/procedural_paper_v1_v2_table or ..._combined).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip paper_v1_v2_compare.png and paper_v1_v2_truth_compare.png (no matplotlib).",
    )
    args = p.parse_args()
    if bool(args.combined) == bool(args.game):
        p.error("Specify exactly one of --game <mario|sokoban> or --combined.")
    if args.output_dir is None:
        args.output_dir = (
            "outputs/procedural_paper_v1_v2_table_combined"
            if args.combined
            else "outputs/procedural_paper_v1_v2_table"
        )
    if not args.combined:
        if args.v1_dir is None:
            args.v1_dir = str(Path("results/v1.3") / f"procedural_{args.game}")
        if args.v2_dir is None:
            args.v2_dir = str(Path("outputs") / f"procedural_{args.game}_v2_blind")
    return args


def _run_for_game(
    game_key: str,
    v1_csv: str | None,
    v2_csv: str | None,
    v1_dir: str | None,
    v2_dir: str | None,
    maps_file: str | None,
) -> dict:
    game = GAMES[game_key]
    defaults = game.DEFAULTS
    v1_path = v1_csv or _latest_csv(v1_dir)
    v2_path = v2_csv or _latest_csv(v2_dir)
    mpath = maps_file or defaults["maps_file"]

    truth_map = load_truth_map(mpath)
    v1_rows = load_results(v1_path, truth_map)
    v1_ok = [r for r in v1_rows if r["status"] == "ok"]
    v2_raw = _load_results_csv_rows(v2_path)

    v1_by_id = {r["map_id"]: r for r in v1_ok}
    v2_by_id = {r["map_id"]: r for r in v2_raw if r.get("status") == "ok"}
    common = sorted(set(v1_by_id) & set(v2_by_id))
    if not common:
        raise RuntimeError(f"[{game_key}] No overlapping map_id between v1 and v2 CSV.")

    v1_stats = _belief_group_stats(v1_ok, "llm_believed_creator")
    v2_stats = _v2_stats_by_v1_belief(v1_by_id, v2_by_id, common)
    return {
        "v1_stats": v1_stats,
        "v2_stats": v2_stats,
        "v1_truth_stats": _truth_group_stats(v1_ok),
        "v2_truth_stats": _v2_stats_by_truth(v2_by_id, truth_map, common),
        "v1_csv": v1_path,
        "v2_csv": v2_path,
        "n_maps_v1_ok": len(v1_ok),
        "n_maps_v2_ok": len([r for r in v2_raw if r.get("status") == "ok"]),
        "n_maps_paired": len(common),
        "v1_group_counts": v1_stats["counts"],
        "v2_group_counts": v2_stats["counts"],
    }


def _truth_group_stats(rows: list[dict]) -> dict:
    """Group by ground-truth creator (Human = first 15 map ids in maps.json, AI = rest)."""
    return _belief_group_stats(rows, "true_creator")


def _v2_stats_by_truth(
    v2_by_id: dict[str, dict],
    truth_map: dict[str, str],
    map_ids: list[str],
) -> dict:
    groups: dict[str, dict[str, list[float]]] = {
        "Human": {m: [] for m, _ in METRICS},
        "AI": {m: [] for m, _ in METRICS},
    }
    counts = {"Human": 0, "AI": 0, "Other": 0}
    for mid_map in map_ids:
        t = truth_map.get(mid_map, "")
        if t not in ("Human", "AI"):
            counts["Other"] += 1
            continue
        v2 = v2_by_id.get(mid_map)
        if not v2:
            continue
        counts[t] += 1
        for metric_id, _col in METRICS:
            colname = CSV_METRIC_COL[metric_id]
            val = to_number(v2.get(colname, ""))
            if val is not None:
                groups[t][metric_id].append(float(val))
    return {"groups": groups, "counts": counts}


def _merge_group_stats(a: dict, b: dict) -> dict:
    """Concatenate per-bin metric value lists from two games (paper-style pooling)."""
    groups: dict[str, dict[str, list[float]]] = {
        "Human": {mid: [] for mid, _ in METRICS},
        "AI": {mid: [] for mid, _ in METRICS},
    }
    for bin_name in ("Human", "AI"):
        for mid, _ in METRICS:
            groups[bin_name][mid] = a["groups"][bin_name][mid] + b["groups"][bin_name][mid]
    counts = {
        "Human": a["counts"]["Human"] + b["counts"]["Human"],
        "AI": a["counts"]["AI"] + b["counts"]["AI"],
        "Other": a["counts"]["Other"] + b["counts"]["Other"],
    }
    return {"groups": groups, "counts": counts}


def _latest_csv(directory: str) -> str:
    out = Path(directory)
    c = sorted(out.glob("results_*.csv"))
    if not c:
        raise FileNotFoundError(f"No results_*.csv under {out}")
    return str(c[-1])


def _load_results_csv_rows(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def _belief_group_stats(rows: list[dict], belief_key: str) -> dict:
    """Group by Human/AI belief; collect per-map metric values."""
    groups: dict[str, dict[str, list[float]]] = {
        "Human": {m: [] for m, _ in METRICS},
        "AI": {m: [] for m, _ in METRICS},
    }
    counts = {"Human": 0, "AI": 0, "Other": 0}
    for row in rows:
        b = row.get(belief_key, "")
        if b not in ("Human", "AI"):
            counts["Other"] += 1
            continue
        counts[b] += 1
        for mid, _ in METRICS:
            v = row.get(mid)
            if v is not None:
                groups[b][mid].append(float(v))
    return {"groups": groups, "counts": counts}


def _v2_stats_by_v1_belief(
    v1_by_id: dict[str, dict],
    v2_by_id: dict[str, dict],
    map_ids: list[str],
) -> dict:
    groups: dict[str, dict[str, list[float]]] = {
        "Human": {m: [] for m, _ in METRICS},
        "AI": {m: [] for m, _ in METRICS},
    }
    counts = {"Human": 0, "AI": 0, "Other": 0}
    for mid_map in map_ids:
        v1 = v1_by_id[mid_map]
        v2 = v2_by_id[mid_map]
        b = v1.get("llm_believed_creator", "")
        if b not in ("Human", "AI"):
            counts["Other"] += 1
            continue
        counts[b] += 1
        for metric_id, col in METRICS:
            colname = CSV_METRIC_COL[metric_id]
            val = to_number(v2.get(colname, ""))
            if val is not None:
                groups[b][metric_id].append(float(val))
    return {"groups": groups, "counts": counts}


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    sd = statistics.pstdev(values)
    return m, sd


def build_table_rows(v1_stats: dict, v2_stats: dict) -> list[dict]:
    rows = []
    for metric_id, _csv in METRICS:
        ph_m, ph_sd = PAPER_PERCEIVED_HUMAN[metric_id]
        pa_m, pa_sd = PAPER_PERCEIVED_AI[metric_id]

        v1h_m, v1h_sd = _mean_sd(v1_stats["groups"]["Human"][metric_id])
        v1a_m, v1a_sd = _mean_sd(v1_stats["groups"]["AI"][metric_id])

        v2h_m, v2h_sd = _mean_sd(v2_stats["groups"]["Human"][metric_id])
        v2a_m, v2a_sd = _mean_sd(v2_stats["groups"]["AI"][metric_id])

        v1_h_list = v1_stats["groups"]["Human"][metric_id]
        v1_a_list = v1_stats["groups"]["AI"][metric_id]
        v2_h_list = v2_stats["groups"]["Human"][metric_id]
        v2_a_list = v2_stats["groups"]["AI"][metric_id]
        _, v1_p = mann_whitney_u_two_sided(v1_h_list, v1_a_list)
        _, v2_p = mann_whitney_u_two_sided(v2_h_list, v2_a_list)

        rows.append({
            "metric_id": metric_id,
            "title": METRIC_TITLES[metric_id],
            "paper_perceived_human_mean": ph_m,
            "paper_perceived_human_sd": ph_sd,
            "paper_perceived_ai_mean": pa_m,
            "paper_perceived_ai_sd": pa_sd,
            "paper_mann_whitney_p": None,
            "v1_believed_human_mean": v1h_m,
            "v1_believed_human_sd": v1h_sd,
            "v1_believed_ai_mean": v1a_m,
            "v1_believed_ai_sd": v1a_sd,
            "v1_mann_whitney_p": v1_p,
            "v1_n_human": len(v1_h_list),
            "v1_n_ai": len(v1_a_list),
            "v2_believed_human_mean": v2h_m,
            "v2_believed_human_sd": v2h_sd,
            "v2_believed_ai_mean": v2a_m,
            "v2_believed_ai_sd": v2a_sd,
            "v2_mann_whitney_p": v2_p,
            "v2_n_human": len(v2_h_list),
            "v2_n_ai": len(v2_a_list),
        })
    return rows


def build_truth_table_rows(v1_truth: dict, v2_truth: dict) -> list[dict]:
    """Mean(SD) and Mann–Whitney p by **ground truth** Human vs AI (maps.json order)."""
    rows = []
    for metric_id, _csv in METRICS:
        v1h_m, v1h_sd = _mean_sd(v1_truth["groups"]["Human"][metric_id])
        v1a_m, v1a_sd = _mean_sd(v1_truth["groups"]["AI"][metric_id])
        v2h_m, v2h_sd = _mean_sd(v2_truth["groups"]["Human"][metric_id])
        v2a_m, v2a_sd = _mean_sd(v2_truth["groups"]["AI"][metric_id])
        v1_h_list = v1_truth["groups"]["Human"][metric_id]
        v1_a_list = v1_truth["groups"]["AI"][metric_id]
        v2_h_list = v2_truth["groups"]["Human"][metric_id]
        v2_a_list = v2_truth["groups"]["AI"][metric_id]
        _, v1_p = mann_whitney_u_two_sided(v1_h_list, v1_a_list)
        _, v2_p = mann_whitney_u_two_sided(v2_h_list, v2_a_list)
        rows.append({
            "metric_id": metric_id,
            "title": METRIC_TITLES[metric_id],
            "v1_truth_human_mean": v1h_m,
            "v1_truth_human_sd": v1h_sd,
            "v1_truth_ai_mean": v1a_m,
            "v1_truth_ai_sd": v1a_sd,
            "v1_truth_mann_whitney_p": v1_p,
            "v1_truth_n_human": len(v1_h_list),
            "v1_truth_n_ai": len(v1_a_list),
            "v2_truth_human_mean": v2h_m,
            "v2_truth_human_sd": v2h_sd,
            "v2_truth_ai_mean": v2a_m,
            "v2_truth_ai_sd": v2a_sd,
            "v2_truth_mann_whitney_p": v2_p,
            "v2_truth_n_human": len(v2_h_list),
            "v2_truth_n_ai": len(v2_a_list),
        })
    return rows


def _fmt(mean: float | None, sd: float | None) -> str:
    if mean is None:
        return "—"
    if sd is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} ({sd:.2f})"


def print_table(label: str, rows: list[dict], v1_counts: dict, v2_counts: dict) -> None:
    print("=" * 120)
    print(f"[{label}] Paper Table 1 style vs v1 vs v2 (v2 grouped by v1 perceived creator)")
    print(f"  v1 belief bins: Human n={v1_counts['Human']}, AI n={v1_counts['AI']}, other={v1_counts['Other']}")
    print(f"  v2 (same bins): Human n={v2_counts['Human']}, AI n={v2_counts['AI']}, other={v2_counts['Other']}")
    print("=" * 120)
    hdr = (
        f"{'Metric':<28} | "
        f"{'Paper H':>12} | {'Paper AI':>12} | "
        f"{'v1 H':>12} | {'v1 AI':>12} | {'v1 p':>7} | "
        f"{'v2 H':>12} | {'v2 AI':>12} | {'v2 p':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (
            f"{r['title'][:28]:<28} | "
            f"{_fmt(r['paper_perceived_human_mean'], r['paper_perceived_human_sd']):>12} | "
            f"{_fmt(r['paper_perceived_ai_mean'], r['paper_perceived_ai_sd']):>12} | "
            f"{_fmt(r['v1_believed_human_mean'], r['v1_believed_human_sd']):>12} | "
            f"{_fmt(r['v1_believed_ai_mean'], r['v1_believed_ai_sd']):>12} | "
            f"{_fmt_p(r['v1_mann_whitney_p']):>7} | "
            f"{_fmt(r['v2_believed_human_mean'], r['v2_believed_human_sd']):>12} | "
            f"{_fmt(r['v2_believed_ai_mean'], r['v2_believed_ai_sd']):>12} | "
            f"{_fmt_p(r['v2_mann_whitney_p']):>7}"
        )
        print(line)
    print()
    print("Legend: Mean (SD). v1_p / v2_p = Mann–Whitney U (two-sided) Human-bin vs AI-bin (per-map scores).")
    print("Paper has no raw scores here → no p from this script.")
    print("v2 uses the same perceived-source bins as v1 (belief from v1 full prompt) for fair comparison.")


def print_truth_table(label: str, rows: list[dict], v1_counts: dict, v2_counts: dict) -> None:
    print("=" * 100)
    print(f"[{label}] Ground truth: v1 vs v2 Mean(SD) by true creator (maps.json: first 15 Human, rest AI)")
    print(f"  v1 truth bins: Human n={v1_counts['Human']}, AI n={v1_counts['AI']}, other={v1_counts['Other']}")
    print(f"  v2 truth bins: Human n={v2_counts['Human']}, AI n={v2_counts['AI']}, other={v2_counts['Other']}")
    print("=" * 100)
    hdr = f"{'Metric':<28} | {'v1 H':>12} | {'v1 AI':>12} | {'v1 p':>7} | {'v2 H':>12} | {'v2 AI':>12} | {'v2 p':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (
            f"{r['title'][:28]:<28} | "
            f"{_fmt(r['v1_truth_human_mean'], r['v1_truth_human_sd']):>12} | "
            f"{_fmt(r['v1_truth_ai_mean'], r['v1_truth_ai_sd']):>12} | "
            f"{_fmt_p(r['v1_truth_mann_whitney_p']):>7} | "
            f"{_fmt(r['v2_truth_human_mean'], r['v2_truth_human_sd']):>12} | "
            f"{_fmt(r['v2_truth_ai_mean'], r['v2_truth_ai_sd']):>12} | "
            f"{_fmt_p(r['v2_truth_mann_whitney_p']):>7}"
        )
        print(line)
    print()
    print(
        "Legend: Mean (SD). v1_p / v2_p = Mann–Whitney U (two-sided) Human-truth vs AI-truth (per-map scores). "
        "Paper Table 1 is perceived-source only; no paper column here."
    )
    print()
def _fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Metric | Paper H | Paper AI | v1 H | v1 AI | v1 p | v2 H | v2 AI | v2 p |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| "
            + f"{r['title']} | "
            + f"{_fmt(r['paper_perceived_human_mean'], r['paper_perceived_human_sd'])} | "
            + f"{_fmt(r['paper_perceived_ai_mean'], r['paper_perceived_ai_sd'])} | "
            + f"{_fmt(r['v1_believed_human_mean'], r['v1_believed_human_sd'])} | "
            + f"{_fmt(r['v1_believed_ai_mean'], r['v1_believed_ai_sd'])} | "
            + f"{_fmt_p(r['v1_mann_whitney_p'])} | "
            + f"{_fmt(r['v2_believed_human_mean'], r['v2_believed_human_sd'])} | "
            + f"{_fmt(r['v2_believed_ai_mean'], r['v2_believed_ai_sd'])} | "
            + f"{_fmt_p(r['v2_mann_whitney_p'])} |"
        )
    return "\n".join(lines)


def _markdown_p_note(combined: bool) -> str:
    pool = ""
    if combined:
        pool = (
            "**Pooled (Mario + Sokoban)**: Mann–Whitney는 두 게임 맵을 한 풀에 넣어 Human-bin vs AI-bin을 비교합니다 "
            "(논문 Table 1과 동일한 ‘합산’ 프레이밍). 다만 두 게임 난이도·스케일이 달라 **교환 가능한 표본**은 아님.\n\n"
        )
    return (
        "### Mann–Whitney p (v1 / v2)\n\n"
        + pool
        + "- **v1 p**: full-prompt 맵 점수, Human-believed vs AI-believed 구간.\n"
        + "- **v2 p**: blind 맵 점수, **동일 구간**(구간 = 해당 맵의 v1 `creator_belief`).\n"
        + "- **의미**: 두 구간 분포가 우연히 이 정도로 갈릴 확률. 맵 수·불균형·구간 정의 등으로 **탐색적 지표**로만 해석하는 것이 좋다.\n"
    )


def markdown_truth_table(rows: list[dict]) -> str:
    lines = [
        "| Metric | v1 Human (truth) | v1 AI (truth) | v1 p | v2 Human (truth) | v2 AI (truth) | v2 p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| "
            + f"{r['title']} | "
            + f"{_fmt(r['v1_truth_human_mean'], r['v1_truth_human_sd'])} | "
            + f"{_fmt(r['v1_truth_ai_mean'], r['v1_truth_ai_sd'])} | "
            + f"{_fmt_p(r['v1_truth_mann_whitney_p'])} | "
            + f"{_fmt(r['v2_truth_human_mean'], r['v2_truth_human_sd'])} | "
            + f"{_fmt(r['v2_truth_ai_mean'], r['v2_truth_ai_sd'])} | "
            + f"{_fmt_p(r['v2_truth_mann_whitney_p'])} |"
        )
    return "\n".join(lines)


def _markdown_truth_p_note(combined: bool) -> str:
    pool = ""
    if combined:
        pool = (
            "**Pooled (Mario + Sokoban)**: Mann–Whitney는 두 게임 맵을 한 풀에 넣어 **진실 Human vs AI** 구간을 비교합니다. "
            "논문 Table 1의 ‘인지된 출처’와는 다른 프레이밍입니다.\n\n"
        )
    return (
        "### Mann–Whitney p (truth bins, v1 / v2)\n\n"
        + pool
        + "- **v1 p**: full-prompt 점수, Human-made vs AI-generated 맵(동일 `maps.json` 순서 규칙).\n"
        + "- **v2 p**: blind 점수, 동일 진실 구간.\n"
    )
def _save_comparison_plots(table: list[dict], path: Path, pooled: bool = False) -> None:
    import matplotlib.pyplot as plt

    n = len(table)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    colors = ("#4477AA", "#CC6677", "#228833")
    labels = ("Paper", "v1", "v2")
    width = 0.24

    for ax, r in zip(axes, table):
        # Human bin (x≈0): three bars; AI bin (x≈1): three bars
        xh = [-width, 0.0, width]
        xa = [1.0 - width, 1.0, 1.0 + width]
        mh = [
            r["paper_perceived_human_mean"],
            r["v1_believed_human_mean"],
            r["v2_believed_human_mean"],
        ]
        sh = [
            r["paper_perceived_human_sd"] or 0.0,
            r["v1_believed_human_sd"] or 0.0,
            r["v2_believed_human_sd"] or 0.0,
        ]
        ma = [
            r["paper_perceived_ai_mean"],
            r["v1_believed_ai_mean"],
            r["v2_believed_ai_mean"],
        ]
        sa = [
            r["paper_perceived_ai_sd"] or 0.0,
            r["v1_believed_ai_sd"] or 0.0,
            r["v2_believed_ai_sd"] or 0.0,
        ]

        for i, (lab, c) in enumerate(zip(labels, colors)):
            ax.bar(
                xh[i],
                mh[i],
                width * 0.92,
                yerr=sh[i],
                label=lab if ax is axes[0] else None,
                color=c,
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.92,
            )
            ax.bar(
                xa[i],
                ma[i],
                width * 0.92,
                yerr=sa[i],
                color=c,
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.92,
            )

        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["Perceived Human bin", "Perceived AI bin"])
        ax.set_ylabel("Mean score (1–5)")
        ax.set_title(r["title"])
        ax.set_ylim(0, 5.5)
        p_txt = f"v1 p={_fmt_p(r['v1_mann_whitney_p'])}  |  v2 p={_fmt_p(r['v2_mann_whitney_p'])}"
        ax.text(0.02, 0.98, p_txt, transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    fig.suptitle(
        "Paper vs v1 vs v2 — Mario + Sokoban pooled (means ± SD)"
        if pooled
        else "Paper vs v1 vs v2 — means ± SD (same two bins)",
        fontsize=11,
        y=1.01,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_truth_comparison_plots(table: list[dict], path: Path, pooled: bool = False) -> None:
    import matplotlib.pyplot as plt

    n = len(table)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 3.0 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    colors = ("#CC6677", "#228833")
    labels = ("v1", "v2")
    width = 0.32

    for ax, r in zip(axes, table):
        xh = [-width / 2, width / 2]
        xa = [1.0 - width / 2, 1.0 + width / 2]
        mh = [r["v1_truth_human_mean"], r["v2_truth_human_mean"]]
        sh = [r["v1_truth_human_sd"] or 0.0, r["v2_truth_human_sd"] or 0.0]
        ma = [r["v1_truth_ai_mean"], r["v2_truth_ai_mean"]]
        sa = [r["v1_truth_ai_sd"] or 0.0, r["v2_truth_ai_sd"] or 0.0]

        for i, (lab, c) in enumerate(zip(labels, colors)):
            ax.bar(
                xh[i],
                mh[i],
                width * 0.88,
                yerr=sh[i],
                label=lab if ax is axes[0] else None,
                color=c,
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.92,
            )
            ax.bar(
                xa[i],
                ma[i],
                width * 0.88,
                yerr=sa[i],
                color=c,
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.92,
            )

        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["Human-made (truth)", "AI-generated (truth)"])
        ax.set_ylabel("Mean score (1–5)")
        ax.set_title(r["title"])
        ax.set_ylim(0, 5.5)
        p_txt = f"v1 p={_fmt_p(r['v1_truth_mann_whitney_p'])}  |  v2 p={_fmt_p(r['v2_truth_mann_whitney_p'])}"
        ax.text(0.02, 0.98, p_txt, transform=ax.transAxes, va="top", fontsize=9, family="monospace")

    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    fig.suptitle(
        "v1 vs v2 — ground truth bins, Mario + Sokoban pooled (means ± SD)"
        if pooled
        else "v1 vs v2 — ground truth bins (means ± SD)",
        fontsize=11,
        y=1.01,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
