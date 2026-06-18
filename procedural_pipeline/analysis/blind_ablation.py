"""Ablation study: does the creator-belief halo effect disappear under blind eval?

Workflow this script answers:

  v1 (full):  prompt asks for creator_belief + confidence + 5 metrics.
              The LLM forms a belief about authorship FIRST, then scores.
  v2 (blind): same video, same Auto-CoT eval steps, same model, but the prompt
              asks ONLY for the 5 metrics. The LLM cannot anchor on a self-formed
              creator belief.

For each map we know:
  - v1 aggregated creator_belief  (Human-designed / AI-generated / Tie)
  - v1 aggregated 5-metric scores
  - v2 aggregated 5-metric scores  (no belief at all)

We then RE-GROUP v2 scores using v1's belief labels, so we can directly compare:

    v1 scores grouped by v1 belief   <-- the original biased view
    v2 scores grouped by v1 belief   <-- "if we had used a debiased rater,
                                         what would those same maps look like?"

If the v2 gap (Believed-Human vs Believed-AI) shrinks toward zero, the bias was
caused by the prompt asking about authorship. If the gap stays, the bias is in
the model's perception itself, independent of the prompt structure.

Usage:
  python -m procedural_pipeline.analyze_blind_ablation --game mario \
      --v1-csv results/v1.3/procedural_mario/results_20260422-214139.csv \
      --v2-csv outputs/procedural_mario_v2_blind/results_<latest>.csv

If --v1-csv / --v2-csv are not given, the script picks the newest results_*.csv
under each --v1-dir / --v2-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from procedural_pipeline.games import GAMES
from procedural_pipeline.analysis.results import (
    METRICS,
    format_number,
    load_truth_map,
    mann_whitney_u_two_sided,
    metric_number,
    normalize_creator_label,
    safe_mean,
    to_number,
)


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    game = GAMES[args.game]
    defaults = game.DEFAULTS

    v1_csv = args.v1_csv or _latest_csv(args.v1_dir)
    v2_csv = args.v2_csv or _latest_csv(args.v2_dir)
    maps_file = args.maps_file or defaults["maps_file"]

    truth_map = load_truth_map(maps_file)
    v1_rows = _load_csv(v1_csv)
    v2_rows = _load_csv(v2_csv)
    v1_by_id = {r["map_id"]: r for r in v1_rows}
    v2_by_id = {r["map_id"]: r for r in v2_rows}

    common_ids = sorted(set(v1_by_id) & set(v2_by_id))
    if not common_ids:
        raise RuntimeError("No overlapping map_id between v1 and v2 CSVs.")

    paired_rows = []
    for map_id in common_ids:
        v1 = v1_by_id[map_id]
        v2 = v2_by_id[map_id]
        v1_belief = normalize_creator_label(v1.get("creator_belief", ""))
        paired_rows.append({
            "map_id": map_id,
            "true_creator": truth_map.get(map_id, ""),
            "v1_creator_belief": v1_belief or "Tie/Other",
            "v1_creator_belief_raw": v1.get("creator_belief", ""),
            "v1_confidence": to_number(v1.get("confidence_level", "")),
            "v2_observation_only": True,
            **{f"v1_{field}": metric_number(v1, field) for _, field in METRICS},
            **{f"v2_{field}": metric_number(v2, field) for _, field in METRICS},
        })

    summary = {
        "game": args.game,
        "v1_csv": str(v1_csv),
        "v2_csv": str(v2_csv),
        "num_v1_maps": len(v1_rows),
        "num_v2_maps": len(v2_rows),
        "num_common_maps": len(paired_rows),
        "by_v1_belief": _by_belief_summary(paired_rows),
        "by_truth": _by_truth_summary(paired_rows),
        "per_map": paired_rows,
    }

    print_summary(summary)
    _save(args.output_dir, paired_rows, summary)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--game", required=True, choices=sorted(GAMES.keys()))
    prelim, _ = pre.parse_known_args()
    game = GAMES[prelim.game]

    parser = argparse.ArgumentParser(
        description="Compare v1 (full) vs v2 (blind) scores grouped by v1's creator belief.",
        parents=[pre],
    )
    parser.add_argument(
        "--v1-csv",
        default=None,
        help="v1 (full prompt) results CSV. If omitted, latest results_*.csv under --v1-dir is used.",
    )
    parser.add_argument(
        "--v1-dir",
        default=str(Path("results/v1.3") / f"procedural_{prelim.game}"),
        help="Fallback dir to look up v1 results CSV (default: results/v1.3/procedural_<game>).",
    )
    parser.add_argument(
        "--v2-csv",
        default=None,
        help="v2 (blind prompt) results CSV. If omitted, latest results_*.csv under --v2-dir is used.",
    )
    parser.add_argument(
        "--v2-dir",
        default=str(Path("outputs") / f"procedural_{prelim.game}_v2_blind"),
        help="Fallback dir to look up v2 results CSV (default: outputs/procedural_<game>_v2_blind).",
    )
    parser.add_argument("--maps-file", default=None, help=f"(default: {game.DEFAULTS['maps_file']})")
    parser.add_argument("--output-dir", default="outputs/procedural_blind_ablation")
    return parser.parse_args()


# --------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------- #
def _latest_csv(directory: str) -> str:
    out = Path(directory)
    candidates = sorted(out.glob("results_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No results_*.csv under {out}")
    return str(candidates[-1])


def _load_csv(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                rows.append(row)
    return rows


# --------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------- #
def _by_belief_summary(paired_rows: list[dict]) -> dict:
    """Group paired rows by v1's creator belief, then compare v1 vs v2 means.

    The headline number per metric is the gap between (Human-believed maps) and
    (AI-believed maps). If v2's gap is much smaller than v1's gap, the bias was
    prompt-induced. If it's the same, the model's visual perception is biased
    on its own.
    """
    groups = {"Human": [], "AI": [], "Tie/Other": []}
    for row in paired_rows:
        belief = row["v1_creator_belief"]
        if belief in groups:
            groups[belief].append(row)
        else:
            groups["Tie/Other"].append(row)

    out = {"counts": {k: len(v) for k, v in groups.items()}, "metrics": {}}

    for _, field in METRICS:
        v1_h = [r[f"v1_{field}"] for r in groups["Human"] if r[f"v1_{field}"] is not None]
        v1_a = [r[f"v1_{field}"] for r in groups["AI"] if r[f"v1_{field}"] is not None]
        v2_h = [r[f"v2_{field}"] for r in groups["Human"] if r[f"v2_{field}"] is not None]
        v2_a = [r[f"v2_{field}"] for r in groups["AI"] if r[f"v2_{field}"] is not None]

        v1_u, v1_p = mann_whitney_u_two_sided(v1_h, v1_a)
        v2_u, v2_p = mann_whitney_u_two_sided(v2_h, v2_a)

        v1_gap = _gap(safe_mean(v1_h), safe_mean(v1_a))
        v2_gap = _gap(safe_mean(v2_h), safe_mean(v2_a))
        bias_reduction_pct = None
        if v1_gap is not None and v2_gap is not None and v1_gap != 0:
            bias_reduction_pct = (1 - abs(v2_gap) / abs(v1_gap)) * 100

        out["metrics"][field] = {
            "v1_human_mean": safe_mean(v1_h),
            "v1_ai_mean": safe_mean(v1_a),
            "v1_gap": v1_gap,
            "v1_p_value": v1_p,
            "v2_human_mean": safe_mean(v2_h),
            "v2_ai_mean": safe_mean(v2_a),
            "v2_gap": v2_gap,
            "v2_p_value": v2_p,
            "bias_reduction_pct": bias_reduction_pct,
        }
    return out


def _by_truth_summary(paired_rows: list[dict]) -> dict:
    """Cross-check: how does v2 differ from v1 when we group by GROUND TRUTH instead?

    This isolates whether the prompt removal also changes how the model rates
    actually-Human vs actually-AI maps. If v1 gap and v2 gap are similar here,
    the model can still tell them apart visually — what changes is just the
    score magnitudes assigned through the belief lens.
    """
    groups = {"Human": [], "AI": []}
    for row in paired_rows:
        if row["true_creator"] in groups:
            groups[row["true_creator"]].append(row)

    out = {"counts": {k: len(v) for k, v in groups.items()}, "metrics": {}}
    for _, field in METRICS:
        v1_h = [r[f"v1_{field}"] for r in groups["Human"] if r[f"v1_{field}"] is not None]
        v1_a = [r[f"v1_{field}"] for r in groups["AI"] if r[f"v1_{field}"] is not None]
        v2_h = [r[f"v2_{field}"] for r in groups["Human"] if r[f"v2_{field}"] is not None]
        v2_a = [r[f"v2_{field}"] for r in groups["AI"] if r[f"v2_{field}"] is not None]
        out["metrics"][field] = {
            "v1_human_mean": safe_mean(v1_h),
            "v1_ai_mean": safe_mean(v1_a),
            "v1_gap": _gap(safe_mean(v1_h), safe_mean(v1_a)),
            "v2_human_mean": safe_mean(v2_h),
            "v2_ai_mean": safe_mean(v2_a),
            "v2_gap": _gap(safe_mean(v2_h), safe_mean(v2_a)),
        }
    return out


def _gap(a, b):
    if a is None or b is None:
        return None
    return a - b


# --------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------- #
def print_summary(summary: dict) -> None:
    print("=" * 78)
    print(f"[{summary['game']}] Blind ablation (v2) vs full (v1)")
    print(f"  v1 csv : {summary['v1_csv']}")
    print(f"  v2 csv : {summary['v2_csv']}")
    print(f"  paired maps: {summary['num_common_maps']}  (v1={summary['num_v1_maps']}, v2={summary['num_v2_maps']})")
    print("=" * 78)

    bb = summary["by_v1_belief"]
    print()
    print(">>> Grouped by v1's creator belief  (the bias-of-interest)")
    print(f"    counts: Human-believed={bb['counts'].get('Human', 0)}  "
          f"AI-believed={bb['counts'].get('AI', 0)}  "
          f"Tie/Other={bb['counts'].get('Tie/Other', 0)}")
    print()
    print(f"    {'metric':<13} | "
          f"{'v1 H':>6} {'v1 A':>6} {'v1Δ':>7} {'v1 p':>6}  ||  "
          f"{'v2 H':>6} {'v2 A':>6} {'v2Δ':>7} {'v2 p':>6}  ||  "
          f"{'gap shrink':>10}")
    print(f"    {'-'*13}-+-{'-'*6}-{'-'*6}-{'-'*7}-{'-'*6}-+-+-{'-'*6}-{'-'*6}-{'-'*7}-{'-'*6}-+-+-{'-'*10}")
    for field, m in bb["metrics"].items():
        shrink = (
            f"{m['bias_reduction_pct']:+6.1f}%"
            if m["bias_reduction_pct"] is not None
            else "   NA "
        )
        print(
            f"    {field:<13} | "
            f"{format_number(m['v1_human_mean']):>6} "
            f"{format_number(m['v1_ai_mean']):>6} "
            f"{format_number(m['v1_gap']):>7} "
            f"{format_number(m['v1_p_value']):>6}     "
            f"{format_number(m['v2_human_mean']):>6} "
            f"{format_number(m['v2_ai_mean']):>6} "
            f"{format_number(m['v2_gap']):>7} "
            f"{format_number(m['v2_p_value']):>6}     "
            f"{shrink:>10}"
        )

    bt = summary["by_truth"]
    print()
    print(">>> Grouped by GROUND TRUTH  (sanity check: visual difference, not belief-driven)")
    print(f"    counts: Human-truth={bt['counts'].get('Human', 0)}  AI-truth={bt['counts'].get('AI', 0)}")
    print()
    print(f"    {'metric':<13} | {'v1 H':>6} {'v1 A':>6} {'v1Δ':>7}  ||  {'v2 H':>6} {'v2 A':>6} {'v2Δ':>7}")
    print(f"    {'-'*13}-+-{'-'*6}-{'-'*6}-{'-'*7}-+-+-{'-'*6}-{'-'*6}-{'-'*7}")
    for field, m in bt["metrics"].items():
        print(
            f"    {field:<13} | "
            f"{format_number(m['v1_human_mean']):>6} "
            f"{format_number(m['v1_ai_mean']):>6} "
            f"{format_number(m['v1_gap']):>7}     "
            f"{format_number(m['v2_human_mean']):>6} "
            f"{format_number(m['v2_ai_mean']):>6} "
            f"{format_number(m['v2_gap']):>7}"
        )

    print()
    print("    Reading guide:")
    print("      v1Δ = mean(believed-Human) - mean(believed-AI) under the FULL prompt")
    print("      v2Δ = same maps' means, but scored under the BLIND prompt")
    print("      gap shrink = how much smaller v2's gap is, relative to v1.")
    print("        Large positive shrink (e.g. >50%) => the gap was prompt-induced (belief halo).")
    print("        Negligible shrink                 => the model rates them differently regardless.")


def _save(output_dir: str, paired_rows: list[dict], summary: dict) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if paired_rows:
        fieldnames = list(paired_rows[0].keys())
        with (out / "blind_ablation_paired.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in paired_rows:
                writer.writerow(row)

    (out / "blind_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[{summary['game']}] wrote blind_ablation_paired.csv + blind_ablation_summary.json -> {out}")


if __name__ == "__main__":
    main()
