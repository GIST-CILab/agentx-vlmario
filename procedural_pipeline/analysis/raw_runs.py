"""Run-level analysis for multi-run evaluation outputs.

Each map has N (default 20) independent LLM judgments in its raw JSON file.
This script expands those to one row per run, giving much higher statistical
power than the per-map aggregate CSV and exposing:

  - Run-level classification accuracy / FPR / FNR (treating each run as a vote)
  - Per-map consistency (majority-fraction and entropy over the N verdicts)
  - Confidence calibration (accuracy stratified by self-reported confidence)
  - Run-level Mann-Whitney U tests for experience metrics (truth vs belief)
  - Tie / failure statistics

Usage:
  python -m procedural_pipeline.analyze_raw_runs --game mario
  python -m procedural_pipeline.analyze_raw_runs --game sokoban --raw-dir outputs/procedural_sokoban/raw
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from procedural_pipeline.games import GAMES
from procedural_pipeline.analysis.results import (
    METRICS,
    METRIC_ALIASES,
    analyze_creator_classification,
    format_number,
    load_truth_map,
    mann_whitney_u_two_sided,
    normalize_creator_label,
    safe_mean,
)


FIELD_TO_DISPLAY = {field: display for display, field in METRICS}


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    game = GAMES[args.game]
    defaults = game.DEFAULTS

    raw_dir = Path(args.raw_dir or Path(defaults["output_dir"]) / "raw")
    maps_file = args.maps_file or defaults["maps_file"]

    truth_map = load_truth_map(maps_file)
    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        raise FileNotFoundError(f"No raw JSON files under {raw_dir}")

    run_rows, map_summaries = load_raw_runs(raw_files, truth_map)
    if not run_rows:
        raise RuntimeError(f"No parseable runs found under {raw_dir}")

    valid_runs = [r for r in run_rows if r["llm_believed_creator"] in {"Human", "AI"}]

    summary = {
        "game": args.game,
        "raw_dir": str(raw_dir),
        "num_raw_files": len(raw_files),
        "num_runs_total": len(run_rows),
        "num_runs_valid": len(valid_runs),
        "num_runs_failed": sum(1 for r in run_rows if not r["parsed_ok"]),
        "run_level_rq1": analyze_creator_classification(valid_runs),
        "run_level_rq4": analyze_run_level_experience(valid_runs),
        "per_map_consistency": summarize_consistency(map_summaries),
        "confidence_calibration": analyze_confidence_calibration(valid_runs),
        "map_summaries": map_summaries,
    }

    print_summary(summary)
    save_outputs(args.output_dir, run_rows, summary)


def parse_args() -> argparse.Namespace:
    # Use a standalone parser (no pipeline auto-run — this script only consumes
    # already-generated raw JSON files).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--game", required=True, choices=sorted(GAMES.keys()))
    prelim, _ = pre.parse_known_args()
    game = GAMES[prelim.game]

    parser = argparse.ArgumentParser(
        description="Per-run analysis of multi-run LLM evaluation raw outputs.",
        parents=[pre],
    )
    parser.add_argument("--raw-dir", default=None, help=f"Raw outputs dir (defaults to {Path(game.DEFAULTS['output_dir'])/'raw'}).")
    parser.add_argument("--maps-file", default=None, help=f"Maps file (defaults to {game.DEFAULTS['maps_file']}).")
    parser.add_argument("--output-dir", default="outputs/procedural_analysis_runs", help="Where to write runs.csv and summary.json.")
    return parser.parse_args()


# --------------------------------------------------------------------- #
# Raw run loading
# --------------------------------------------------------------------- #
def load_raw_runs(
    raw_files: list[Path], truth_map: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    """Expand every raw map file into (run_rows, map_summaries).

    run_rows: one row per (map_id, run_index) — all runs, parsed or not.
    map_summaries: per-map stats (vote distribution, consistency, parsed count).
    """
    run_rows: list[dict] = []
    map_summaries: list[dict] = []

    for raw_file in raw_files:
        try:
            data = json.loads(raw_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] failed to read {raw_file}: {exc}")
            continue

        map_id = data.get("map_id") or raw_file.stem
        map_status = data.get("status", "")
        runs = (data.get("llm") or {}).get("runs") or data.get("runs") or []
        if not runs and isinstance(data.get("llm"), dict):
            runs = data["llm"].get("runs", [])

        parsed_in_map: list[dict] = []
        for run in runs:
            parsed = run.get("parsed_judgment")
            row = {
                "map_id": map_id,
                "true_creator": truth_map.get(map_id, ""),
                "run_index": run.get("run_index"),
                "map_status": map_status,
                "parsed_ok": parsed is not None,
                "error": str(run.get("error") or run.get("parse_error") or ""),
                "llm_believed_creator": "",
                "llm_believed_creator_raw": "",
                "confidence_level": None,
                "reasoning": "",
                "fun": None,
                "challenging": None,
                "frustrating": None,
                "surprising": None,
                "design": None,
            }
            if isinstance(parsed, dict):
                belief = parsed.get("creator_belief") or {}
                if isinstance(belief, dict):
                    raw_label = str(belief.get("value") or "").strip()
                    row["llm_believed_creator_raw"] = raw_label
                    row["llm_believed_creator"] = normalize_creator_label(raw_label)
                confidence = parsed.get("confidence_level") or {}
                if isinstance(confidence, dict):
                    row["confidence_level"] = _to_float(confidence.get("value"))
                row["reasoning"] = str(parsed.get("reasoning_for_creator_belief") or "")
                for _display, metric in METRICS:
                    item = _get_metric(parsed, metric)
                    if isinstance(item, dict):
                        row[metric] = _to_float(item.get("score"))
                parsed_in_map.append(row)

            run_rows.append(row)

        map_summaries.append(_summarize_map(map_id, truth_map.get(map_id, ""), runs, parsed_in_map))

    return run_rows, map_summaries


def _summarize_map(map_id: str, true_creator: str, runs: list, parsed_rows: list[dict]) -> dict:
    votes = {"Human": 0, "AI": 0, "Other": 0}
    confidences_by_label: dict[str, list[float]] = {"Human": [], "AI": []}
    for row in parsed_rows:
        label = row["llm_believed_creator"]
        if label in votes:
            votes[label] += 1
            if row["confidence_level"] is not None:
                confidences_by_label[label].append(row["confidence_level"])
        else:
            votes["Other"] += 1

    num_parsed = len(parsed_rows)
    n_runs = len(runs)
    labels_counted = {k: v for k, v in votes.items() if k in ("Human", "AI") and v > 0}
    if labels_counted:
        top = max(labels_counted.values())
        winners = [k for k, v in labels_counted.items() if v == top]
        if len(winners) == 1:
            majority_label = winners[0]
            majority_fraction = top / num_parsed if num_parsed else None
        else:
            # Tie on count -> break by total confidence (mirrors aggregator).
            conf_sums = {k: sum(confidences_by_label[k]) for k in winners}
            top_conf = max(conf_sums.values())
            conf_winners = [k for k, v in conf_sums.items() if v == top_conf]
            majority_label = conf_winners[0] if len(conf_winners) == 1 else "Tie"
            majority_fraction = top / num_parsed if num_parsed else None
    else:
        majority_label = ""
        majority_fraction = None

    entropy = _vote_entropy([votes["Human"], votes["AI"]])
    correct_vote_count = votes.get(true_creator, 0) if true_creator in ("Human", "AI") else 0

    return {
        "map_id": map_id,
        "true_creator": true_creator,
        "num_runs": n_runs,
        "num_parsed": num_parsed,
        "votes": votes,
        "majority_label": majority_label,
        "majority_fraction": majority_fraction,
        "entropy_bits": entropy,
        "agreement_with_truth_fraction": (correct_vote_count / num_parsed) if num_parsed else None,
    }


# --------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------- #
def analyze_run_level_experience(rows: list[dict]) -> dict:
    result = {}
    for display_name, field_name in METRICS:
        truth_human = _collect(rows, "true_creator", "Human", field_name)
        truth_ai = _collect(rows, "true_creator", "AI", field_name)
        belief_human = _collect(rows, "llm_believed_creator", "Human", field_name)
        belief_ai = _collect(rows, "llm_believed_creator", "AI", field_name)

        truth_u, truth_p = mann_whitney_u_two_sided(truth_human, truth_ai)
        belief_u, belief_p = mann_whitney_u_two_sided(belief_human, belief_ai)

        result[field_name] = {
            "truth_human_mean": safe_mean(truth_human),
            "truth_ai_mean": safe_mean(truth_ai),
            "truth_p_value": truth_p,
            "truth_u": truth_u,
            "truth_n_human": len(truth_human),
            "truth_n_ai": len(truth_ai),
            "belief_human_mean": safe_mean(belief_human),
            "belief_ai_mean": safe_mean(belief_ai),
            "belief_p_value": belief_p,
            "belief_u": belief_u,
            "belief_n_human": len(belief_human),
            "belief_n_ai": len(belief_ai),
            "belief_bias_detected": belief_p is not None and belief_p < 0.05,
        }
    return result


def summarize_consistency(map_summaries: list[dict]) -> dict:
    fractions = [m["majority_fraction"] for m in map_summaries if m["majority_fraction"] is not None]
    entropies = [m["entropy_bits"] for m in map_summaries if m["entropy_bits"] is not None]
    unanimous = sum(1 for m in map_summaries if m["majority_fraction"] == 1.0)
    close_calls = sum(
        1 for m in map_summaries
        if m["majority_fraction"] is not None and m["majority_fraction"] < 0.6
    )
    return {
        "num_maps": len(map_summaries),
        "majority_fraction_mean": safe_mean(fractions),
        "majority_fraction_min": min(fractions) if fractions else None,
        "majority_fraction_max": max(fractions) if fractions else None,
        "entropy_mean_bits": safe_mean(entropies),
        "num_unanimous_maps": unanimous,
        "num_close_call_maps": close_calls,  # majority < 60%
    }


def analyze_confidence_calibration(rows: list[dict]) -> dict:
    """For each confidence bucket (1/2/3), compute run-level accuracy.

    Tells us: does the LLM's self-reported confidence actually correlate with
    being right? If accuracy is flat across buckets -> confidence is miscalibrated.
    """
    buckets: dict[int, list[dict]] = {}
    for row in rows:
        if row["true_creator"] not in ("Human", "AI"):
            continue
        conf = row["confidence_level"]
        if conf is None:
            continue
        bucket = int(round(float(conf)))
        buckets.setdefault(bucket, []).append(row)

    result = {}
    for level in sorted(buckets):
        bucket_rows = buckets[level]
        correct = sum(1 for r in bucket_rows if r["llm_believed_creator"] == r["true_creator"])
        result[str(level)] = {
            "n": len(bucket_rows),
            "accuracy": correct / len(bucket_rows) if bucket_rows else None,
            "num_human_true": sum(1 for r in bucket_rows if r["true_creator"] == "Human"),
            "num_ai_true": sum(1 for r in bucket_rows if r["true_creator"] == "AI"),
        }
    return result


# --------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------- #
def print_summary(summary: dict) -> None:
    print(f"=== [{summary['game']}] Per-Run Analysis ===")
    print(f"  raw dir        : {summary['raw_dir']}")
    print(f"  maps (files)   : {summary['num_raw_files']}")
    print(f"  runs total     : {summary['num_runs_total']}")
    print(f"  runs parsed OK : {summary['num_runs_valid']}")
    print(f"  runs failed    : {summary['num_runs_failed']}")
    print()

    print("--- RQ1 (Run-level classification) ---")
    rq1 = summary["run_level_rq1"]
    if rq1["accuracy"] is None:
        print("  (no valid rows)")
    else:
        print(f"  Accuracy: {rq1['accuracy'] * 100:.1f}%")
        print(f"  FPR (Human→AI): {rq1['fpr'] * 100:.1f}%")
        print(f"  FNR (AI→Human): {rq1['fnr'] * 100:.1f}%")
        c = rq1["counts"]
        print(f"  counts: TP={c['tp']}  TN={c['tn']}  FP={c['fp']}  FN={c['fn']}")
    print()

    print("--- Per-map consistency (across the N runs per map) ---")
    cons = summary["per_map_consistency"]
    print(f"  maps analyzed          : {cons['num_maps']}")
    print(f"  majority fraction mean : {format_number(cons['majority_fraction_mean'])}")
    print(f"  majority fraction min  : {format_number(cons['majority_fraction_min'])}")
    print(f"  majority fraction max  : {format_number(cons['majority_fraction_max'])}")
    print(f"  entropy (bits) mean    : {format_number(cons['entropy_mean_bits'])}")
    print(f"  unanimous maps (20/0)  : {cons['num_unanimous_maps']}")
    print(f"  close-call maps (<60%) : {cons['num_close_call_maps']}")
    print()

    print("--- Confidence calibration (run-level accuracy by self-reported confidence) ---")
    for level, stats in summary["confidence_calibration"].items():
        acc = format_number(stats["accuracy"]) if stats["accuracy"] is not None else "NA"
        print(f"  confidence={level}: n={stats['n']:>4}  accuracy={acc}  (truth Human={stats['num_human_true']}, AI={stats['num_ai_true']})")
    print()

    print("--- Per-map detail ---")
    for m in summary["map_summaries"]:
        votes = m["votes"]
        mf = format_number(m["majority_fraction"]) if m["majority_fraction"] is not None else "NA"
        print(
            f"  {m['map_id']:<12} truth={m['true_creator']:<5}  "
            f"parsed={m['num_parsed']:>2}/{m['num_runs']:<2}  "
            f"H={votes['Human']:>2}  AI={votes['AI']:>2}  Other={votes['Other']:>2}  "
            f"majority={m['majority_label']:<8} frac={mf}  "
            f"entropy={format_number(m['entropy_bits'])}"
        )
    print()

    print("--- RQ4 (Run-level experience-metric bias) ---")
    print("    n per group is much larger than the aggregate-CSV analysis,")
    print("    which makes Mann-Whitney U tests more powerful.")
    for field_name, r in summary["run_level_rq4"].items():
        display = FIELD_TO_DISPLAY.get(field_name, field_name).upper()
        print(f"\n  [{display}]")
        print(
            "    Truth  means: "
            f"Human={format_number(r['truth_human_mean'])} (n={r['truth_n_human']})  "
            f"AI={format_number(r['truth_ai_mean'])} (n={r['truth_n_ai']})  "
            f"p={format_number(r['truth_p_value'])}"
        )
        print(
            "    Belief means: "
            f"Human={format_number(r['belief_human_mean'])} (n={r['belief_n_human']})  "
            f"AI={format_number(r['belief_ai_mean'])} (n={r['belief_n_ai']})  "
            f"p={format_number(r['belief_p_value'])}"
        )
        if r["belief_bias_detected"]:
            print("    * Belief-based bias is statistically significant (p < 0.05).")


def save_outputs(output_dir: str, run_rows: list[dict], summary: dict) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_rows:
        fieldnames = [
            "map_id", "run_index", "true_creator",
            "llm_believed_creator", "llm_believed_creator_raw",
            "parsed_ok", "confidence_level",
            "fun", "challenging", "frustrating", "surprising", "design",
            "reasoning", "error", "map_status",
        ]
        with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in run_rows:
                writer.writerow(row)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[{summary['game']}] wrote runs.csv and summary.json to {out_dir}")


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _collect(rows: list[dict], label_field: str, label_value: str, metric_field: str) -> list[float]:
    values = []
    for row in rows:
        if row.get(label_field) == label_value and row.get(metric_field) is not None:
            values.append(float(row[metric_field]))
    return values


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_metric(data: dict, metric: str):
    if metric in data:
        return data.get(metric)
    for alias in METRIC_ALIASES.get(metric, ()):
        if alias in data:
            return data.get(alias)
    return None


def _vote_entropy(counts: list[int]) -> float | None:
    total = sum(counts)
    if total == 0:
        return None
    entropy = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


if __name__ == "__main__":
    main()
