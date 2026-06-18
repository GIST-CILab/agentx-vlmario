from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from scipy.stats import mannwhitneyu, wilcoxon

from procedural_pipeline.analysis.results import METRICS, _latest_results_csv, load_results, load_truth_map
from procedural_pipeline.analysis.truth_belief_plot import holm_bonferroni
from procedural_pipeline.games import GAMES
from procedural_pipeline.paths import project_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp1_rows, exp1_sources = load_experiment1_rows(args)
    exp3_pairs, exp3_sources = load_experiment3_pairs(args)

    exp1_results = analyze_experiment1(exp1_rows)
    exp3_results = analyze_experiment3(exp3_pairs)
    exp3_truth_results = analyze_experiment3_truth_groups(exp3_pairs)
    add_holm_q(exp1_results)
    add_holm_q(exp3_results)
    add_holm_q(exp3_truth_results)

    summary = {
        "description": (
            "Nonparametric tests after normality violations: Experiment 1 uses "
            "two-sided Mann-Whitney U tests for independent AI/Human groups; "
            "Experiment 3 uses two-sided Wilcoxon signed-rank tests for paired "
            "Forced Human vs Forced AI scores on the same maps."
        ),
        "experiment1_sources": exp1_sources,
        "experiment3_sources": exp3_sources,
        "experiment1": exp1_results,
        "experiment3": exp3_results,
        "experiment3_truth_groups": exp3_truth_results,
    }
    write_outputs(out_dir, summary, exp1_results, exp3_results, exp3_truth_results)

    print(f"wrote -> {out_dir / 'nonparametric_tests_summary.json'}")
    print(f"wrote -> {out_dir / 'experiment1_mann_whitney_u.csv'}")
    print(f"wrote -> {out_dir / 'experiment3_wilcoxon_signed_rank.csv'}")
    print("\nExperiment 1 Mann-Whitney U:")
    for row in exp1_results:
        print(
            f"  {row['grouping']:6s} {row['metric']:12s} "
            f"AI={row['ai_mean']:.4f} Human={row['human_mean']:.4f} "
            f"diff={row['human_minus_ai']:.4f} p={row['p_value']:.6g} q={row['q_holm']:.6g}"
        )
    print("\nExperiment 3 Wilcoxon signed-rank:")
    for row in exp3_results:
        print(
            f"  {row['metric']:12s} "
            f"Human={row['forced_human_mean']:.4f} AI={row['forced_ai_mean']:.4f} "
            f"diff={row['mean_diff_human_minus_ai']:.4f} p={row['p_value']:.6g} q={row['q_holm']:.6g}"
        )
    print("\nExperiment 3 truth-group Mann-Whitney U:")
    for row in exp3_truth_results:
        print(
            f"  {row['forced_condition']:12s} {row['metric']:12s} "
            f"truthAI={row['truth_ai_mean']:.4f} truthHuman={row['truth_human_mean']:.4f} "
            f"diff={row['truth_human_minus_ai']:.4f} p={row['p_value']:.6g} q={row['q_holm']:.6g}"
        )


def parse_args() -> argparse.Namespace:
    default_root = project_path("outputs", "procedural_experiments")
    parser = argparse.ArgumentParser(
        description="Run nonparametric tests for Experiment 1 and Experiment 3."
    )
    parser.add_argument("--experiment-root", default=str(default_root))
    parser.add_argument("--output-dir", default=str(project_path("outputs", "procedural_nonparametric_tests")))
    parser.add_argument("--mario-maps-file", default=None)
    parser.add_argument("--sokoban-maps-file", default=None)
    parser.add_argument("--exp1-mario-results-csv", default=None)
    parser.add_argument("--exp1-sokoban-results-csv", default=None)
    parser.add_argument("--exp3-mario-ai-results-csv", default=None)
    parser.add_argument("--exp3-mario-human-results-csv", default=None)
    parser.add_argument("--exp3-sokoban-ai-results-csv", default=None)
    parser.add_argument("--exp3-sokoban-human-results-csv", default=None)
    return parser.parse_args()


def load_experiment1_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows = []
    sources = {}
    for game_key in ("mario", "sokoban"):
        game = GAMES[game_key]
        maps_file = getattr(args, f"{game_key}_maps_file") or game.DEFAULTS["maps_file"]
        truth_map = load_truth_map(maps_file)
        results_csv = getattr(args, f"exp1_{game_key}_results_csv") or _latest_results_csv(
            str(Path(args.experiment_root) / "01_creator_judgment" / game_key)
        )
        game_rows = load_results(results_csv, truth_map)
        for row in game_rows:
            row["game"] = game_key
        rows.extend(row for row in game_rows if row.get("status") == "ok")
        sources[game_key] = {"results_csv": str(results_csv), "maps_file": str(maps_file)}
    return rows, sources


def load_experiment3_pairs(args: argparse.Namespace) -> tuple[list[dict], dict]:
    pairs = []
    sources = {}
    for game_key in ("mario", "sokoban"):
        game = GAMES[game_key]
        maps_file = getattr(args, f"{game_key}_maps_file") or game.DEFAULTS["maps_file"]
        truth_map = load_truth_map(maps_file)
        ai_csv = getattr(args, f"exp3_{game_key}_ai_results_csv") or _latest_results_csv(
            str(Path(args.experiment_root) / "03_forced_creator" / game_key / "ai")
        )
        human_csv = getattr(args, f"exp3_{game_key}_human_results_csv") or _latest_results_csv(
            str(Path(args.experiment_root) / "03_forced_creator" / game_key / "human")
        )
        ai_rows = {row["map_id"]: row for row in load_results(ai_csv, truth_map) if row.get("status") == "ok"}
        human_rows = {row["map_id"]: row for row in load_results(human_csv, truth_map) if row.get("status") == "ok"}
        for map_id in sorted(set(ai_rows) & set(human_rows)):
            pairs.append(
                {
                    "game": game_key,
                    "map_id": map_id,
                    "true_creator": truth_map.get(map_id, ""),
                    "forced_ai": ai_rows[map_id],
                    "forced_human": human_rows[map_id],
                }
            )
        sources[game_key] = {
            "ai_results_csv": str(ai_csv),
            "human_results_csv": str(human_csv),
            "maps_file": str(maps_file),
        }
    return pairs, sources


def analyze_experiment1(rows: list[dict]) -> list[dict]:
    results = []
    for grouping, field in (("truth", "true_creator"), ("belief", "llm_believed_creator")):
        for metric, _title in METRICS:
            ai = collect(rows, metric, field, "AI")
            human = collect(rows, metric, field, "Human")
            stat, p_value = mann_whitney(ai, human)
            results.append(
                {
                    "experiment": "01_creator_judgment",
                    "test": "Mann-Whitney U, two-sided",
                    "grouping": grouping,
                    "metric": metric,
                    "ai_n": len(ai),
                    "ai_mean": safe_mean(ai),
                    "ai_sd": safe_sd(ai),
                    "human_n": len(human),
                    "human_mean": safe_mean(human),
                    "human_sd": safe_sd(human),
                    "human_minus_ai": subtract(safe_mean(human), safe_mean(ai)),
                    "statistic": stat,
                    "p_value": p_value,
                }
            )
    return results


def analyze_experiment3(pairs: list[dict]) -> list[dict]:
    results = []
    for metric, _title in METRICS:
        human = []
        ai = []
        for pair in pairs:
            human_value = pair["forced_human"].get(metric)
            ai_value = pair["forced_ai"].get(metric)
            if human_value is None or ai_value is None:
                continue
            human.append(float(human_value))
            ai.append(float(ai_value))
        diffs = [left - right for left, right in zip(human, ai)]
        stat, p_value = wilcoxon_signed_rank(human, ai)
        results.append(
            {
                "experiment": "03_forced_creator",
                "test": "Wilcoxon signed-rank, paired, two-sided",
                "comparison": "forced_human_vs_forced_ai",
                "metric": metric,
                "n_pairs": len(diffs),
                "forced_human_mean": safe_mean(human),
                "forced_human_sd": safe_sd(human),
                "forced_ai_mean": safe_mean(ai),
                "forced_ai_sd": safe_sd(ai),
                "mean_diff_human_minus_ai": safe_mean(diffs),
                "median_diff_human_minus_ai": statistics.median(diffs) if diffs else None,
                "sd_diff_human_minus_ai": safe_sd(diffs),
                "statistic": stat,
                "p_value": p_value,
            }
        )
    return results


def analyze_experiment3_truth_groups(pairs: list[dict]) -> list[dict]:
    results = []
    conditions = (("forced_ai", "AI"), ("forced_human", "Human"))
    for condition_key, condition_label in conditions:
        for metric, _title in METRICS:
            ai_values = []
            human_values = []
            for pair in pairs:
                row = pair[condition_key]
                value = row.get(metric)
                if value is None:
                    continue
                if pair.get("true_creator") == "AI":
                    ai_values.append(float(value))
                elif pair.get("true_creator") == "Human":
                    human_values.append(float(value))
            stat, p_value = mann_whitney(ai_values, human_values)
            results.append(
                {
                    "experiment": "03_forced_creator",
                    "test": "Mann-Whitney U, two-sided",
                    "forced_condition": condition_label,
                    "grouping": "truth",
                    "metric": metric,
                    "truth_ai_n": len(ai_values),
                    "truth_ai_mean": safe_mean(ai_values),
                    "truth_ai_sd": safe_sd(ai_values),
                    "truth_human_n": len(human_values),
                    "truth_human_mean": safe_mean(human_values),
                    "truth_human_sd": safe_sd(human_values),
                    "truth_human_minus_ai": subtract(safe_mean(human_values), safe_mean(ai_values)),
                    "statistic": stat,
                    "p_value": p_value,
                }
            )
    return results


def collect(rows: list[dict], metric: str, field: str, group: str) -> list[float]:
    values = []
    for row in rows:
        if row.get(field) == group and row.get(metric) is not None:
            values.append(float(row[metric]))
    return values


def mann_whitney(ai: list[float], human: list[float]) -> tuple[float | None, float | None]:
    if not ai or not human:
        return None, None
    result = mannwhitneyu(ai, human, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def wilcoxon_signed_rank(human: list[float], ai: list[float]) -> tuple[float | None, float | None]:
    if not human or not ai:
        return None, None
    diffs = [left - right for left, right in zip(human, ai)]
    if all(diff == 0 for diff in diffs):
        return 0.0, 1.0
    result = wilcoxon(human, ai, alternative="two-sided", zero_method="wilcox", method="auto")
    return float(result.statistic), float(result.pvalue)


def add_holm_q(rows: list[dict]) -> None:
    with_p = [row for row in rows if row.get("p_value") is not None]
    q_values = holm_bonferroni([row["p_value"] for row in with_p])
    for row, q_value in zip(with_p, q_values):
        row["q_holm"] = q_value
    for row in rows:
        row.setdefault("q_holm", None)


def write_outputs(
    out_dir: Path,
    summary: dict,
    exp1_rows: list[dict],
    exp3_rows: list[dict],
    exp3_truth_rows: list[dict],
) -> None:
    (out_dir / "nonparametric_tests_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(out_dir / "experiment1_mann_whitney_u.csv", exp1_rows)
    write_csv(out_dir / "experiment3_wilcoxon_signed_rank.csv", exp3_rows)
    write_csv(out_dir / "experiment3_truth_mann_whitney_u.csv", exp3_truth_rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def safe_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.stdev(values)


def subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
