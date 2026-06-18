from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from procedural_pipeline.analysis.results import (
    METRICS,
    _latest_results_csv,
    load_results,
    load_truth_map,
    mann_whitney_u_two_sided,
)
from procedural_pipeline.analysis.truth_belief_plot import (
    GROUP_COLORS,
    GROUP_HATCHES,
    GROUPS,
    fmt,
    holm_bonferroni,
    mean,
    normalize_likert,
    save_figure_png_and_svg,
    significance_label,
    significance_text,
    std,
)
from procedural_pipeline.games import GAMES
from procedural_pipeline.paths import project_path


NO_CREATOR_COLOR = "#B8B8B8"
NO_CREATOR_HATCH = ""


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp2_summary = build_experiment2_summary(args)
    write_exp2_outputs(out_dir, exp2_summary)
    if not args.no_plot:
        exp2_plot = out_dir / "experiment2_truth_metric_plot.png"
        save_two_bar_metric_plot(
            exp2_summary["rows"],
            exp2_plot,
            score_mode=args.score_mode,
            left_key="truth_ai",
            right_key="truth_human",
            left_label="AI",
            right_label="Human",
            title_prefix="truth",
            ylabel_prefix=score_mode_ylabel(args.score_mode),
        )
        print(f"wrote experiment 2 plot -> {exp2_plot}")

    exp3_summary = build_experiment3_summary(args)
    write_exp3_outputs(out_dir, exp3_summary)
    if not args.no_plot:
        exp3_plot = out_dir / "experiment3_forced_creator_metric_plot.png"
        save_three_bar_metric_plot(
            exp3_summary["rows"],
            exp3_plot,
            score_mode=args.score_mode,
            title_prefix="forced creator",
            ylabel_prefix=score_mode_ylabel(args.score_mode),
        )
        print(f"wrote experiment 3 plot -> {exp3_plot}")

    exp3_truth_summary = build_experiment3_truth_group_summary(args)
    write_exp3_truth_outputs(out_dir, exp3_truth_summary)
    if not args.no_plot:
        forced_ai_truth_plot = out_dir / "experiment3_forced_ai_truth_metric_plot.png"
        save_two_bar_metric_plot(
            exp3_truth_summary["forced_ai_rows"],
            forced_ai_truth_plot,
            score_mode=args.score_mode,
            left_key="truth_ai",
            right_key="truth_human",
            left_label="True AI",
            right_label="True Human",
            title_prefix="forced AI, grouped by truth",
            ylabel_prefix=score_mode_ylabel(args.score_mode),
        )
        print(f"wrote experiment 3 forced AI truth plot -> {forced_ai_truth_plot}")

        forced_human_truth_plot = out_dir / "experiment3_forced_human_truth_metric_plot.png"
        save_two_bar_metric_plot(
            exp3_truth_summary["forced_human_rows"],
            forced_human_truth_plot,
            score_mode=args.score_mode,
            left_key="truth_ai",
            right_key="truth_human",
            left_label="True AI",
            right_label="True Human",
            title_prefix="forced Human, grouped by truth",
            ylabel_prefix=score_mode_ylabel(args.score_mode),
        )
        print(f"wrote experiment 3 forced Human truth plot -> {forced_human_truth_plot}")

    print(f"wrote summaries -> {out_dir}")


def parse_args() -> argparse.Namespace:
    default_root = project_path("outputs", "procedural_experiments")
    parser = argparse.ArgumentParser(
        description=(
            "Create combined Mario+Sokoban plots for experiment 2 truth groups "
            "and experiment 3 forced-creator conditions."
        )
    )
    parser.add_argument("--experiment-root", default=str(default_root))
    parser.add_argument("--mario-maps-file", default=None)
    parser.add_argument("--sokoban-maps-file", default=None)
    parser.add_argument("--exp2-mario-results-csv", default=None)
    parser.add_argument("--exp2-sokoban-results-csv", default=None)
    parser.add_argument("--exp3-mario-ai-results-csv", default=None)
    parser.add_argument("--exp3-mario-human-results-csv", default=None)
    parser.add_argument("--exp3-sokoban-ai-results-csv", default=None)
    parser.add_argument("--exp3-sokoban-human-results-csv", default=None)
    parser.add_argument(
        "--score-mode",
        choices=["normalized", "raw"],
        default="normalized",
        help="normalized maps Likert 1->-1, 3->0, 5->1; raw plots original 1-5 means.",
    )
    parser.add_argument("--output-dir", default=str(project_path("outputs", "procedural_condition_plots")))
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def build_experiment2_summary(args: argparse.Namespace) -> dict:
    rows, sources = load_experiment2_rows(args)
    valid_rows = [row for row in rows if row.get("status") == "ok"]
    summary_rows = []
    for metric, title in METRICS:
        truth_ai = describe_group(valid_rows, "true_creator", "AI", metric)
        truth_human = describe_group(valid_rows, "true_creator", "Human", metric)
        u_value, p_value = mann_whitney_u_two_sided(truth_ai["values"], truth_human["values"])
        summary_rows.append(
            {
                "metric": metric,
                "title": title,
                "truth_ai": truth_ai,
                "truth_human": truth_human,
                "human_minus_ai": subtract(truth_human["mean"], truth_ai["mean"]),
                "normalized_human_minus_ai": subtract(
                    truth_human["normalized_mean"],
                    truth_ai["normalized_mean"],
                ),
                "test": "Mann-Whitney U, two-sided",
                "p_value": p_value,
                "u_value": u_value,
            }
        )
    add_q_values(summary_rows)
    return {
        "experiment": "02_no_creator",
        "description": "Creator not asked; grouped by ground-truth creator (first 15 Human, last 15 AI per game).",
        "score_mode": args.score_mode,
        "sources": sources,
        "num_rows": len(rows),
        "num_valid_rows": len(valid_rows),
        "group_counts": {
            "truth": {group: sum(1 for row in valid_rows if row.get("true_creator") == group) for group in GROUPS},
        },
        "significance_test": {
            "test": "Mann-Whitney U, two-sided",
            "multiple_comparison_correction": "Holm-Bonferroni across five metrics.",
            "stars": {"*": "q < .05", "**": "q < .01", "***": "q < .001", "n.s.": "q >= .05"},
        },
        "rows": summary_rows,
    }


def build_experiment3_summary(args: argparse.Namespace) -> dict:
    ai_rows, human_rows, sources = load_experiment3_rows(args)
    no_creator_rows, no_creator_sources = load_experiment2_rows(args)
    no_creator_valid_rows = [row for row in no_creator_rows if row.get("status") == "ok"]
    pairs = pair_forced_rows(ai_rows, human_rows)
    summary_rows = []
    for metric, title in METRICS:
        ai_values = [pair["ai"][metric] for pair in pairs if pair["ai"].get(metric) is not None and pair["human"].get(metric) is not None]
        human_values = [pair["human"][metric] for pair in pairs if pair["ai"].get(metric) is not None and pair["human"].get(metric) is not None]
        no_creator_values = [
            float(row[metric])
            for row in no_creator_valid_rows
            if row.get(metric) is not None
        ]
        diffs = [human - ai for ai, human in zip(ai_values, human_values)]
        p_value = paired_sign_test_p(diffs)
        summary_rows.append(
            {
                "metric": metric,
                "title": title,
                "forced_ai": describe_values(ai_values),
                "no_creator": describe_values(no_creator_values),
                "forced_human": describe_values(human_values),
                "human_minus_ai": mean(diffs),
                "normalized_human_minus_ai": normalize_likert_delta(mean(diffs)),
                "paired_n": len(diffs),
                "positive_diffs": sum(1 for diff in diffs if diff > 0),
                "negative_diffs": sum(1 for diff in diffs if diff < 0),
                "zero_diffs": sum(1 for diff in diffs if diff == 0),
                "test": "paired sign test, two-sided",
                "p_value": p_value,
            }
        )
    add_q_values(summary_rows)
    return {
        "experiment": "03_forced_creator",
        "description": "Same maps are evaluated under forced AI and forced Human creator labels, pooled across Mario+Sokoban.",
        "score_mode": args.score_mode,
        "sources": sources,
        "no_creator_sources": no_creator_sources,
        "num_ai_rows": len(ai_rows),
        "num_no_creator_rows": len(no_creator_rows),
        "num_no_creator_valid_rows": len(no_creator_valid_rows),
        "num_human_rows": len(human_rows),
        "num_paired_maps": len(pairs),
        "significance_test": {
            "test": "paired sign test, two-sided",
            "comparisons": "For each metric, compare forced Human score vs forced AI score on the same map.",
            "multiple_comparison_correction": "Holm-Bonferroni across five metrics.",
            "stars": {"*": "q < .05", "**": "q < .01", "***": "q < .001", "n.s.": "q >= .05"},
        },
        "rows": summary_rows,
    }


def build_experiment3_truth_group_summary(args: argparse.Namespace) -> dict:
    ai_rows, human_rows, sources = load_experiment3_rows(args)
    ai_valid = [row for row in ai_rows if row.get("status") == "ok"]
    human_valid = [row for row in human_rows if row.get("status") == "ok"]
    forced_ai_rows = build_truth_group_rows(ai_valid, forced_label="AI")
    forced_human_rows = build_truth_group_rows(human_valid, forced_label="Human")
    add_q_values_for_prefixed_rows(forced_ai_rows, "truth")
    add_q_values_for_prefixed_rows(forced_human_rows, "truth")
    return {
        "experiment": "03_forced_creator_truth_groups",
        "description": (
            "Experiment 3 grouped by ground truth separately for the forced-AI condition "
            "and the forced-Human condition."
        ),
        "score_mode": args.score_mode,
        "sources": sources,
        "forced_ai_num_valid_rows": len(ai_valid),
        "forced_human_num_valid_rows": len(human_valid),
        "group_counts": {
            "forced_ai": {group: sum(1 for row in ai_valid if row.get("true_creator") == group) for group in GROUPS},
            "forced_human": {group: sum(1 for row in human_valid if row.get("true_creator") == group) for group in GROUPS},
        },
        "significance_test": {
            "test": "Mann-Whitney U, two-sided",
            "comparisons": "Within each forced condition, compare true-AI maps vs true-Human maps for each metric.",
            "multiple_comparison_correction": "Holm-Bonferroni across five metrics within each forced condition.",
            "stars": {"*": "q < .05", "**": "q < .01", "***": "q < .001", "n.s.": "q >= .05"},
        },
        "forced_ai_rows": forced_ai_rows,
        "forced_human_rows": forced_human_rows,
    }


def build_truth_group_rows(rows: list[dict], *, forced_label: str) -> list[dict]:
    summary_rows = []
    for metric, title in METRICS:
        truth_ai = describe_group(rows, "true_creator", "AI", metric)
        truth_human = describe_group(rows, "true_creator", "Human", metric)
        u_value, p_value = mann_whitney_u_two_sided(truth_ai["values"], truth_human["values"])
        summary_rows.append(
            {
                "metric": metric,
                "title": title,
                "forced_condition": forced_label,
                "truth_ai": truth_ai,
                "truth_human": truth_human,
                "human_minus_ai": subtract(truth_human["mean"], truth_ai["mean"]),
                "normalized_human_minus_ai": subtract(
                    truth_human["normalized_mean"],
                    truth_ai["normalized_mean"],
                ),
                "truth_mann_whitney_u": u_value,
                "truth_mann_whitney_p": p_value,
                "p_value": p_value,
                "test": "Mann-Whitney U, two-sided",
            }
        )
    return summary_rows


def load_experiment2_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    sources = {}
    for game_key in ("mario", "sokoban"):
        game = GAMES[game_key]
        results_csv = getattr(args, f"exp2_{game_key}_results_csv") or _latest_results_csv(
            str(Path(args.experiment_root) / "02_no_creator" / game_key)
        )
        maps_file = getattr(args, f"{game_key}_maps_file") or game.DEFAULTS["maps_file"]
        game_rows = load_game_rows(game_key, results_csv, maps_file)
        rows.extend(game_rows)
        sources[game_key] = {"results_csv": str(results_csv), "maps_file": str(maps_file), "num_rows": len(game_rows)}
    return rows, sources


def load_experiment3_rows(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    ai_rows: list[dict] = []
    human_rows: list[dict] = []
    sources = {}
    for game_key in ("mario", "sokoban"):
        game = GAMES[game_key]
        maps_file = getattr(args, f"{game_key}_maps_file") or game.DEFAULTS["maps_file"]
        ai_csv = getattr(args, f"exp3_{game_key}_ai_results_csv") or _latest_results_csv(
            str(Path(args.experiment_root) / "03_forced_creator" / game_key / "ai")
        )
        human_csv = getattr(args, f"exp3_{game_key}_human_results_csv") or _latest_results_csv(
            str(Path(args.experiment_root) / "03_forced_creator" / game_key / "human")
        )
        game_ai_rows = load_game_rows(game_key, ai_csv, maps_file)
        game_human_rows = load_game_rows(game_key, human_csv, maps_file)
        for row in game_ai_rows:
            row["forced_condition"] = "AI"
        for row in game_human_rows:
            row["forced_condition"] = "Human"
        ai_rows.extend(game_ai_rows)
        human_rows.extend(game_human_rows)
        sources[game_key] = {
            "ai_results_csv": str(ai_csv),
            "human_results_csv": str(human_csv),
            "maps_file": str(maps_file),
            "num_ai_rows": len(game_ai_rows),
            "num_human_rows": len(game_human_rows),
        }
    return ai_rows, human_rows, sources


def load_game_rows(game_key: str, results_csv: str, maps_file: str) -> list[dict]:
    truth_map = load_truth_map(maps_file)
    rows = load_results(results_csv, truth_map)
    for row in rows:
        row["game"] = game_key
    return rows


def pair_forced_rows(ai_rows: list[dict], human_rows: list[dict]) -> list[dict]:
    ai_by_key = {
        (row.get("game"), row.get("map_id")): row
        for row in ai_rows
        if row.get("status") == "ok"
    }
    human_by_key = {
        (row.get("game"), row.get("map_id")): row
        for row in human_rows
        if row.get("status") == "ok"
    }
    pairs = []
    for key in sorted(set(ai_by_key) & set(human_by_key)):
        pairs.append({"game": key[0], "map_id": key[1], "ai": ai_by_key[key], "human": human_by_key[key]})
    return pairs


def describe_group(rows: list[dict], group_field: str, group_label: str, metric: str) -> dict:
    values = [
        float(row[metric])
        for row in rows
        if row.get(group_field) == group_label and row.get(metric) is not None
    ]
    return describe_values(values)


def describe_values(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": mean(values),
        "normalized_mean": normalize_likert(mean(values)),
        "std": std(values),
        "values": values,
    }


def add_q_values(rows: list[dict]) -> None:
    p_values = [row["p_value"] for row in rows if row.get("p_value") is not None]
    q_values = holm_bonferroni([float(p) for p in p_values])
    q_iter = iter(q_values)
    for row in rows:
        row["q_holm"] = next(q_iter) if row.get("p_value") is not None else None
        row["significance"] = significance_label(row["q_holm"])


def add_q_values_for_prefixed_rows(rows: list[dict], prefix: str) -> None:
    p_key = f"{prefix}_mann_whitney_p"
    q_key = f"{prefix}_mann_whitney_q_holm"
    p_values = [row[p_key] for row in rows if row.get(p_key) is not None]
    q_values = holm_bonferroni([float(p) for p in p_values])
    q_iter = iter(q_values)
    for row in rows:
        row[q_key] = next(q_iter) if row.get(p_key) is not None else None
        row[f"{prefix}_significance"] = significance_label(row[q_key])
        row["q_holm"] = row[q_key]
        row["significance"] = row[f"{prefix}_significance"]


def write_exp2_outputs(out_dir: Path, summary: dict) -> None:
    (out_dir / "experiment2_truth_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_rows_csv(out_dir / "experiment2_truth_summary.csv", summary["rows"], "truth_ai", "truth_human")


def write_exp3_outputs(out_dir: Path, summary: dict) -> None:
    (out_dir / "experiment3_forced_creator_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_rows_csv(
        out_dir / "experiment3_forced_creator_summary.csv",
        summary["rows"],
        "forced_ai",
        "forced_human",
        middle_key="no_creator",
    )


def write_exp3_truth_outputs(out_dir: Path, summary: dict) -> None:
    (out_dir / "experiment3_forced_truth_table.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_forced_truth_rows_csv(
        out_dir / "experiment3_forced_truth_table.csv",
        summary["forced_ai_rows"],
        summary["forced_human_rows"],
    )


def write_forced_truth_rows_csv(path: Path, forced_ai_rows: list[dict], forced_human_rows: list[dict]) -> None:
    fields = [
        "forced_condition",
        "metric",
        "truth_ai_n",
        "truth_ai_mean",
        "truth_ai_normalized_mean",
        "truth_ai_std",
        "truth_human_n",
        "truth_human_mean",
        "truth_human_normalized_mean",
        "truth_human_std",
        "human_minus_ai",
        "normalized_human_minus_ai",
        "truth_mann_whitney_u",
        "truth_mann_whitney_p",
        "truth_mann_whitney_q_holm",
        "truth_significance",
        "test",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in [*forced_ai_rows, *forced_human_rows]:
            writer.writerow(flatten_forced_truth_row(row))


def flatten_forced_truth_row(row: dict) -> dict:
    flat = {
        "forced_condition": row["forced_condition"],
        "metric": row["metric"],
        "human_minus_ai": fmt(row.get("human_minus_ai")),
        "normalized_human_minus_ai": fmt(row.get("normalized_human_minus_ai")),
        "truth_mann_whitney_u": fmt(row.get("truth_mann_whitney_u")),
        "truth_mann_whitney_p": fmt(row.get("truth_mann_whitney_p")),
        "truth_mann_whitney_q_holm": fmt(row.get("truth_mann_whitney_q_holm")),
        "truth_significance": row.get("truth_significance", ""),
        "test": row.get("test", ""),
    }
    for key in ("truth_ai", "truth_human"):
        stats = row[key]
        flat[f"{key}_n"] = stats["n"]
        flat[f"{key}_mean"] = fmt(stats["mean"])
        flat[f"{key}_normalized_mean"] = fmt(stats["normalized_mean"])
        flat[f"{key}_std"] = fmt(stats["std"])
    return flat


def write_rows_csv(path: Path, rows: list[dict], left_key: str, right_key: str, middle_key: str | None = None) -> None:
    fields = [
        "metric",
        f"{left_key}_n",
        f"{left_key}_mean",
        f"{left_key}_normalized_mean",
        f"{left_key}_std",
    ]
    if middle_key:
        fields.extend(
            [
                f"{middle_key}_n",
                f"{middle_key}_mean",
                f"{middle_key}_normalized_mean",
                f"{middle_key}_std",
            ]
        )
    fields.extend([
        f"{right_key}_n",
        f"{right_key}_mean",
        f"{right_key}_normalized_mean",
        f"{right_key}_std",
        "human_minus_ai",
        "normalized_human_minus_ai",
        "paired_n",
        "positive_diffs",
        "negative_diffs",
        "zero_diffs",
        "test",
        "p_value",
        "q_holm",
        "significance",
    ])
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_row(row, left_key, right_key, middle_key=middle_key))


def flatten_row(row: dict, left_key: str, right_key: str, middle_key: str | None = None) -> dict:
    flat = {
        "metric": row["metric"],
        "human_minus_ai": fmt(row.get("human_minus_ai")),
        "normalized_human_minus_ai": fmt(row.get("normalized_human_minus_ai")),
        "paired_n": row.get("paired_n", ""),
        "positive_diffs": row.get("positive_diffs", ""),
        "negative_diffs": row.get("negative_diffs", ""),
        "zero_diffs": row.get("zero_diffs", ""),
        "test": row.get("test", ""),
        "p_value": fmt(row.get("p_value")),
        "q_holm": fmt(row.get("q_holm")),
        "significance": row.get("significance", ""),
    }
    keys = [left_key]
    if middle_key:
        keys.append(middle_key)
    keys.append(right_key)
    for key in keys:
        stats = row[key]
        flat[f"{key}_n"] = stats["n"]
        flat[f"{key}_mean"] = fmt(stats["mean"])
        flat[f"{key}_normalized_mean"] = fmt(stats["normalized_mean"])
        flat[f"{key}_std"] = fmt(stats["std"])
    return flat


def save_two_bar_metric_plot(
    rows: list[dict],
    path: Path,
    *,
    score_mode: str,
    left_key: str,
    right_key: str,
    left_label: str,
    right_label: str,
    title_prefix: str,
    ylabel_prefix: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    y_limits = compute_y_limits(rows, score_mode, left_key, right_key)
    fig, ax = plt.subplots(figsize=(10.4, 5.4), constrained_layout=True)
    bar_width = 0.28
    group_gap = 0.95
    x = 0.0

    for row in rows:
        left_value = plot_value(row[left_key]["mean"], score_mode)
        right_value = plot_value(row[right_key]["mean"], score_mode)
        ax.bar(
            x - bar_width / 1.7,
            left_value,
            width=bar_width,
            color=GROUP_COLORS["AI"],
            edgecolor="black",
            hatch=GROUP_HATCHES["AI"],
            linewidth=1.0,
            alpha=0.95,
        )
        ax.bar(
            x + bar_width / 1.7,
            right_value,
            width=bar_width,
            color=GROUP_COLORS["Human"],
            edgecolor="black",
            hatch=GROUP_HATCHES["Human"],
            linewidth=1.0,
            alpha=0.95,
        )

        top_y = y_limits["top_label"]
        ax.plot([x - 0.43, x + 0.43], [top_y, top_y], color="black", linewidth=1.4, clip_on=False)
        ax.text(x, top_y + y_limits["title_lift"], row["title"], ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(
            x,
            top_y - y_limits["text_drop"],
            significance_text(row.get("p_value"), row.get("q_holm")),
            ha="center",
            va="top",
            fontsize=8,
        )
        x += group_gap

    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", color="#D9D9D9", linewidth=1.0, alpha=0.8)
    ax.set_xlim(-0.65, x - group_gap + 0.65)
    ax.set_ylim(y_limits["y_min"], y_limits["y_max"])
    ax.set_xticks([])
    ax.set_ylabel(ylabel_prefix)
    ax.set_title(title_prefix, fontsize=12, fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor=GROUP_COLORS["AI"], edgecolor="black", hatch=GROUP_HATCHES["AI"], label=left_label),
            Patch(facecolor=GROUP_COLORS["Human"], edgecolor="black", hatch=GROUP_HATCHES["Human"], label=right_label),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
    )
    save_figure_png_and_svg(fig, path, dpi=180)
    plt.close(fig)


def save_three_bar_metric_plot(
    rows: list[dict],
    path: Path,
    *,
    score_mode: str,
    title_prefix: str,
    ylabel_prefix: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    y_limits = compute_y_limits(rows, score_mode, "forced_ai", "no_creator", "forced_human")
    fig, ax = plt.subplots(figsize=(10.8, 5.4), constrained_layout=True)
    bar_width = 0.22
    group_gap = 0.95
    x = 0.0

    for row in rows:
        values = [
            ("forced_ai", "AI", GROUP_COLORS["AI"], GROUP_HATCHES["AI"], -bar_width * 1.15),
            ("no_creator", "No creator", NO_CREATOR_COLOR, NO_CREATOR_HATCH, 0.0),
            ("forced_human", "Human", GROUP_COLORS["Human"], GROUP_HATCHES["Human"], bar_width * 1.15),
        ]
        for key, _label, color, hatch, offset in values:
            ax.bar(
                x + offset,
                plot_value(row[key]["mean"], score_mode),
                width=bar_width,
                color=color,
                edgecolor="black",
                hatch=hatch,
                linewidth=1.0,
                alpha=0.95,
            )

        top_y = y_limits["top_label"]
        ax.plot([x - 0.42, x + 0.42], [top_y, top_y], color="black", linewidth=1.4, clip_on=False)
        ax.text(x, top_y + y_limits["title_lift"], row["title"], ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(
            x,
            top_y - y_limits["text_drop"],
            significance_text(row.get("p_value"), row.get("q_holm")),
            ha="center",
            va="top",
            fontsize=8,
        )
        x += group_gap

    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", color="#D9D9D9", linewidth=1.0, alpha=0.8)
    ax.set_xlim(-0.65, x - group_gap + 0.65)
    ax.set_ylim(y_limits["y_min"], y_limits["y_max"])
    ax.set_xticks([])
    ax.set_ylabel(ylabel_prefix)
    ax.set_title(f"{title_prefix} + no creator baseline", fontsize=12, fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor=GROUP_COLORS["AI"], edgecolor="black", hatch=GROUP_HATCHES["AI"], label="Forced AI"),
            Patch(facecolor=NO_CREATOR_COLOR, edgecolor="black", hatch=NO_CREATOR_HATCH, label="No creator"),
            Patch(facecolor=GROUP_COLORS["Human"], edgecolor="black", hatch=GROUP_HATCHES["Human"], label="Forced Human"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )
    save_figure_png_and_svg(fig, path, dpi=180)
    plt.close(fig)


def compute_y_limits(rows: list[dict], score_mode: str, *keys: str) -> dict[str, float]:
    values = []
    for row in rows:
        for key in keys:
            value = row[key]["mean"]
            if value is not None:
                values.append(plot_value(value, score_mode))

    if score_mode == "raw":
        max_value = max(values) if values else 5.0
        y_max = min(5.35, max(2.0, max_value + 0.65))
        return {"y_min": 0.0, "y_max": y_max, "top_label": y_max - 0.2, "text_drop": 0.18, "title_lift": 0.05}

    min_value = min(values) if values else -1.0
    max_value = max(values) if values else 1.0
    y_min = min(-1.05, min_value - 0.18)
    y_max = max(1.15, max_value + 0.24)
    return {"y_min": y_min, "y_max": y_max, "top_label": y_max - 0.12, "text_drop": 0.10, "title_lift": 0.03}


def plot_value(value: float | None, score_mode: str) -> float:
    if value is None:
        return 0.0
    if score_mode == "normalized":
        normalized = normalize_likert(value)
        return 0.0 if normalized is None else normalized
    return float(value)


def score_mode_ylabel(score_mode: str) -> str:
    if score_mode == "normalized":
        return "Normalized Likert score (1=-1, 3=0, 5=1)"
    return "Average Likert score"


def paired_sign_test_p(diffs: list[float]) -> float | None:
    positives = sum(1 for diff in diffs if diff > 0)
    negatives = sum(1 for diff in diffs if diff < 0)
    n = positives + negatives
    if n == 0:
        return None
    k = min(positives, negatives)
    probability = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability)


def normalize_likert_delta(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 2.0


def subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


if __name__ == "__main__":
    main()
