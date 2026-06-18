from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from procedural_pipeline.analysis.results import METRICS, _latest_results_csv, load_results, load_truth_map
from procedural_pipeline.analysis.truth_belief_plot import fmt, save_figure_png_and_svg
from procedural_pipeline.games import GAMES
from procedural_pipeline.paths import project_path


METRIC_TITLES = {metric: title for metric, title in METRICS}
INDICATORS = [
    ("pearson_r", "Pearson r"),
    ("spearman_rho", "Spearman rho"),
    ("kendall_tau_b", "Kendall tau-b"),
    ("icc_consistency", "ICC consistency"),
    ("icc_absolute", "ICC absolute"),
    ("top10_overlap", "Top-10 overlap"),
    ("top15_overlap", "Top-15 overlap"),
]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, sources = load_forced_pairs(args)
    summary_rows = build_consistency_rows(pairs, args.metrics)
    summary = {
        "experiment": "03_forced_creator",
        "description": (
            "Aggregate map-level consistency between forced-Human and forced-AI creator labels. "
            "Raw 20-run rows are not used; each point is one map's aggregate score."
        ),
        "sources": sources,
        "num_pairs": len(pairs),
        "metrics": args.metrics,
        "indicators": {
            "pearson_r": "Linear association between map scores under forced-Human and forced-AI labels.",
            "spearman_rho": "Rank correlation; asks whether high-scoring maps remain high-scoring.",
            "kendall_tau_b": "Pairwise rank-order agreement with tie correction.",
            "icc_consistency": "Two-way consistency ICC(C,1); ignores systematic mean shifts between labels.",
            "icc_absolute": "Two-way absolute-agreement ICC(A,1); penalizes systematic mean shifts between labels.",
            "top10_overlap": "Fraction of forced-Human top-10 maps that also appear in forced-AI top-10.",
            "top15_overlap": "Fraction of forced-Human top-15 maps that also appear in forced-AI top-15.",
            "mean_delta_human_minus_ai": "Forced-Human mean score minus forced-AI mean score.",
            "sd_delta_human_minus_ai": "STDEV.S of per-map forced-Human minus forced-AI differences.",
            "mean_abs_rank_change": "Average absolute rank movement from forced-Human to forced-AI.",
        },
        "rows": summary_rows,
    }

    write_outputs(out_dir, summary, summary_rows, pairs, args.metrics)
    if not args.no_plot:
        save_scatter_grid(pairs, summary_rows, args.metrics, out_dir / "forced_creator_consistency_scatter.png")
        save_indicator_heatmap(summary_rows, out_dir / "forced_creator_consistency_heatmap.png")
        save_rank_delta_plot(summary_rows, out_dir / "forced_creator_rank_delta_summary.png")

    print(f"wrote consistency summary -> {out_dir / 'forced_creator_consistency_summary.json'}")
    print(f"wrote consistency csv     -> {out_dir / 'forced_creator_consistency_summary.csv'}")


def parse_args() -> argparse.Namespace:
    default_root = project_path("outputs", "procedural_experiments")
    parser = argparse.ArgumentParser(
        description="Analyze whether forced-Human and forced-AI labels preserve map-level rating consistency."
    )
    parser.add_argument("--experiment-root", default=str(default_root))
    parser.add_argument("--mario-maps-file", default=None)
    parser.add_argument("--sokoban-maps-file", default=None)
    parser.add_argument("--exp3-mario-ai-results-csv", default=None)
    parser.add_argument("--exp3-mario-human-results-csv", default=None)
    parser.add_argument("--exp3-sokoban-ai-results-csv", default=None)
    parser.add_argument("--exp3-sokoban-human-results-csv", default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[metric for metric, _title in METRICS],
        choices=[metric for metric, _title in METRICS],
    )
    parser.add_argument("--output-dir", default=str(project_path("outputs", "procedural_forced_consistency")))
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def load_forced_pairs(args: argparse.Namespace) -> tuple[list[dict], dict]:
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
        ai_rows = [row for row in load_results(ai_csv, truth_map) if row.get("status") == "ok"]
        human_rows = [row for row in load_results(human_csv, truth_map) if row.get("status") == "ok"]
        ai_by_id = {row["map_id"]: row for row in ai_rows}
        human_by_id = {row["map_id"]: row for row in human_rows}
        for map_id in sorted(set(ai_by_id) & set(human_by_id)):
            pairs.append(
                {
                    "game": game_key,
                    "map_id": map_id,
                    "true_creator": truth_map.get(map_id, ""),
                    "forced_ai": ai_by_id[map_id],
                    "forced_human": human_by_id[map_id],
                }
            )
        sources[game_key] = {
            "ai_results_csv": str(ai_csv),
            "human_results_csv": str(human_csv),
            "maps_file": str(maps_file),
            "num_ai_rows": len(ai_rows),
            "num_human_rows": len(human_rows),
            "num_pairs": len(set(ai_by_id) & set(human_by_id)),
        }
    return pairs, sources


def build_consistency_rows(pairs: list[dict], metrics: list[str]) -> list[dict]:
    rows = []
    for metric in metrics:
        values = metric_pairs(pairs, metric)
        human = [item["forced_human"] for item in values]
        ai = [item["forced_ai"] for item in values]
        deltas = [h - a for h, a in zip(human, ai)]
        rank_changes = absolute_rank_changes(human, ai)
        kendall = kendall_tau_b(human, ai)
        row = {
            "metric": metric,
            "title": METRIC_TITLES[metric],
            "n": len(values),
            "forced_human_mean": safe_mean(human),
            "forced_ai_mean": safe_mean(ai),
            "mean_delta_human_minus_ai": safe_mean(deltas),
            "sd_delta_human_minus_ai": sample_std(deltas),
            "pearson_r": pearson(human, ai),
            "spearman_rho": pearson(average_ranks(human), average_ranks(ai)),
            "kendall_tau_b": kendall["tau_b"],
            "kendall_concordant": kendall["concordant"],
            "kendall_discordant": kendall["discordant"],
            "kendall_ties_human_only": kendall["ties_x_only"],
            "kendall_ties_ai_only": kendall["ties_y_only"],
            "kendall_ties_both": kendall["ties_both"],
            "pair_order_agreement": kendall["pair_order_agreement"],
            "icc_consistency": icc_two_way(human, ai, absolute=False),
            "icc_absolute": icc_two_way(human, ai, absolute=True),
            "top10_overlap": top_k_overlap(values, "forced_human", "forced_ai", 10),
            "top15_overlap": top_k_overlap(values, "forced_human", "forced_ai", 15),
            "mean_abs_rank_change": safe_mean(rank_changes),
            "median_abs_rank_change": median(rank_changes),
            "max_abs_rank_change": max(rank_changes) if rank_changes else None,
        }
        rows.append(row)
    return rows


def metric_pairs(pairs: list[dict], metric: str) -> list[dict]:
    values = []
    for pair in pairs:
        human_value = pair["forced_human"].get(metric)
        ai_value = pair["forced_ai"].get(metric)
        if human_value is None or ai_value is None:
            continue
        values.append(
            {
                "game": pair["game"],
                "map_id": pair["map_id"],
                "true_creator": pair["true_creator"],
                "forced_human": float(human_value),
                "forced_ai": float(ai_value),
                "delta_human_minus_ai": float(human_value) - float(ai_value),
            }
        )
    return values


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denom_x = math.sqrt(sum(value * value for value in dx))
    denom_y = math.sqrt(sum(value * value for value in dy))
    if denom_x == 0 or denom_y == 0:
        return None
    return sum(left * right for left, right in zip(dx, dy)) / (denom_x * denom_y)


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[indexed[position][0]] = average_rank
        index = end
    return ranks


def kendall_tau_b(x: list[float], y: list[float]) -> dict:
    concordant = discordant = ties_x_only = ties_y_only = ties_both = 0
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            dx = compare(x[i], x[j])
            dy = compare(y[i], y[j])
            if dx == 0 and dy == 0:
                ties_both += 1
            elif dx == 0:
                ties_x_only += 1
            elif dy == 0:
                ties_y_only += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x_only) * (concordant + discordant + ties_y_only))
    tau_b = (concordant - discordant) / denom if denom else None
    total_non_both = concordant + discordant + ties_x_only + ties_y_only
    agreement = concordant / total_non_both if total_non_both else None
    return {
        "tau_b": tau_b,
        "concordant": concordant,
        "discordant": discordant,
        "ties_x_only": ties_x_only,
        "ties_y_only": ties_y_only,
        "ties_both": ties_both,
        "pair_order_agreement": agreement,
    }


def compare(left: float, right: float) -> int:
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def icc_two_way(human: list[float], ai: list[float], *, absolute: bool) -> float | None:
    if len(human) != len(ai) or len(human) < 2:
        return None
    n = len(human)
    k = 2
    matrix = [[human[index], ai[index]] for index in range(n)]
    target_means = [sum(row) / k for row in matrix]
    rater_means = [sum(row[col] for row in matrix) / n for col in range(k)]
    grand_mean = sum(target_means) / n
    ss_targets = k * sum((value - grand_mean) ** 2 for value in target_means)
    ss_raters = n * sum((value - grand_mean) ** 2 for value in rater_means)
    ss_total = sum((value - grand_mean) ** 2 for row in matrix for value in row)
    ss_error = ss_total - ss_targets - ss_raters
    ms_targets = ss_targets / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    if absolute:
        denom = ms_targets + (k - 1) * ms_error + (k * (ms_raters - ms_error) / n)
    else:
        denom = ms_targets + (k - 1) * ms_error
    if denom == 0:
        return None
    return (ms_targets - ms_error) / denom


def top_k_overlap(values: list[dict], left_key: str, right_key: str, k: int) -> float | None:
    if len(values) < k:
        return None
    left_top = {item["map_id"] for item in sorted(values, key=lambda item: (-item[left_key], item["map_id"]))[:k]}
    right_top = {item["map_id"] for item in sorted(values, key=lambda item: (-item[right_key], item["map_id"]))[:k]}
    return len(left_top & right_top) / k


def absolute_rank_changes(human: list[float], ai: list[float]) -> list[float]:
    human_ranks = average_ranks([-value for value in human])
    ai_ranks = average_ranks([-value for value in ai])
    return [abs(left - right) for left, right in zip(human_ranks, ai_ranks)]


def write_outputs(out_dir: Path, summary: dict, rows: list[dict], pairs: list[dict], metrics: list[str]) -> None:
    (out_dir / "forced_creator_consistency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "forced_creator_consistency_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "metric",
            "title",
            "n",
            "forced_human_mean",
            "forced_ai_mean",
            "mean_delta_human_minus_ai",
            "sd_delta_human_minus_ai",
            "pearson_r",
            "spearman_rho",
            "kendall_tau_b",
            "pair_order_agreement",
            "icc_consistency",
            "icc_absolute",
            "top10_overlap",
            "top15_overlap",
            "mean_abs_rank_change",
            "median_abs_rank_change",
            "max_abs_rank_change",
            "kendall_concordant",
            "kendall_discordant",
            "kendall_ties_human_only",
            "kendall_ties_ai_only",
            "kendall_ties_both",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})

    with (out_dir / "forced_creator_consistency_points.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = ["game", "map_id", "true_creator", "metric", "forced_human", "forced_ai", "delta_human_minus_ai"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            for row in metric_pairs(pairs, metric):
                writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def save_scatter_grid(pairs: list[dict], rows: list[dict], metrics: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    columns = len(metrics)
    fig, axes = plt.subplots(1, columns, figsize=(3.0 * columns, 3.2), constrained_layout=True, sharex=True, sharey=True)
    if columns == 1:
        axes = [axes]
    row_by_metric = {row["metric"]: row for row in rows}
    for ax, metric in zip(axes, metrics):
        points = metric_pairs(pairs, metric)
        for creator, color, marker in [("AI", "#7EADD8", "o"), ("Human", "#E85C52", "s")]:
            xs = [point["forced_human"] for point in points if point["true_creator"] == creator]
            ys = [point["forced_ai"] for point in points if point["true_creator"] == creator]
            ax.scatter(xs, ys, s=28, alpha=0.78, color=color, edgecolor="black", linewidth=0.35, marker=marker, label=creator)
        ax.plot([1, 5], [1, 5], color="#555555", linewidth=1.0, linestyle="--")
        ax.set_xlim(1, 5)
        ax.set_ylim(1, 5)
        ax.grid(True, color="#E1E1E1", linewidth=0.8, linestyle="--")
        stat = row_by_metric[metric]
        ax.set_title(METRIC_TITLES[metric], fontsize=11, fontweight="bold")
        ax.text(
            0.04,
            0.96,
            f"rho={format_short(stat['spearman_rho'])}\ntau={format_short(stat['kendall_tau_b'])}\nICCc={format_short(stat['icc_consistency'])}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
        )
    axes[0].set_ylabel("Forced AI score")
    for ax in axes:
        ax.set_xlabel("Forced Human score")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    save_figure_png_and_svg(fig, path, dpi=180)
    plt.close(fig)


def save_indicator_heatmap(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.6, 4.4), constrained_layout=True)
    data = [[row.get(key) for key, _label in INDICATORS] for row in rows]
    image_data = [[0.0 if value is None else max(-1.0, min(1.0, float(value))) for value in row] for row in data]
    im = ax.imshow(image_data, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(INDICATORS)))
    ax.set_xticklabels([label for _key, label in INDICATORS], rotation=35, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["title"] for row in rows])
    for y, row in enumerate(data):
        for x, value in enumerate(row):
            text = "NA" if value is None else f"{value:.2f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=8, color="black")
    ax.set_title("Forced-Human vs Forced-AI Consistency Indicators", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    save_figure_png_and_svg(fig, path, dpi=180)
    plt.close(fig)


def save_rank_delta_plot(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["title"] for row in rows]
    mean_abs_rank = [row["mean_abs_rank_change"] or 0.0 for row in rows]
    mean_delta = [row["mean_delta_human_minus_ai"] or 0.0 for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), constrained_layout=True)
    axes[0].bar(labels, mean_abs_rank, color="#8FAADC", edgecolor="black", linewidth=0.8)
    axes[0].set_title("Mean Absolute Rank Change", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Rank positions")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", color="#E1E1E1", linestyle="--")

    axes[1].bar(labels, mean_delta, color="#F4A261", edgecolor="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Mean Score Shift: Human - AI", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Likert points")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", color="#E1E1E1", linestyle="--")
    save_figure_png_and_svg(fig, path, dpi=180)
    plt.close(fig)


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.stdev(values)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def format_short(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}"


if __name__ == "__main__":
    main()
