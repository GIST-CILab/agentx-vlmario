from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from procedural_pipeline.analysis.results import (
    METRICS,
    _latest_results_csv,
    load_results,
    load_truth_map,
    mann_whitney_u_two_sided,
)
from procedural_pipeline.games import GAMES
from procedural_pipeline.paths import project_path


GROUPS = ("AI", "Human")
GROUP_COLORS = {
    "AI": "#7EADD8",
    "Human": "#E85C52",
}
GROUP_HATCHES = {
    "AI": "\\\\",
    "Human": "..",
}


def main() -> None:
    args = parse_args()
    rows, sources = load_combined_experiment1_rows(args)
    valid_rows = [row for row in rows if row.get("status") == "ok"]
    belief_rows = [row for row in valid_rows if row.get(
        "llm_believed_creator") in GROUPS]

    summary_rows = build_summary_rows(valid_rows, belief_rows)
    summary = {
        "experiment": "01_creator_judgment",
        "score_mode": args.score_mode,
        "sources": sources,
        "num_rows": len(rows),
        "num_valid_rows": len(valid_rows),
        "num_belief_valid_rows": len(belief_rows),
        "significance_test": {
            "test": "Mann-Whitney U, two-sided",
            "comparisons": "For each metric, compare AI vs Human within truth groups and within belief groups.",
            "multiple_comparison_correction": "Holm-Bonferroni across all truth/belief metric comparisons; graph stars use corrected q-values.",
            "stars": {"*": "q < .05", "**": "q < .01", "***": "q < .001", "n.s.": "q >= .05"},
        },
        "group_counts": {
            "truth": count_groups(valid_rows, "true_creator"),
            "belief": count_groups(belief_rows, "llm_believed_creator"),
        },
        "rows": summary_rows,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_outputs(out_dir, summary, summary_rows)

    if not args.no_plot:
        plot_path = out_dir / "truth_belief_metric_plot.png"
        save_truth_belief_plot(summary_rows, plot_path,
                               score_mode=args.score_mode)
        print(f"wrote plot -> {plot_path}")

    print(f"wrote summary -> {out_dir / 'truth_belief_metric_summary.json'}")
    print(f"wrote csv     -> {out_dir / 'truth_belief_metric_summary.csv'}")


def parse_args() -> argparse.Namespace:
    default_root = project_path(
        "outputs", "procedural_experiments", "01_creator_judgment")
    parser = argparse.ArgumentParser(
        description=(
            "Pool experiment 1 Mario+Sokoban rows and plot five metric means grouped "
            "by ground truth creator and by LLM creator belief."
        )
    )
    parser.add_argument("--experiment-root", default=str(default_root))
    parser.add_argument("--mario-results-csv", default=None)
    parser.add_argument("--sokoban-results-csv", default=None)
    parser.add_argument("--mario-maps-file", default=None)
    parser.add_argument("--sokoban-maps-file", default=None)
    parser.add_argument(
        "--score-mode",
        choices=["normalized", "centered", "raw"],
        default="normalized",
        help=(
            "normalized maps Likert 1->-1, 3->0, 5->1; "
            "centered subtracts each metric's grand mean; "
            "raw plots the original 1-5 Likert means."
        ),
    )
    parser.add_argument(
        "--output-dir", default=str(project_path("outputs", "procedural_truth_belief_plot")))
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def load_combined_experiment1_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    sources = {}
    for game_key in ("mario", "sokoban"):
        game = GAMES[game_key]
        results_csv = getattr(args, f"{game_key}_results_csv") or _latest_results_csv(
            str(Path(args.experiment_root) / game_key)
        )
        maps_file = getattr(
            args, f"{game_key}_maps_file") or game.DEFAULTS["maps_file"]
        truth_map = load_truth_map(maps_file)
        game_rows = load_results(results_csv, truth_map)
        for row in game_rows:
            row["game"] = game_key
        rows.extend(game_rows)
        sources[game_key] = {
            "results_csv": str(results_csv),
            "maps_file": str(maps_file),
            "num_rows": len(game_rows),
        }
    return rows, sources


def build_summary_rows(valid_rows: list[dict], belief_rows: list[dict]) -> list[dict]:
    output = []
    for metric, title in METRICS:
        grand_values = [float(row[metric])
                        for row in valid_rows if row.get(metric) is not None]
        grand_mean = mean(grand_values)
        truth_ai = describe_group(valid_rows, "true_creator", "AI", metric)
        truth_human = describe_group(
            valid_rows, "true_creator", "Human", metric)
        belief_ai = describe_group(
            belief_rows, "llm_believed_creator", "AI", metric)
        belief_human = describe_group(
            belief_rows, "llm_believed_creator", "Human", metric)
        truth_u, truth_p = mann_whitney_u_two_sided(
            truth_ai["values"], truth_human["values"])
        belief_u, belief_p = mann_whitney_u_two_sided(
            belief_ai["values"], belief_human["values"])
        output.append(
            {
                "metric": metric,
                "title": title,
                "grand_mean": grand_mean,
                "grand_normalized_mean": normalize_likert(grand_mean),
                "truth_ai": truth_ai,
                "truth_human": truth_human,
                "belief_ai": belief_ai,
                "belief_human": belief_human,
                "truth_human_minus_ai": subtract(truth_human["mean"], truth_ai["mean"]),
                "belief_human_minus_ai": subtract(belief_human["mean"], belief_ai["mean"]),
                "human_belief_minus_truth": subtract(belief_human["mean"], truth_human["mean"]),
                "ai_belief_minus_truth": subtract(belief_ai["mean"], truth_ai["mean"]),
                "truth_normalized_human_minus_ai": subtract(truth_human["normalized_mean"], truth_ai["normalized_mean"]),
                "belief_normalized_human_minus_ai": subtract(belief_human["normalized_mean"], belief_ai["normalized_mean"]),
                "human_normalized_belief_minus_truth": subtract(belief_human["normalized_mean"], truth_human["normalized_mean"]),
                "ai_normalized_belief_minus_truth": subtract(belief_ai["normalized_mean"], truth_ai["normalized_mean"]),
                "truth_mann_whitney_u": truth_u,
                "truth_mann_whitney_p": truth_p,
                "belief_mann_whitney_u": belief_u,
                "belief_mann_whitney_p": belief_p,
            }
        )
    add_holm_adjusted_p_values(output)
    return output


def describe_group(rows: list[dict], group_field: str, group_label: str, metric: str) -> dict:
    values = [
        float(row[metric])
        for row in rows
        if row.get(group_field) == group_label and row.get(metric) is not None
    ]
    return {
        "n": len(values),
        "mean": mean(values),
        "normalized_mean": normalize_likert(mean(values)),
        "std": std(values),
        "values": values,
    }


def count_groups(rows: list[dict], field: str) -> dict[str, int]:
    return {group: sum(1 for row in rows if row.get(field) == group) for group in GROUPS}


def write_summary_outputs(out_dir: Path, summary: dict, rows: list[dict]) -> None:
    (out_dir / "truth_belief_metric_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fields = [
        "metric",
        "grand_mean",
        "grand_normalized_mean",
        "truth_ai_n",
        "truth_ai_mean",
        "truth_ai_std",
        "truth_ai_normalized_mean",
        "truth_human_n",
        "truth_human_mean",
        "truth_human_std",
        "truth_human_normalized_mean",
        "belief_ai_n",
        "belief_ai_mean",
        "belief_ai_std",
        "belief_ai_normalized_mean",
        "belief_human_n",
        "belief_human_mean",
        "belief_human_std",
        "belief_human_normalized_mean",
        "truth_human_minus_ai",
        "belief_human_minus_ai",
        "human_belief_minus_truth",
        "ai_belief_minus_truth",
        "truth_normalized_human_minus_ai",
        "belief_normalized_human_minus_ai",
        "human_normalized_belief_minus_truth",
        "ai_normalized_belief_minus_truth",
        "truth_mann_whitney_u",
        "truth_mann_whitney_p",
        "truth_mann_whitney_q_holm",
        "truth_significance",
        "belief_mann_whitney_u",
        "belief_mann_whitney_p",
        "belief_mann_whitney_q_holm",
        "belief_significance",
    ]
    with (out_dir / "truth_belief_metric_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_summary_row(row))


def flatten_summary_row(row: dict) -> dict:
    flat = {
        "metric": row["metric"],
        "grand_mean": fmt(row["grand_mean"]),
        "grand_normalized_mean": fmt(row["grand_normalized_mean"]),
        "truth_human_minus_ai": fmt(row["truth_human_minus_ai"]),
        "belief_human_minus_ai": fmt(row["belief_human_minus_ai"]),
        "human_belief_minus_truth": fmt(row["human_belief_minus_truth"]),
        "ai_belief_minus_truth": fmt(row["ai_belief_minus_truth"]),
        "truth_normalized_human_minus_ai": fmt(row["truth_normalized_human_minus_ai"]),
        "belief_normalized_human_minus_ai": fmt(row["belief_normalized_human_minus_ai"]),
        "human_normalized_belief_minus_truth": fmt(row["human_normalized_belief_minus_truth"]),
        "ai_normalized_belief_minus_truth": fmt(row["ai_normalized_belief_minus_truth"]),
        "truth_mann_whitney_u": fmt(row["truth_mann_whitney_u"]),
        "truth_mann_whitney_p": fmt(row["truth_mann_whitney_p"]),
        "truth_mann_whitney_q_holm": fmt(row["truth_mann_whitney_q_holm"]),
        "truth_significance": significance_label(row["truth_mann_whitney_q_holm"]),
        "belief_mann_whitney_u": fmt(row["belief_mann_whitney_u"]),
        "belief_mann_whitney_p": fmt(row["belief_mann_whitney_p"]),
        "belief_mann_whitney_q_holm": fmt(row["belief_mann_whitney_q_holm"]),
        "belief_significance": significance_label(row["belief_mann_whitney_q_holm"]),
    }
    for group_key in ("truth_ai", "truth_human", "belief_ai", "belief_human"):
        stats = row[group_key]
        flat[f"{group_key}_n"] = stats["n"]
        flat[f"{group_key}_mean"] = fmt(stats["mean"])
        flat[f"{group_key}_std"] = fmt(stats["std"])
        flat[f"{group_key}_normalized_mean"] = fmt(stats["normalized_mean"])
    return flat


def save_truth_belief_plot(rows: list[dict], path: Path, *, score_mode: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(11.6, 3.4), constrained_layout=True)
    group_width = 1.0
    metric_gap = 0.45
    bar_width = 0.32
    x = 0.0
    metric_centers = []
    group_centers = []
    y_limits = compute_y_limits(rows, score_mode)
    top_y = y_limits["top_label"]
    header_drop = y_limits["header_drop"]
    text_drop = y_limits["text_drop"]
    title_lift = y_limits["title_lift"]

    for row in rows:
        truth_center = x
        belief_center = x + group_width
        metric_centers.append((truth_center + belief_center) / 2)
        group_centers.extend([truth_center, belief_center])

        draw_pair(ax, truth_center, row, "truth", score_mode, bar_width)
        draw_pair(ax, belief_center, row, "belief", score_mode, bar_width)

        ax.plot(
            [truth_center - 0.42, belief_center + 0.42],
            [top_y, top_y],
            color="black",
            linewidth=1.5,
            clip_on=False,
        )
        ax.plot(
            [(truth_center + belief_center) / 2,
             (truth_center + belief_center) / 2],
            [top_y, top_y - header_drop],
            color="black",
            linewidth=1.5,
            clip_on=False,
        )
        ax.text(truth_center, top_y - text_drop, "truth",
                ha="center", va="top", fontsize=10, fontweight="bold")
        ax.text(belief_center, top_y - text_drop, "belief",
                ha="center", va="top", fontsize=10, fontweight="bold")
        ax.text((truth_center + belief_center) / 2, top_y + title_lift,
                row["title"], ha="center", va="bottom", fontsize=11, fontweight="bold")

        x += group_width * 2 + metric_gap

    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", color="#D9D9D9", linewidth=1.0, alpha=0.8)
    ax.set_xticks([])
    ax.set_xlim(-0.7, x - metric_gap - 0.3)

    if score_mode == "normalized":
        ax.set_ylabel("Normalized Likert score (1=-1, 3=0, 5=1)")
    elif score_mode == "centered":
        ax.set_ylabel("Mean score difference from metric grand mean")
    else:
        ax.set_ylabel("Average Likert score")
    ax.set_ylim(y_limits["y_min"], y_limits["y_max"])
    ax.yaxis.set_major_locator(MultipleLocator(1.0))

    legend = [
        Patch(facecolor=GROUP_COLORS["AI"], edgecolor="black",
              hatch=GROUP_HATCHES["AI"], label="AI"),
        Patch(facecolor=GROUP_COLORS["Human"], edgecolor="black",
              hatch=GROUP_HATCHES["Human"], label="Human"),
    ]
    ax.legend(handles=legend, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    save_figure_png_and_svg(fig, path, dpi=180)
    plt.close(fig)


def save_figure_png_and_svg(fig, path: Path, *, dpi: int = 180) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def draw_pair(ax, center: float, row: dict, prefix: str, score_mode: str, bar_width: float) -> None:
    for offset, group in [(-bar_width * 0.65, "AI"), (bar_width * 0.65, "Human")]:
        stats = row[f"{prefix}_{group.lower()}"]
        value = stats["mean"]
        if value is None:
            continue
        value = transform_score(value, row, score_mode)
        ax.bar(
            center + offset,
            value,
            width=bar_width,
            color=GROUP_COLORS[group],
            edgecolor="black",
            linewidth=1.0,
            hatch=GROUP_HATCHES[group],
            alpha=0.95,
        )


def compute_y_limits(rows: list[dict], score_mode: str) -> dict[str, float]:
    values = []
    for row in rows:
        for prefix in ("truth", "belief"):
            for group in ("ai", "human"):
                stats = row[f"{prefix}_{group}"]
                value = stats["mean"]
                if value is None:
                    continue
                values.append(transform_score(value, row, score_mode))

    if score_mode == "raw":
        max_value = max(values) if values else 5.0
        y_min = 0.0
        y_max = min(5.35, max_value + 0.65)
        y_max = max(y_max, 2.0)
        return {
            "y_min": y_min,
            "y_max": y_max,
            "top_label": y_max - 0.22,
            "header_drop": 0.20,
            "text_drop": 0.24,
            "title_lift": 0.05,
        }

    min_value = min(values) if values else -1.0
    max_value = max(values) if values else 1.0
    y_min = min(-1.05, min_value - 0.2)
    y_max = max(1.15, max_value + 0.25)
    return {
        "y_min": y_min,
        "y_max": y_max,
        "top_label": y_max - 0.13,
        "header_drop": 0.08,
        "text_drop": 0.10,
        "title_lift": 0.03,
    }


def add_holm_adjusted_p_values(rows: list[dict]) -> None:
    tests = []
    for row in rows:
        for prefix in ("truth", "belief"):
            p_value = row.get(f"{prefix}_mann_whitney_p")
            if p_value is not None:
                tests.append((row, prefix, float(p_value)))

    adjusted = holm_bonferroni([p for _, _, p in tests])
    for (row, prefix, _), q_value in zip(tests, adjusted):
        row[f"{prefix}_mann_whitney_q_holm"] = q_value

    for row in rows:
        for prefix in ("truth", "belief"):
            row.setdefault(f"{prefix}_mann_whitney_q_holm", None)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    if not p_values:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (original_index, p_value) in enumerate(indexed):
        raw_adjusted = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, raw_adjusted)
        adjusted[original_index] = running_max
    return adjusted


def significance_text(p_value: float | None, q_value: float | None) -> str:
    if p_value is None:
        return "p=NA"
    if q_value is None:
        return f"p={format_p_value(p_value)}"
    return f"p={format_p_value(p_value)}\nq={format_p_value(q_value)} {significance_label(q_value)}".rstrip()


def significance_label(q_value: float | None) -> str:
    if q_value is None:
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "n.s."


def format_p_value(value: float) -> str:
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".lstrip("0")


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.stdev(values)


def subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def transform_score(value: float, row: dict, score_mode: str) -> float:
    if score_mode == "normalized":
        normalized = normalize_likert(value)
        return 0.0 if normalized is None else normalized
    if score_mode == "centered":
        return value - row["grand_mean"]
    return value


def normalize_likert(value: float | None) -> float | None:
    if value is None:
        return None
    return (float(value) - 3.0) / 2.0


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
