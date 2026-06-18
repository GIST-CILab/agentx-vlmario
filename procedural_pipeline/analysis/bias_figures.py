from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

from procedural_pipeline.analysis.results import (
    _latest_results_csv,
    load_results,
    load_truth_map,
    metric_number,
    normalize_creator_label,
    to_number,
)
from procedural_pipeline.analysis.truth_belief_plot import (
    GROUP_COLORS,
    GROUP_HATCHES,
    fmt,
    normalize_likert,
    save_figure_png_and_svg,
)
from procedural_pipeline.games import GAMES
from procedural_pipeline.paths import project_path


METRIC_TITLES = {
    "fun": "Fun",
    "challenging": "Challenging",
    "frustrating": "Frustrating",
    "surprising": "Surprising",
    "design": "Design",
}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp1_rows, exp1_sources = load_experiment1_rows(args)
    confusion_summary = build_confusion_summary(exp1_rows, exp1_sources)
    write_confusion_outputs(out_dir, confusion_summary, exp1_rows)
    if not args.no_plot:
        heatmap_path = out_dir / "figure2_confusion_matrix.png"
        sankey_path = out_dir / "figure2_creator_sankey.png"
        save_confusion_heatmap(confusion_summary, heatmap_path)
        save_creator_sankey(confusion_summary, sankey_path)
        print(f"wrote Figure 2 heatmap -> {heatmap_path}")
        print(f"wrote Figure 2 sankey  -> {sankey_path}")

    paired_rows, exp3_sources = load_experiment3_paired_rows(args)
    slope_summary = build_slope_summary(
        paired_rows,
        exp3_sources,
        args.metrics,
        args.score_mode,
        args.slope_line_width,
    )
    write_slope_outputs(out_dir, slope_summary)
    if not args.no_plot:
        for metric in args.metrics:
            slope_path = out_dir / f"figure3_forced_slope_{metric}.png"
            save_forced_slope_graph(slope_summary, metric, slope_path, args.score_mode, args.slope_line_width)
            print(f"wrote Figure 3 slope ({metric}) -> {slope_path}")

    print(f"wrote summaries -> {out_dir}")


def parse_args() -> argparse.Namespace:
    default_root = project_path("outputs", "procedural_experiments")
    parser = argparse.ArgumentParser(
        description="Create paper-style bias figures: confusion/Sankey for creator detection and slope graphs for forced priming."
    )
    parser.add_argument("--experiment-root", default=str(default_root))
    parser.add_argument("--mario-maps-file", default=None)
    parser.add_argument("--sokoban-maps-file", default=None)
    parser.add_argument("--exp1-mario-results-csv", default=None)
    parser.add_argument("--exp1-sokoban-results-csv", default=None)
    parser.add_argument("--exp3-mario-ai-results-csv", default=None)
    parser.add_argument("--exp3-mario-human-results-csv", default=None)
    parser.add_argument("--exp3-sokoban-ai-results-csv", default=None)
    parser.add_argument("--exp3-sokoban-human-results-csv", default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["fun", "challenging", "frustrating", "surprising", "design"],
        choices=sorted(METRIC_TITLES),
        help="Metrics to draw slope graphs for.",
    )
    parser.add_argument(
        "--score-mode",
        choices=["raw", "normalized"],
        default="raw",
        help="Slope graph y-axis: raw 1-5 scores or normalized 1->-1, 3->0, 5->1.",
    )
    parser.add_argument(
        "--slope-source",
        choices=["aggregate", "raw-runs"],
        default="aggregate",
        help=(
            "aggregate draws one line per map from results CSV; raw-runs draws one line "
            "per raw LLM run, usually 60 maps x 20 runs."
        ),
    )
    parser.add_argument(
        "--slope-line-width",
        choices=["constant", "std"],
        default="std",
        help=(
            "constant uses equal line widths; std scales aggregate map lines by "
            "the mean STDEV.S of forced-Human and forced-AI 20-run scores. "
            "raw-runs always uses constant thin lines."
        ),
    )
    parser.add_argument("--output-dir", default=str(project_path("outputs", "procedural_bias_figures")))
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def load_experiment1_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    sources = {}
    for game_key in ("mario", "sokoban"):
        game = GAMES[game_key]
        results_csv = getattr(args, f"exp1_{game_key}_results_csv") or _latest_results_csv(
            str(Path(args.experiment_root) / "01_creator_judgment" / game_key)
        )
        maps_file = getattr(args, f"{game_key}_maps_file") or game.DEFAULTS["maps_file"]
        truth_map = load_truth_map(maps_file)
        game_rows = load_results(results_csv, truth_map)
        for row in game_rows:
            row["game"] = game_key
            row["confusion_category"] = confusion_category(row)
        rows.extend(game_rows)
        sources[game_key] = {"results_csv": str(results_csv), "maps_file": str(maps_file), "num_rows": len(game_rows)}
    return rows, sources


def load_experiment3_paired_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if args.slope_source == "raw-runs":
        return load_experiment3_paired_raw_run_rows(args)

    paired_rows: list[dict] = []
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
        ai_rows = load_results_with_stds(ai_csv, truth_map)
        human_rows = load_results_with_stds(human_csv, truth_map)
        ai_by_id = {row["map_id"]: row for row in ai_rows if row.get("status") == "ok"}
        human_by_id = {row["map_id"]: row for row in human_rows if row.get("status") == "ok"}
        for map_id in sorted(set(ai_by_id) & set(human_by_id)):
            paired_rows.append(
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
            "num_pairs": len(set(ai_by_id) & set(human_by_id)),
            "slope_source": "aggregate",
        }
    return paired_rows, sources


def load_results_with_stds(results_csv: str, truth_map: dict[str, str]) -> list[dict]:
    rows = []
    with Path(results_csv).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            map_id = row.get("map_id", "")
            parsed = {
                "map_id": map_id,
                "status": row.get("status", ""),
                "true_creator": truth_map.get(map_id, ""),
                "llm_believed_creator": normalize_creator_label(row.get("creator_belief", "")),
                "confidence_level": to_number(row.get("confidence_level", "")),
            }
            for metric in METRIC_TITLES:
                parsed[metric] = metric_number(row, metric)
                parsed[f"{metric}_std"] = to_number(row.get(f"{metric}_std", ""))
            rows.append(parsed)
    return rows


def load_experiment3_paired_raw_run_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    paired_rows: list[dict] = []
    sources = {}
    for game_key in ("mario", "sokoban"):
        game = GAMES[game_key]
        maps_file = getattr(args, f"{game_key}_maps_file") or game.DEFAULTS["maps_file"]
        truth_map = load_truth_map(maps_file)
        ai_raw_dir = Path(args.experiment_root) / "03_forced_creator" / game_key / "ai" / "raw"
        human_raw_dir = Path(args.experiment_root) / "03_forced_creator" / game_key / "human" / "raw"
        ai_runs = load_raw_runs_by_key(ai_raw_dir, truth_map)
        human_runs = load_raw_runs_by_key(human_raw_dir, truth_map)
        common_keys = sorted(set(ai_runs) & set(human_runs))
        for key in common_keys:
            game_name, map_id, run_index = key
            paired_rows.append(
                {
                    "game": game_name,
                    "map_id": map_id,
                    "run_index": run_index,
                    "true_creator": truth_map.get(map_id, ""),
                    "forced_ai": ai_runs[key],
                    "forced_human": human_runs[key],
                }
            )
        sources[game_key] = {
            "ai_raw_dir": str(ai_raw_dir),
            "human_raw_dir": str(human_raw_dir),
            "maps_file": str(maps_file),
            "num_ai_runs": len(ai_runs),
            "num_human_runs": len(human_runs),
            "num_pairs": len(common_keys),
            "slope_source": "raw-runs",
        }
    return paired_rows, sources


def load_raw_runs_by_key(raw_dir: Path, truth_map: dict[str, str]) -> dict[tuple[str, str, int], dict]:
    runs_by_key = {}
    game_key = raw_dir.parts[-3] if len(raw_dir.parts) >= 3 else ""
    for path in sorted(raw_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "ok":
            continue
        map_id = data.get("map_id") or path.stem
        for fallback_index, run in enumerate(data.get("llm", {}).get("runs", []) or [], start=1):
            judgment = run.get("parsed_judgment")
            if not isinstance(judgment, dict):
                continue
            run_index = int(run.get("run_index") or fallback_index)
            row = {
                "game": game_key,
                "map_id": map_id,
                "run_index": run_index,
                "true_creator": truth_map.get(map_id, ""),
            }
            for metric in METRIC_TITLES:
                row[metric] = metric_score(judgment, metric)
            runs_by_key[(game_key, map_id, run_index)] = row
    return runs_by_key


def metric_score(judgment: dict, metric: str) -> float | None:
    item = judgment.get(metric)
    if isinstance(item, dict):
        value = item.get("score")
    else:
        value = item
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def build_confusion_summary(rows: list[dict], sources: dict) -> dict:
    valid = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("true_creator") in {"AI", "Human"}
        and row.get("llm_believed_creator") in {"AI", "Human"}
    ]
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    matrix = {
        "AI": {"AI": 0, "Human": 0},
        "Human": {"AI": 0, "Human": 0},
    }
    for row in valid:
        true_label = row["true_creator"]
        guessed = row["llm_believed_creator"]
        matrix[true_label][guessed] += 1
        category = confusion_category(row)
        if category in counts:
            counts[category] += 1
    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    return {
        "figure": "Figure 2",
        "experiment": "01_creator_judgment",
        "note": "Experiment 2 does not ask for creator belief, so creator-detection confusion uses experiment 1.",
        "sources": sources,
        "num_rows": len(rows),
        "num_valid_rows": len(valid),
        "matrix": matrix,
        "counts": counts,
        "rates": {
            "accuracy": divide(tp + tn, tp + tn + fp + fn),
            "ADSR": divide(tp, tp + fn),
            "FPR": divide(fp, fp + tn),
            "FNR": divide(fn, tp + fn),
            "human_correct_rate": divide(tn, tn + fp),
        },
    }


def build_slope_summary(
    pairs: list[dict],
    sources: dict,
    metrics: list[str],
    score_mode: str,
    line_width_mode: str,
) -> dict:
    metric_rows = {}
    for metric in metrics:
        rows = []
        for pair in pairs:
            human_value = pair["forced_human"].get(metric)
            ai_value = pair["forced_ai"].get(metric)
            if human_value is None or ai_value is None:
                continue
            human_value = float(human_value)
            ai_value = float(ai_value)
            rows.append(
                {
                    "game": pair["game"],
                    "map_id": pair["map_id"],
                    "run_index": pair.get("run_index", ""),
                    "true_creator": pair["true_creator"],
                    "forced_human": human_value,
                    "forced_ai": ai_value,
                    "delta_ai_minus_human": ai_value - human_value,
                    "plot_forced_human": plot_value(human_value, score_mode),
                    "plot_forced_ai": plot_value(ai_value, score_mode),
                    "plot_delta_ai_minus_human": plot_value(ai_value, score_mode) - plot_value(human_value, score_mode),
                    "forced_human_std": pair["forced_human"].get(f"{metric}_std"),
                    "forced_ai_std": pair["forced_ai"].get(f"{metric}_std"),
                    "mean_condition_std": mean_optional(
                        [
                            pair["forced_human"].get(f"{metric}_std"),
                            pair["forced_ai"].get(f"{metric}_std"),
                        ]
                    ),
                }
            )
        metric_rows[metric] = {
            "rows": rows,
            "n": len(rows),
            "mean_forced_human": mean([row["forced_human"] for row in rows]),
            "mean_forced_ai": mean([row["forced_ai"] for row in rows]),
            "mean_delta_ai_minus_human": mean([row["delta_ai_minus_human"] for row in rows]),
            "mean_plot_forced_human": mean([row["plot_forced_human"] for row in rows]),
            "mean_plot_forced_ai": mean([row["plot_forced_ai"] for row in rows]),
            "mean_plot_delta_ai_minus_human": mean([row["plot_delta_ai_minus_human"] for row in rows]),
            "num_drop": sum(1 for row in rows if row["delta_ai_minus_human"] < 0),
            "num_rise": sum(1 for row in rows if row["delta_ai_minus_human"] > 0),
            "num_same": sum(1 for row in rows if row["delta_ai_minus_human"] == 0),
        }
    return {
        "figure": "Figure 3",
        "experiment": "03_forced_creator",
        "score_mode": score_mode,
        "line_width_mode": line_width_mode,
        "slope_source": "raw-runs" if any("run_index" in pair for pair in pairs) else "aggregate",
        "sources": sources,
        "num_pairs": len(pairs),
        "metrics": metric_rows,
    }


def save_confusion_heatmap(summary: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    matrix = summary["matrix"]
    values = [
        [matrix["AI"]["AI"], matrix["AI"]["Human"]],
        [matrix["Human"]["AI"], matrix["Human"]["Human"]],
    ]
    max_value = max(max(row) for row in values) or 1
    fig, ax = plt.subplots(figsize=(4.7, 4.2), constrained_layout=True)
    ax.imshow(values, cmap="Reds", vmin=0, vmax=max_value)
    ax.set_xticks([0, 1], labels=["Guessed AI", "Guessed Human"])
    ax.set_yticks([0, 1], labels=["True AI", "True Human"])
    ax.set_xlabel("VLM guess")
    ax.set_ylabel("Ground truth")
    ax.set_title("Creator Confusion Matrix", fontsize=12, fontweight="bold")
    for y in range(2):
        for x in range(2):
            count = values[y][x]
            ax.text(x, y, str(count), ha="center", va="center", fontsize=16, fontweight="bold")
    rates = summary["rates"]
    subtitle = f"ADSR={fmt(rates['ADSR'])}  FPR={fmt(rates['FPR'])}  FNR={fmt(rates['FNR'])}"
    ax.text(0.5, -0.22, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=9)
    save_figure_png_and_svg(fig, path, dpi=180)
    plt.close(fig)


def save_creator_sankey(summary: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    flows = [
        ("AI", "AI", summary["matrix"]["AI"]["AI"], "#7EADD8"),
        ("AI", "Human", summary["matrix"]["AI"]["Human"], "#9C9C9C"),
        ("Human", "AI", summary["matrix"]["Human"]["AI"], "#E5A0A0"),
        ("Human", "Human", summary["matrix"]["Human"]["Human"], "#E85C52"),
    ]
    totals_left = {
        "AI": sum(value for src, _dst, value, _color in flows if src == "AI"),
        "Human": sum(value for src, _dst, value, _color in flows if src == "Human"),
    }
    totals_right = {
        "AI": sum(value for _src, dst, value, _color in flows if dst == "AI"),
        "Human": sum(value for _src, dst, value, _color in flows if dst == "Human"),
    }
    total = sum(totals_left.values()) or 1
    scale = 0.72 / total
    left_x, right_x = 0.14, 0.86
    node_width = 0.055
    left_positions = node_positions(totals_left, scale)
    right_positions = node_positions(totals_right, scale)
    left_offsets = {key: 0.0 for key in totals_left}
    right_offsets = {key: 0.0 for key in totals_right}

    fig, ax = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for src, dst, value, color in flows:
        if value <= 0:
            continue
        height = value * scale
        y0 = left_positions[src][0] + left_offsets[src]
        y1 = right_positions[dst][0] + right_offsets[dst]
        draw_ribbon(ax, left_x + node_width, y0, right_x, y1, height, color, alpha=0.72)
        left_offsets[src] += height
        right_offsets[dst] += height

    draw_nodes(ax, left_x, node_width, left_positions, totals_left, "Ground truth")
    draw_nodes(ax, right_x, node_width, right_positions, totals_right, "VLM guess")
    ax.text(0.5, 0.96, "Creator Detection Flow", ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(
        0.5,
        0.05,
        f"AI→Human errors (FN): {summary['counts']['FN']} / True AI {totals_left['AI']}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    save_figure_png_and_svg(fig, path, dpi=180)
    plt.close(fig)


def save_forced_slope_graph(
    summary: dict,
    metric: str,
    path: Path,
    score_mode: str,
    line_width_mode: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    metric_summary = summary["metrics"][metric]
    rows = metric_summary["rows"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    x_human, x_ai = 0.0, 1.0
    std_values = [row["mean_condition_std"] for row in rows if row.get("mean_condition_std") is not None]
    for row in rows:
        y_human = row["plot_forced_human"]
        y_ai = row["plot_forced_ai"]
        color = "#D55E5E" if y_ai < y_human else "#5D8AC4" if y_ai > y_human else "#999999"
        line_width = slope_line_width(row, std_values, line_width_mode, summary["slope_source"])
        line_alpha = 0.08 if summary["slope_source"] == "raw-runs" else 0.40
        ax.plot([x_human, x_ai], [y_human, y_ai], color=color, alpha=line_alpha, linewidth=line_width)

    mean_human = metric_summary["mean_plot_forced_human"]
    mean_ai = metric_summary["mean_plot_forced_ai"]
    ax.plot([x_human, x_ai], [mean_human, mean_ai], color="black", linewidth=3.0, marker="o", markersize=6)
    ax.text(x_human - 0.03, mean_human, f"{mean_human:.2f}", ha="right", va="center", fontsize=9, fontweight="bold")
    ax.text(x_ai + 0.03, mean_ai, f"{mean_ai:.2f}", ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(-0.28, 1.28)
    if score_mode == "normalized":
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Normalized Likert score (1=-1, 3=0, 5=1)")
    else:
        ax.set_ylim(0.8, 5.2)
        ax.set_ylabel("Average Likert score")
    ax.set_xticks([0, 1], labels=["Forced Human", "Forced AI"])
    ax.grid(axis="y", linestyle="--", color="#D9D9D9", alpha=0.8)
    ax.set_title(f"Priming Effect: {METRIC_TITLES[metric]}", fontsize=12, fontweight="bold")
    ax.text(
        0.5,
        0.03,
        f"drop={metric_summary['num_drop']}  rise={metric_summary['num_rise']}  same={metric_summary['num_same']}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#D55E5E", lw=2, label="drops when labeled AI"),
            Line2D([0], [0], color="#5D8AC4", lw=2, label="rises when labeled AI"),
            Line2D([0], [0], color="black", lw=3, label="mean"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=1,
        frameon=False,
        fontsize=8,
    )
    save_figure_png_and_svg(fig, path, dpi=180)
    plt.close(fig)


def node_positions(totals: dict[str, int], scale: float) -> dict[str, tuple[float, float]]:
    top = 0.82
    gap = 0.10
    positions = {}
    cursor = top
    for label in ("AI", "Human"):
        height = totals[label] * scale
        positions[label] = (cursor - height, cursor)
        cursor -= height + gap
    return positions


def draw_ribbon(ax, x0: float, y0: float, x1: float, y1: float, height: float, color: str, alpha: float) -> None:
    verts = [
        (x0, y0),
        ((x0 + x1) / 2, y0),
        ((x0 + x1) / 2, y1),
        (x1, y1),
        (x1, y1 + height),
        ((x0 + x1) / 2, y1 + height),
        ((x0 + x1) / 2, y0 + height),
        (x0, y0 + height),
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def draw_nodes(ax, x: float, width: float, positions: dict, totals: dict, title: str) -> None:
    ax.text(x + width / 2, 0.90, title, ha="center", va="bottom", fontsize=10, fontweight="bold")
    for label in ("AI", "Human"):
        y0, y1 = positions[label]
        color = GROUP_COLORS[label]
        hatch = GROUP_HATCHES[label]
        ax.add_patch(Rectangle((x, y0), width, y1 - y0, facecolor=color, edgecolor="black", hatch=hatch, linewidth=1.0))
        ha = "right" if x < 0.5 else "left"
        tx = x - 0.02 if x < 0.5 else x + width + 0.02
        ax.text(tx, (y0 + y1) / 2, f"{label}\n{totals[label]}", ha=ha, va="center", fontsize=10)


def write_confusion_outputs(out_dir: Path, summary: dict, rows: list[dict]) -> None:
    (out_dir / "figure2_confusion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (out_dir / "figure2_confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["true_creator", "guessed_ai", "guessed_human"])
        writer.writeheader()
        for true_label in ("AI", "Human"):
            writer.writerow(
                {
                    "true_creator": true_label,
                    "guessed_ai": summary["matrix"][true_label]["AI"],
                    "guessed_human": summary["matrix"][true_label]["Human"],
                }
            )
    with (out_dir / "figure2_creator_detection_rows.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fields = ["game", "map_id", "true_creator", "llm_believed_creator", "confusion_category", "confidence_level"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_slope_outputs(out_dir: Path, summary: dict) -> None:
    (out_dir / "figure3_forced_slope_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (out_dir / "figure3_forced_slope_rows.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fields = [
            "metric",
            "game",
            "map_id",
            "run_index",
            "true_creator",
            "forced_human",
            "forced_ai",
            "delta_ai_minus_human",
            "forced_human_std",
            "forced_ai_std",
            "mean_condition_std",
            "plot_forced_human",
            "plot_forced_ai",
            "plot_delta_ai_minus_human",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for metric, metric_summary in summary["metrics"].items():
            for row in metric_summary["rows"]:
                writer.writerow({"metric": metric, **row})


def confusion_category(row: dict) -> str:
    true_label = row.get("true_creator")
    guessed = row.get("llm_believed_creator")
    if true_label == "AI" and guessed == "AI":
        return "TP"
    if true_label == "Human" and guessed == "Human":
        return "TN"
    if true_label == "Human" and guessed == "AI":
        return "FP"
    if true_label == "AI" and guessed == "Human":
        return "FN"
    return ""


def plot_value(value: float, score_mode: str) -> float:
    if score_mode == "normalized":
        normalized = normalize_likert(value)
        return 0.0 if normalized is None else normalized
    return value


def slope_line_width(row: dict, std_values: list[float], line_width_mode: str, slope_source: str) -> float:
    if slope_source == "raw-runs" or line_width_mode == "constant":
        return 0.55 if slope_source == "raw-runs" else 1.0
    std_value = row.get("mean_condition_std")
    if std_value is None or not std_values:
        return 1.0
    min_std = min(std_values)
    max_std = max(std_values)
    if max_std == min_std:
        return 1.6
    normalized = (float(std_value) - min_std) / (max_std - min_std)
    return 0.45 + normalized * 2.35


def mean_optional(values: list) -> float | None:
    numeric = []
    for value in values:
        if value is None:
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return mean(numeric)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


if __name__ == "__main__":
    main()
