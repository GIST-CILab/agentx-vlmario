import argparse
import csv
import json
from pathlib import Path

from procedural_pipeline.execution.options import add_evaluation_steps_prompt_arg, add_runtime_args
from procedural_pipeline.analysis.paper_proxy import (
    PAPER_HUMAN_ACCURACY,
    PAPER_MEANS,
    analyze_internal_bias,
    analyze_proxy_accuracy,
    calculate_paper_mean_correlation,
    compare_against_paper_means,
    print_summary,
)
from procedural_pipeline.analysis.results import (
    _latest_results_csv,
    ensure_results_csv,
    load_results,
    load_truth_map,
)
from procedural_pipeline.games import GAMES


def main() -> None:
    args = parse_args()

    all_rows: list[dict] = []
    game_counts: dict = {}
    for game_key in sorted(GAMES.keys()):
        game = GAMES[game_key]
        defaults = game.DEFAULTS
        csv_override = getattr(args, f"{game_key}_results_csv", None)
        maps_override = getattr(args, f"{game_key}_maps_file", None)

        if csv_override:
            results_csv = csv_override
        else:
            try:
                results_csv = _latest_results_csv(defaults["output_dir"])
            except FileNotFoundError:
                if args.skip_run:
                    raise
                per_game_args = _per_game_pipeline_args(game, args)
                results_csv = ensure_results_csv(game, per_game_args)

        maps_file = maps_override or defaults["maps_file"]
        rows = load_game_rows(game_key, results_csv, maps_file)
        all_rows.extend(rows)
        game_counts[f"{game_key}_total"] = len(rows)
        game_counts[f"{game_key}_valid"] = count_valid_rows(rows)

    valid_rows = [row for row in all_rows if row["status"] == "ok" and row["llm_believed_creator"] in {"Human", "AI"}]

    summary = {
        "game": "combined",
        "num_total_rows": len(all_rows),
        "num_valid_rows": len(valid_rows),
        "paper_human_accuracy_reference": PAPER_HUMAN_ACCURACY,
        "game_counts": game_counts,
        "rq1_proxy": analyze_proxy_accuracy(valid_rows),
        "internal_bias": analyze_internal_bias(valid_rows),
        "paper_mean_comparison": compare_against_paper_means(valid_rows, PAPER_MEANS),
        "paper_mean_correlation": calculate_paper_mean_correlation(valid_rows, PAPER_MEANS),
    }

    print_summary(summary)
    save_outputs(args.output_dir, all_rows, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combined Mario+Sokoban comparison against paper statistics.")
    for game_key in sorted(GAMES.keys()):
        parser.add_argument(f"--{game_key}-results-csv", default=None)
        parser.add_argument(f"--{game_key}-maps-file", default=None)
    parser.add_argument("--output-dir", default="outputs/procedural_combined_analysis")
    parser.add_argument("--skip-run", action="store_true", help="Do not auto-run any game's pipeline when its CSV is missing.")
    # Pipeline args shared across games when auto-running.
    add_runtime_args(parser)
    add_evaluation_steps_prompt_arg(parser)
    return parser.parse_args()


def _per_game_pipeline_args(game, args) -> argparse.Namespace:
    """Assemble a pipeline Namespace for a specific game, pulling shared values from combined args."""
    data = {
        "game": game.KEY,
        "maps_file": game.DEFAULTS["maps_file"],
        "criteria_file": game.DEFAULTS["criteria_file"],
        "pipeline_output_dir": game.DEFAULTS["output_dir"],
        "output_dir": game.DEFAULTS["output_dir"],
        "model": args.model,
        "limit": args.limit,
        "num_runs": args.num_runs,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "concurrency": args.concurrency,
        "skip_run": False,
    }
    # Pick up game-specific defaults (e.g. sokoban --solver-power) via a throwaway parser.
    tmp = argparse.ArgumentParser(add_help=False)
    game.add_arguments(tmp)
    game_defaults = vars(tmp.parse_args([]))
    data.update(game_defaults)
    return argparse.Namespace(**data)


def load_game_rows(game: str, results_csv: str, maps_file: str) -> list[dict]:
    truth_map = load_truth_map(maps_file)
    rows = load_results(results_csv, truth_map)
    for row in rows:
        row["game"] = game
    return rows


def count_valid_rows(rows: list[dict]) -> int:
    return sum(1 for row in rows if row["status"] == "ok" and row["llm_believed_creator"] in {"Human", "AI"})


def save_outputs(output_dir: str, rows: list[dict], summary: dict) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "combined_paper_proxy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (out_dir / "combined_annotated_results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "game",
                "map_id",
                "status",
                "true_creator",
                "llm_believed_creator",
                "confidence_level",
                "fun",
                "challenging",
                "frustrating",
                "surprising",
                "design",
                "reasoning",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "game": row.get("game", ""),
                "map_id": row.get("map_id", ""),
                "status": row.get("status", ""),
                "true_creator": row.get("true_creator", ""),
                "llm_believed_creator": row.get("llm_believed_creator", ""),
                "confidence_level": row.get("confidence_level", ""),
                "fun": row.get("fun", ""),
                "challenging": row.get("challenging", ""),
                "frustrating": row.get("frustrating", ""),
                "surprising": row.get("surprising", ""),
                "design": row.get("design", ""),
                "reasoning": row.get("reasoning", ""),
                "error": row.get("error", ""),
            })

    csv_rows = []
    for belief, group_result in summary["paper_mean_comparison"].items():
        for metric, metric_result in group_result["metrics"].items():
            csv_rows.append({
                "belief_group": belief,
                "metric": metric,
                "llm_mean": metric_result["llm_mean"],
                "paper_mean": metric_result["paper_mean"],
                "absolute_error": metric_result["absolute_error"],
                "group_mae": group_result["group_mae"],
            })

    with (out_dir / "combined_paper_proxy_mae.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["belief_group", "metric", "llm_mean", "paper_mean", "absolute_error", "group_mae"],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
