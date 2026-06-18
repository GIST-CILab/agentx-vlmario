import argparse
import csv
import json
import math
from pathlib import Path

from procedural_pipeline.execution.options import add_evaluation_steps_prompt_arg, add_runtime_args
from procedural_pipeline.execution.runner import run as run_pipeline
from procedural_pipeline.games import GAMES


METRICS = [
    ("fun", "fun"),
    ("challenging", "challenging"),
    ("frustrating", "frustrating"),
    ("surprising", "surprising"),
    ("design", "design"),
]
METRIC_ALIASES = {
    "fun": ("enjoyment",),
    "challenging": ("difficulty", "challenge"),
    "frustrating": ("frustration",),
    "surprising": ("novelty", "surprise"),
    "design": ("aesthetics",),
}


def main() -> None:
    args = parse_args()
    game = GAMES[args.game]
    defaults = game.DEFAULTS

    results_csv = args.results_csv or ensure_results_csv(game, args)
    maps_file = args.maps_file or defaults["maps_file"]

    truth_map = load_truth_map(maps_file)
    rows = load_results(results_csv, truth_map)

    valid_rows = [row for row in rows if row["status"] == "ok" and row["llm_believed_creator"] in {"Human", "AI"}]
    summary = {
        "game": args.game,
        "results_csv": str(results_csv),
        "num_total_rows": len(rows),
        "num_valid_rows": len(valid_rows),
        "rq1": analyze_creator_classification(valid_rows),
        "rq4": analyze_experience_metrics(valid_rows),
    }

    print_summary(summary)
    save_outputs(args.output_dir, rows, summary)


def parse_args() -> argparse.Namespace:
    parser = _build_parser("Analyze procedural PCG evaluation results.")
    parser.add_argument("--output-dir", default="outputs/procedural_analysis", help="Directory for analysis outputs (not the pipeline's).")
    return parser.parse_args()


def _build_parser(description: str) -> argparse.ArgumentParser:
    """Build an analyze-style CLI parser that also knows how to auto-run the pipeline."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--game", required=True, choices=sorted(GAMES.keys()))
    prelim, _ = pre.parse_known_args()
    game = GAMES[prelim.game]

    parser = argparse.ArgumentParser(description=description, parents=[pre])
    parser.add_argument("--results-csv", default=None, help="CSV path (defaults to newest results_*.csv in the game's output dir).")
    parser.add_argument("--maps-file", default=None, help=f"Maps file (defaults to {game.DEFAULTS['maps_file']}).")
    parser.add_argument("--skip-run", action="store_true", help="Do not auto-run the pipeline when results CSV is missing.")

    # Pipeline args (used only when we need to auto-run)
    parser.add_argument("--pipeline-output-dir", default=game.DEFAULTS["output_dir"], help="Pipeline output dir used when auto-running.")
    parser.add_argument("--criteria-file", default=game.DEFAULTS["criteria_file"])
    add_runtime_args(parser)
    add_evaluation_steps_prompt_arg(parser)
    game.add_arguments(parser)
    return parser


def _latest_results_csv(output_dir: str) -> str:
    out = Path(output_dir)
    candidates = sorted(out.glob("results_*.csv"))
    if not candidates:
        fallback = out / "results.csv"
        if fallback.exists():
            return str(fallback)
        raise FileNotFoundError(f"No results_*.csv found under {out}")
    return str(candidates[-1])


def ensure_results_csv(game, args) -> str:
    """Return the latest results CSV for `game`; auto-run the pipeline if missing."""
    output_dir = args.pipeline_output_dir if hasattr(args, "pipeline_output_dir") else game.DEFAULTS["output_dir"]
    try:
        return _latest_results_csv(output_dir)
    except FileNotFoundError:
        if getattr(args, "skip_run", False):
            raise
        print(
            f"[{game.KEY}] no results_*.csv found under {output_dir} — running the pipeline first.\n"
            f"  (use --skip-run to disable this auto-generation)"
        )
        pipeline_args = _build_pipeline_args(game, args)
        run_pipeline(game, pipeline_args)
        return _latest_results_csv(output_dir)


def _build_pipeline_args(game, args) -> argparse.Namespace:
    """Derive a Namespace that `runner.run` expects from analyze-style CLI args."""
    from dotenv import load_dotenv

    load_dotenv(override=True)
    data = vars(args).copy()
    defaults = game.DEFAULTS
    # Pipeline's --output-dir is derived from pipeline_output_dir (the analyze script owns --output-dir).
    data["output_dir"] = getattr(args, "pipeline_output_dir", None) or defaults["output_dir"]
    # Analyze CLI defaults maps-file / criteria-file to None to distinguish "not provided" from user override.
    # Fall back to the game's defaults before handing the Namespace to the pipeline runner.
    if not data.get("maps_file"):
        data["maps_file"] = defaults["maps_file"]
    if not data.get("criteria_file"):
        data["criteria_file"] = defaults["criteria_file"]
    return argparse.Namespace(**data)


def load_truth_map(maps_file: str) -> dict[str, str]:
    ordered_map_ids = list(json.loads(Path(maps_file).read_text(encoding="utf-8")).keys())
    truth_map = {}
    for index, map_id in enumerate(ordered_map_ids):
        truth_map[map_id] = "Human" if index < 15 else "AI"
    return truth_map


def load_results(results_csv: str, truth_map: dict[str, str]) -> list[dict]:
    rows = []
    with Path(results_csv).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            map_id = row.get("map_id", "")
            rows.append({
                "map_id": map_id,
                "status": row.get("status", ""),
                "true_creator": truth_map.get(map_id, ""),
                "llm_believed_creator": normalize_creator_label(row.get("creator_belief", "")),
                "confidence_level": to_number(row.get("confidence_level", "")),
                "fun": metric_number(row, "fun"),
                "challenging": metric_number(row, "challenging"),
                "frustrating": metric_number(row, "frustrating"),
                "surprising": metric_number(row, "surprising"),
                "design": metric_number(row, "design"),
                "reasoning": row.get("reasoning_for_creator_belief", ""),
                "error": row.get("error", ""),
            })
    return rows


def normalize_creator_label(value: str) -> str:
    lowered = (value or "").strip().lower()
    if lowered in {"human", "human-designed", "human designed"}:
        return "Human"
    if lowered in {"ai", "ai-generated", "ai generated"}:
        return "AI"
    return ""


def to_number(value: str):
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def metric_number(row: dict, metric: str):
    value = row.get(metric, "")
    if str(value).strip():
        return to_number(value)
    for alias in METRIC_ALIASES.get(metric, ()):
        value = row.get(alias, "")
        if str(value).strip():
            return to_number(value)
    return None


def analyze_creator_classification(rows: list[dict]) -> dict:
    if not rows:
        return {"accuracy": None, "fpr": None, "fnr": None, "counts": {}}

    tn = fp = fn = tp = 0
    for row in rows:
        true_label = row["true_creator"]
        pred_label = row["llm_believed_creator"]
        if true_label == "Human" and pred_label == "Human":
            tn += 1
        elif true_label == "Human" and pred_label == "AI":
            fp += 1
        elif true_label == "AI" and pred_label == "Human":
            fn += 1
        elif true_label == "AI" and pred_label == "AI":
            tp += 1

    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None
    return {
        "accuracy": accuracy,
        "fpr": fpr,
        "fnr": fnr,
        "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def analyze_experience_metrics(rows: list[dict]) -> dict:
    result = {}
    for display_name, field_name in METRICS:
        truth_human = collect_metric(rows, "true_creator", "Human", display_name)
        truth_ai = collect_metric(rows, "true_creator", "AI", display_name)
        belief_human = collect_metric(rows, "llm_believed_creator", "Human", display_name)
        belief_ai = collect_metric(rows, "llm_believed_creator", "AI", display_name)

        truth_u, truth_p = mann_whitney_u_two_sided(truth_human, truth_ai)
        belief_u, belief_p = mann_whitney_u_two_sided(belief_human, belief_ai)

        result[field_name] = {
            "truth_human_mean": safe_mean(truth_human),
            "truth_ai_mean": safe_mean(truth_ai),
            "truth_p_value": truth_p,
            "truth_u": truth_u,
            "belief_human_mean": safe_mean(belief_human),
            "belief_ai_mean": safe_mean(belief_ai),
            "belief_p_value": belief_p,
            "belief_u": belief_u,
            "belief_bias_detected": belief_p is not None and belief_p < 0.05,
        }
    return result


def collect_metric(rows: list[dict], label_field: str, label_value: str, metric_field: str) -> list[float]:
    values = []
    for row in rows:
        if row.get(label_field) == label_value and row.get(metric_field) is not None:
            values.append(float(row[metric_field]))
    return values


def mann_whitney_u_two_sided(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    if not x or not y:
        return None, None

    combined = [(value, 0) for value in x] + [(value, 1) for value in y]
    combined.sort(key=lambda item: item[0])

    ranks = [0.0] * len(combined)
    tie_sizes = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        tie_sizes.append(j - i)
        i = j

    rank_sum_x = sum(rank for rank, (_, group) in zip(ranks, combined) if group == 0)
    n1 = len(x)
    n2 = len(y)
    u1 = rank_sum_x - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    mean_u = n1 * n2 / 2.0
    n = n1 + n2
    tie_correction = sum(t ** 3 - t for t in tie_sizes)
    variance = (n1 * n2 / 12.0) * ((n + 1) - tie_correction / (n * (n - 1))) if n > 1 else 0.0
    if variance <= 0:
        return u, 1.0

    z = (u - mean_u + 0.5) / math.sqrt(variance)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return u, p


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def print_summary(summary: dict) -> None:
    rq1 = summary["rq1"]
    print(f"=== [{summary['game']}] 1. AI/Human 구별 능력 평가 (RQ1) ===")
    if rq1["accuracy"] is None:
        print("유효한 행이 없습니다.\n")
    else:
        print(f"정확도 (Accuracy): {rq1['accuracy'] * 100:.1f}%")
        print(f"위양성율 (FPR - Human을 AI로 오인): {rq1['fpr'] * 100:.1f}%")
        print(f"위음성율 (FNR - AI를 Human으로 오인): {rq1['fnr'] * 100:.1f}%\n")

    print(f"=== [{summary['game']}] 2. 실제 제작자(Truth) vs LLM의 믿음(Belief)에 따른 경험 평가 (RQ4) ===")
    for metric_name, metric_result in summary["rq4"].items():
        print(f"\n[{metric_name.upper()} 지표 분석]")
        print(
            "  - 실제 (Truth) 기준 평균: "
            f"Human={format_number(metric_result['truth_human_mean'])}, "
            f"AI={format_number(metric_result['truth_ai_mean'])} "
            f"(p-value: {format_number(metric_result['truth_p_value'])})"
        )
        print(
            "  - 믿음 (Belief) 기준 평균: "
            f"Human={format_number(metric_result['belief_human_mean'])}, "
            f"AI={format_number(metric_result['belief_ai_mean'])} "
            f"(p-value: {format_number(metric_result['belief_p_value'])})"
        )
        if metric_result["belief_bias_detected"]:
            print("    * LLM은 자신이 판단한 제작자에 따라 유의미한 점수 편향(Bias)을 보였습니다!")


def save_outputs(output_dir: str, rows: list[dict], summary: dict) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else [
        "map_id", "status", "true_creator", "llm_believed_creator", "confidence_level",
        "fun", "challenging", "frustrating", "surprising", "design", "reasoning", "error",
    ]
    with (out_dir / "annotated_results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def format_number(value):
    if value is None:
        return "NA"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    main()
