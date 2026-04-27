import argparse
import csv
import json
from pathlib import Path

from procedural_pipeline.analyze_results import (
    _build_parser,
    ensure_results_csv,
    format_number,
    load_results,
    load_truth_map,
    mann_whitney_u_two_sided,
    safe_mean,
)
from procedural_pipeline.games import GAMES


METRICS = ["fun", "challenge", "frustration", "surprise", "design"]

PAPER_MEANS = {
    "AI": {
        "fun": 2.92,
        "challenge": 3.88,
        "frustration": 3.60,
        "surprise": 2.65,
        "design": 2.70,
    },
    "Human": {
        "fun": 3.72,
        "challenge": 3.65,
        "frustration": 2.84,
        "surprise": 2.62,
        "design": 3.57,
    },
}

PAPER_HUMAN_ACCURACY = 0.53


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
        "paper_human_accuracy_reference": PAPER_HUMAN_ACCURACY,
        "rq1_proxy": analyze_proxy_accuracy(valid_rows),
        "internal_bias": analyze_internal_bias(valid_rows),
        "paper_mean_comparison": compare_against_paper_means(valid_rows, PAPER_MEANS),
        "paper_mean_correlation": calculate_paper_mean_correlation(valid_rows, PAPER_MEANS),
    }

    print_summary(summary)
    save_outputs(args.output_dir, rows, summary)


def parse_args() -> argparse.Namespace:
    parser = _build_parser("Compare LLM ratings against paper-level summary statistics.")
    parser.add_argument("--output-dir", default="outputs/procedural_analysis_proxy", help="Directory for analysis outputs (not the pipeline's).")
    return parser.parse_args()


def analyze_proxy_accuracy(rows: list[dict]) -> dict:
    if not rows:
        return {
            "llm_accuracy": None,
            "paper_human_accuracy": PAPER_HUMAN_ACCURACY,
            "accuracy_gap": None,
        }

    correct = sum(1 for row in rows if row["true_creator"] == row["llm_believed_creator"])
    llm_accuracy = correct / len(rows)
    return {
        "llm_accuracy": llm_accuracy,
        "paper_human_accuracy": PAPER_HUMAN_ACCURACY,
        "accuracy_gap": llm_accuracy - PAPER_HUMAN_ACCURACY,
    }


def analyze_internal_bias(rows: list[dict]) -> dict:
    result = {}
    for metric in METRICS:
        belief_human = collect_metric(rows, "Human", metric)
        belief_ai = collect_metric(rows, "AI", metric)
        stat, p_value = mann_whitney_u_two_sided(belief_human, belief_ai)

        belief_human_mean = safe_mean(belief_human)
        belief_ai_mean = safe_mean(belief_ai)
        paper_trend = get_trend(PAPER_MEANS["Human"][metric], PAPER_MEANS["AI"][metric])
        llm_trend = get_trend(belief_human_mean, belief_ai_mean)

        result[metric] = {
            "belief_human_mean": belief_human_mean,
            "belief_ai_mean": belief_ai_mean,
            "paper_trend": paper_trend,
            "llm_trend": llm_trend,
            "trend_matches_paper": paper_trend == llm_trend if llm_trend else None,
            "u_stat": stat,
            "p_value": p_value,
            "significant": p_value is not None and p_value < 0.05,
        }
    return result


def compare_against_paper_means(rows: list[dict], paper_means: dict) -> dict:
    result = {}
    for belief in ["AI", "Human"]:
        belief_rows = [row for row in rows if row["llm_believed_creator"] == belief]
        metric_errors = {}
        errors = []

        for metric in METRICS:
            llm_values = [float(row[metric]) for row in belief_rows if row.get(metric) is not None]
            llm_mean = safe_mean(llm_values)
            paper_mean = paper_means[belief][metric]
            mae = abs(llm_mean - paper_mean) if llm_mean is not None else None
            metric_errors[metric] = {
                "llm_mean": llm_mean,
                "paper_mean": paper_mean,
                "absolute_error": mae,
            }
            if mae is not None:
                errors.append(mae)

        result[belief] = {
            "metrics": metric_errors,
            "group_mae": safe_mean(errors),
        }
    return result


def calculate_paper_mean_correlation(rows: list[dict], paper_means: dict) -> dict:
    result = {}
    overall_llm = []
    overall_paper = []

    for belief in ["AI", "Human"]:
        belief_rows = [row for row in rows if row["llm_believed_creator"] == belief]
        llm_points = []
        paper_points = []

        for metric in METRICS:
            llm_values = [float(row[metric]) for row in belief_rows if row.get(metric) is not None]
            llm_mean = safe_mean(llm_values)
            paper_mean = paper_means[belief][metric]
            if llm_mean is not None:
                llm_points.append(llm_mean)
                paper_points.append(paper_mean)
                overall_llm.append(llm_mean)
                overall_paper.append(paper_mean)

        result[belief] = {
            "pearson_correlation": pearson_correlation(llm_points, paper_points),
            "num_points": len(llm_points),
        }

    result["overall"] = {
        "pearson_correlation": pearson_correlation(overall_llm, overall_paper),
        "num_points": len(overall_llm),
    }
    return result


def collect_metric(rows: list[dict], belief: str, metric: str) -> list[float]:
    values = []
    for row in rows:
        if row["llm_believed_creator"] == belief and row.get(metric) is not None:
            values.append(float(row[metric]))
    return values


def get_trend(human_mean, ai_mean) -> str | None:
    if human_mean is None or ai_mean is None:
        return None
    if human_mean > ai_mean:
        return "Human > AI"
    if ai_mean > human_mean:
        return "AI > Human"
    return "Human = AI"


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]

    denominator_x = sum(value * value for value in centered_x) ** 0.5
    denominator_y = sum(value * value for value in centered_y) ** 0.5
    if denominator_x == 0 or denominator_y == 0:
        return None

    numerator = sum(a * b for a, b in zip(centered_x, centered_y))
    return numerator / (denominator_x * denominator_y)


def print_summary(summary: dict) -> None:
    game = summary.get("game", "")
    rq1 = summary["rq1_proxy"]
    print(f"=== [{game}] 1. AI/Human 구별 능력 평가 (RQ1 재현) ===")
    if rq1["llm_accuracy"] is None:
        print("유효한 행이 없습니다.\n")
    else:
        print(f"LLM 정답률: {rq1['llm_accuracy'] * 100:.1f}% (논문 사람 정답률: 약 {rq1['paper_human_accuracy'] * 100:.0f}%)")
        print("  * 해석: LLM의 정답률이 50% 근처면 사람처럼 AI 맵을 잘 구별하지 못한다는 뜻입니다.\n")

    print(f"=== [{game}] 2. LLM 내부의 인지 편향 검증 (RQ4 재현) ===")
    for metric, result in summary["internal_bias"].items():
        print(f"\n[{metric.upper()}]")
        print(
            "  - LLM 평균 점수 | "
            f"Believed Human: {format_number(result['belief_human_mean'])} / "
            f"Believed AI: {format_number(result['belief_ai_mean'])}"
        )
        if result["trend_matches_paper"] is None:
            print("  - 경향성 비교 불가")
        elif result["trend_matches_paper"]:
            print(f"  경향성 일치 (사람과 LLM 모두 {result['paper_trend']} 경향)")
        else:
            print(f"  경향성 불일치 (사람: {result['paper_trend']} / LLM: {result['llm_trend']})")
        print(
            "  - 통계적 유의미성(p-value): "
            f"{format_number(result['p_value'])}"
            f"{' (*유의미함)' if result['significant'] else ''}"
        )

    print(f"\n=== [{game}] 3. 논문 수치와의 절대 오차(MAE) 비교 ===")
    for belief, group_result in summary["paper_mean_comparison"].items():
        print(f"\n[LLM이 '{belief}'이 만들었다고 판단한 그룹]")
        for metric, metric_result in group_result["metrics"].items():
            print(
                f"  - {metric}: "
                f"LLM {format_number(metric_result['llm_mean'])} vs "
                f"논문 {format_number(metric_result['paper_mean'])} "
                f"(오차: {format_number(metric_result['absolute_error'])})"
            )
        print(f"  - 그룹 평균 MAE: {format_number(group_result['group_mae'])}")

    print(f"\n=== [{game}] 4. 논문 평균과의 상관계수(Correlation) ===")
    for belief, corr_result in summary["paper_mean_correlation"].items():
        label = "전체" if belief == "overall" else f"LLM이 '{belief}'이 만들었다고 판단한 그룹"
        print(
            f"{label}: Pearson r={format_number(corr_result['pearson_correlation'])} "
            f"(points={corr_result['num_points']})"
        )


def save_outputs(output_dir: str, rows: list[dict], summary: dict) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "paper_proxy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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

    with (out_dir / "paper_proxy_mae.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["belief_group", "metric", "llm_mean", "paper_mean", "absolute_error", "group_mae"],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
