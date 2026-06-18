from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from procedural_pipeline.analysis.results import _latest_results_csv, load_results, load_truth_map
from procedural_pipeline.games import GAMES
from procedural_pipeline.paths import project_path


LABELS = ("AI", "Human")


def main() -> None:
    args = parse_args()
    rows, sources = load_experiment1_rows(args)
    valid_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("true_creator") in LABELS
        and row.get("llm_believed_creator") in LABELS
    ]

    by_game = {
        game_key: summarize_detection_and_confidence(
            [row for row in valid_rows if row.get("game") == game_key]
        )
        for game_key in ("mario", "sokoban")
    }
    combined = summarize_detection_and_confidence(valid_rows)

    summary = {
        "experiment": "01_creator_judgment",
        "sources": sources,
        "num_rows": len(rows),
        "num_valid_rows": len(valid_rows),
        "definitions": {
            "ADSR": "AI detection success rate: true-AI maps predicted as AI = TP / (TP + FN).",
            "FPR": "False positive rate: true-Human maps predicted as AI = FP / (FP + TN).",
            "FNR": "False negative rate: true-AI maps predicted as Human = FN / (TP + FN).",
            "confidence_level": "Creator-choice confidence reported by the model, averaged with STDEV.S.",
        },
        "combined": combined,
        "by_game": by_game,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(out_dir, summary, valid_rows)
    print_summary(summary)
    print(f"wrote summary -> {out_dir / 'detection_confidence_summary.json'}")
    print(f"wrote tables  -> {out_dir}")


def parse_args() -> argparse.Namespace:
    default_root = project_path("outputs", "procedural_experiments", "01_creator_judgment")
    parser = argparse.ArgumentParser(
        description="Compute experiment 1 AI-detection rates and creator-confidence summaries."
    )
    parser.add_argument("--experiment-root", default=str(default_root))
    parser.add_argument("--mario-results-csv", default=None)
    parser.add_argument("--sokoban-results-csv", default=None)
    parser.add_argument("--mario-maps-file", default=None)
    parser.add_argument("--sokoban-maps-file", default=None)
    parser.add_argument("--output-dir", default=str(project_path("outputs", "procedural_detection_confidence")))
    return parser.parse_args()


def load_experiment1_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    sources = {}
    for game_key in ("mario", "sokoban"):
        game = GAMES[game_key]
        results_csv = getattr(args, f"{game_key}_results_csv") or _latest_results_csv(
            str(Path(args.experiment_root) / game_key)
        )
        maps_file = getattr(args, f"{game_key}_maps_file") or game.DEFAULTS["maps_file"]
        truth_map = load_truth_map(maps_file)
        game_rows = load_results(results_csv, truth_map)
        for row in game_rows:
            row["game"] = game_key
            row["correct_creator_belief"] = row.get("true_creator") == row.get("llm_believed_creator")
            row["confusion_category"] = confusion_category(row)
        rows.extend(game_rows)
        sources[game_key] = {
            "results_csv": str(results_csv),
            "maps_file": str(maps_file),
            "num_rows": len(game_rows),
        }
    return rows, sources


def summarize_detection_and_confidence(rows: list[dict]) -> dict:
    counts = confusion_counts(rows)
    return {
        "num_rows": len(rows),
        "confusion_counts": counts,
        "rates": detection_rates(counts),
        "confidence": {
            "overall": describe_confidence(rows),
            "by_correctness": {
                "correct": describe_confidence([row for row in rows if row.get("correct_creator_belief")]),
                "incorrect": describe_confidence([row for row in rows if not row.get("correct_creator_belief")]),
            },
            "by_true_creator": {
                label: describe_confidence([row for row in rows if row.get("true_creator") == label])
                for label in LABELS
            },
            "by_believed_creator": {
                label: describe_confidence([row for row in rows if row.get("llm_believed_creator") == label])
                for label in LABELS
            },
            "by_confusion_category": {
                category: describe_confidence([row for row in rows if row.get("confusion_category") == category])
                for category in ("TP", "TN", "FP", "FN")
            },
        },
    }


def confusion_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for row in rows:
        category = confusion_category(row)
        if category in counts:
            counts[category] += 1
    return counts


def confusion_category(row: dict) -> str:
    true_label = row.get("true_creator")
    predicted = row.get("llm_believed_creator")
    if true_label == "AI" and predicted == "AI":
        return "TP"
    if true_label == "Human" and predicted == "Human":
        return "TN"
    if true_label == "Human" and predicted == "AI":
        return "FP"
    if true_label == "AI" and predicted == "Human":
        return "FN"
    return ""


def detection_rates(counts: dict[str, int]) -> dict:
    tp = counts.get("TP", 0)
    tn = counts.get("TN", 0)
    fp = counts.get("FP", 0)
    fn = counts.get("FN", 0)
    total = tp + tn + fp + fn
    return {
        "accuracy": divide(tp + tn, total),
        "ADSR": divide(tp, tp + fn),
        "FPR": divide(fp, fp + tn),
        "FNR": divide(fn, tp + fn),
        "human_correct_rate": divide(tn, tn + fp),
    }


def describe_confidence(rows: list[dict]) -> dict:
    values = [
        float(row["confidence_level"])
        for row in rows
        if row.get("confidence_level") is not None
    ]
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev_s(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "values": values,
    }


def write_outputs(out_dir: Path, summary: dict, rows: list[dict]) -> None:
    (out_dir / "detection_confidence_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_detection_csv(out_dir / "detection_rates.csv", summary)
    write_confidence_csv(out_dir / "confidence_by_game.csv", summary)
    write_map_rows_csv(out_dir / "creator_detection_rows.csv", rows)


def write_detection_csv(path: Path, summary: dict) -> None:
    fields = ["scope", "TP", "TN", "FP", "FN", "accuracy", "ADSR", "FPR", "FNR", "human_correct_rate"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(flatten_detection_row("combined", summary["combined"]))
        for game_key, game_summary in summary["by_game"].items():
            writer.writerow(flatten_detection_row(game_key, game_summary))


def flatten_detection_row(scope: str, data: dict) -> dict:
    counts = data["confusion_counts"]
    rates = data["rates"]
    return {
        "scope": scope,
        "TP": counts["TP"],
        "TN": counts["TN"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "accuracy": fmt(rates["accuracy"]),
        "ADSR": fmt(rates["ADSR"]),
        "FPR": fmt(rates["FPR"]),
        "FNR": fmt(rates["FNR"]),
        "human_correct_rate": fmt(rates["human_correct_rate"]),
    }


def write_confidence_csv(path: Path, summary: dict) -> None:
    fields = ["scope", "grouping", "group", "n", "mean", "std", "min", "max"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        write_confidence_rows(writer, "combined", summary["combined"]["confidence"])
        for game_key, game_summary in summary["by_game"].items():
            write_confidence_rows(writer, game_key, game_summary["confidence"])


def write_confidence_rows(writer: csv.DictWriter, scope: str, confidence: dict) -> None:
    writer.writerow(flatten_confidence_row(scope, "overall", "all", confidence["overall"]))
    for grouping, groups in (
        ("correctness", confidence["by_correctness"]),
        ("true_creator", confidence["by_true_creator"]),
        ("believed_creator", confidence["by_believed_creator"]),
        ("confusion_category", confidence["by_confusion_category"]),
    ):
        for group, stats in groups.items():
            writer.writerow(flatten_confidence_row(scope, grouping, group, stats))


def flatten_confidence_row(scope: str, grouping: str, group: str, stats: dict) -> dict:
    return {
        "scope": scope,
        "grouping": grouping,
        "group": group,
        "n": stats["n"],
        "mean": fmt(stats["mean"]),
        "std": fmt(stats["std"]),
        "min": fmt(stats["min"]),
        "max": fmt(stats["max"]),
    }


def write_map_rows_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "game",
        "map_id",
        "true_creator",
        "llm_believed_creator",
        "correct_creator_belief",
        "confusion_category",
        "confidence_level",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def print_summary(summary: dict) -> None:
    for scope, data in [("combined", summary["combined"]), *summary["by_game"].items()]:
        rates = data["rates"]
        counts = data["confusion_counts"]
        print(
            f"[{scope}] TP={counts['TP']} TN={counts['TN']} FP={counts['FP']} FN={counts['FN']} "
            f"ADSR={fmt(rates['ADSR'])} FPR={fmt(rates['FPR'])} FNR={fmt(rates['FNR'])}"
        )


def divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def stdev_s(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
