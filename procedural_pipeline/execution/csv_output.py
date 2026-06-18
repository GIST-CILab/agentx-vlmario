import csv
from pathlib import Path

from procedural_pipeline.judge import EXPERIENCE_METRICS, EXPERIENCE_METRIC_ALIASES


CSV_COLUMNS = [
    "map_index",
    "map_id",
    "status",
    "experiment",
    "true_creator",
    "forced_creator",
    "evaluation_steps_in_prompt",
    "map_path",
    "video_path",
    "temperature",
    "top_p",
    "num_runs",
    "num_parsed",
    "creator_belief",
    "creator_belief_votes",
    "confidence_level",
    "reasoning_for_creator_belief",
    "fun",
    "fun_std",
    "challenging",
    "challenging_std",
    "frustrating",
    "frustrating_std",
    "surprising",
    "surprising_std",
    "design",
    "design_std",
    "error",
]


def flatten_result(result: dict) -> dict:
    judgment = result.get("judgment") or {}
    row = {
        "map_index": result.get("map_index", ""),
        "map_id": result.get("map_id", ""),
        "status": result.get("status", ""),
        "experiment": result.get("experiment", ""),
        "true_creator": result.get("true_creator", ""),
        "forced_creator": result.get("forced_creator", ""),
        "evaluation_steps_in_prompt": result.get("evaluation_steps_in_prompt", ""),
        "map_path": result.get("map_path", ""),
        "video_path": result.get("video_path", ""),
        "temperature": format_number(result.get("temperature", "")),
        "top_p": format_number(result.get("top_p", "")),
        "num_runs": result.get("num_runs", ""),
        "num_parsed": result.get("num_parsed", ""),
        "creator_belief": get_value(judgment, "creator_belief"),
        "creator_belief_votes": format_votes(judgment.get("creator_belief")),
        "confidence_level": format_number(get_confidence(judgment)),
        "reasoning_for_creator_belief": judgment.get("reasoning_for_creator_belief", ""),
        "error": result.get("error", ""),
    }
    for metric in EXPERIENCE_METRICS:
        row[metric] = format_number(get_score(judgment, metric))
        row[f"{metric}_std"] = format_number(get_field(judgment, metric, "std"))
    return row


def write_csv(path: str, results: list[dict]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(flatten_result(result))


def get_score(judgment: dict, name: str):
    item = get_metric_value(judgment, name, {})
    if isinstance(item, dict):
        score = item.get("score")
        if score not in (None, ""):
            return score
    elif item not in (None, ""):
        return item

    nested = judgment.get("experience_ratings", {})
    if isinstance(nested, dict):
        nested_item = get_metric_value(nested, name, {})
        if isinstance(nested_item, dict):
            score = nested_item.get("score")
            if score not in (None, ""):
                return score
        elif nested_item not in (None, ""):
            return nested_item
    return ""


def get_field(judgment: dict, name: str, field: str):
    item = get_metric_value(judgment, name, {})
    if isinstance(item, dict):
        value = item.get(field)
        if value not in (None, ""):
            return value
    return ""


def get_metric_value(data: dict, name: str, default=None):
    if name in data:
        return data.get(name)
    for alias in EXPERIENCE_METRIC_ALIASES.get(name, ()):
        if alias in data:
            return data.get(alias)
    return default


def get_value(judgment: dict, name: str):
    item = judgment.get(name, {})
    if isinstance(item, dict):
        return item.get("value", "")
    if item not in (None, ""):
        return item
    return ""


def get_confidence(judgment: dict):
    item = judgment.get("confidence_level", {})
    if isinstance(item, dict):
        for key in ("value", "score"):
            value = item.get(key)
            if value not in (None, ""):
                return value
    elif item not in (None, ""):
        return item
    return ""


def format_votes(value):
    if not isinstance(value, dict):
        return ""
    votes = value.get("votes")
    if not isinstance(votes, dict) or not votes:
        return ""
    return ";".join(f"{label}={count}" for label, count in sorted(votes.items(), key=lambda kv: -kv[1]))


def format_number(value):
    if value in (None, ""):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.3f}" if isinstance(value, float) else str(value)
    return str(value)
