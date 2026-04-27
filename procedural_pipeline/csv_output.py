import csv
from pathlib import Path


CSV_COLUMNS = [
    "map_index",
    "map_id",
    "status",
    "map_path",
    "video_path",
    "num_runs",
    "num_parsed",
    "creator_belief",
    "creator_belief_votes",
    "confidence_level",
    "reasoning_for_creator_belief",
    "enjoyment",
    "enjoyment_std",
    "difficulty",
    "difficulty_std",
    "frustration",
    "frustration_std",
    "novelty",
    "novelty_std",
    "aesthetics",
    "aesthetics_std",
    "error",
]


def flatten_result(result: dict) -> dict:
    judgment = result.get("judgment") or {}
    return {
        "map_index": result.get("map_index", ""),
        "map_id": result.get("map_id", ""),
        "status": result.get("status", ""),
        "map_path": result.get("map_path", ""),
        "video_path": result.get("video_path", ""),
        "num_runs": result.get("num_runs", ""),
        "num_parsed": result.get("num_parsed", ""),
        "creator_belief": get_value(judgment, "creator_belief"),
        "creator_belief_votes": format_votes(judgment.get("creator_belief")),
        "confidence_level": format_number(get_confidence(judgment)),
        "reasoning_for_creator_belief": judgment.get("reasoning_for_creator_belief", ""),
        "enjoyment": format_number(get_score(judgment, "enjoyment")),
        "enjoyment_std": format_number(get_field(judgment, "enjoyment", "std")),
        "difficulty": format_number(get_score(judgment, "difficulty")),
        "difficulty_std": format_number(get_field(judgment, "difficulty", "std")),
        "frustration": format_number(get_score(judgment, "frustration")),
        "frustration_std": format_number(get_field(judgment, "frustration", "std")),
        "novelty": format_number(get_score(judgment, "novelty")),
        "novelty_std": format_number(get_field(judgment, "novelty", "std")),
        "aesthetics": format_number(get_score(judgment, "aesthetics")),
        "aesthetics_std": format_number(get_field(judgment, "aesthetics", "std")),
        "error": result.get("error", ""),
    }


def write_csv(path: str, results: list[dict]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(flatten_result(result))


def get_score(judgment: dict, name: str):
    item = judgment.get(name, {})
    if isinstance(item, dict):
        score = item.get("score")
        if score not in (None, ""):
            return score
    elif item not in (None, ""):
        return item

    nested = judgment.get("experience_ratings", {})
    if isinstance(nested, dict):
        nested_item = nested.get(name, {})
        if isinstance(nested_item, dict):
            score = nested_item.get("score")
            if score not in (None, ""):
                return score
        elif nested_item not in (None, ""):
            return nested_item
    return ""


def get_field(judgment: dict, name: str, field: str):
    item = judgment.get(name, {})
    if isinstance(item, dict):
        value = item.get(field)
        if value not in (None, ""):
            return value
    return ""


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
