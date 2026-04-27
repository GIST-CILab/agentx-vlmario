import base64
import json
import os
import statistics
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


GAME_PROFILES = {
    "mario": {
        "game_name": "Super Mario Bros",
        "subject_noun": "level",
        "verb": "playing",
        "observation_cue": "the layout, enemy placement, difficulty flow, and how the player interacts with the world",
    },
    "sokoban": {
        "game_name": "Sokoban",
        "subject_noun": "puzzle",
        "verb": "solving",
        "observation_cue": "the layout, box and goal placement, and how the player pushes boxes onto targets",
    },
}


EXPERIENCE_METRICS = ("enjoyment", "difficulty", "frustration", "novelty", "aesthetics")


class EvaluationTraceError(RuntimeError):
    def __init__(self, message: str, trace: dict):
        super().__init__(message)
        self.trace = trace


def load_criteria(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_game_profile(name_or_profile) -> dict:
    if isinstance(name_or_profile, dict):
        return name_or_profile
    if name_or_profile in GAME_PROFILES:
        return GAME_PROFILES[name_or_profile]
    raise ValueError(f"Unknown game profile: {name_or_profile}")


def require_api_key() -> str:
    api_key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPEN_ROUTER_API_KEY or OPENROUTER_API_KEY is required.")
    return api_key


# --------------------------------------------------------------------- #
# G-Eval style Auto Chain-of-Thoughts (CoT) evaluation-step generation.
# Called once per game, before any level is evaluated.
# --------------------------------------------------------------------- #
def generate_evaluation_steps(criteria: dict, game_profile, model: str) -> dict:
    profile = resolve_game_profile(game_profile)
    api_key = require_api_key()

    metrics = criteria["experience_metrics"]["metrics"]
    lines = [
        f"You are designing a standardized evaluation protocol for gameplay videos of {profile['game_name']} {profile['subject_noun']}s.",
        f"A reviewer will watch a short video of a player {profile['verb']} the {profile['subject_noun']} and rate it against the criteria below on a 1-5 Likert scale.",
        "",
        "Evaluation Criteria:",
    ]
    for metric in metrics:
        lines.append(f"- {metric['name']}: {metric['survey_item']} ({metric['description']})")

    lines.extend([
        "",
        "Evaluation Steps:",
        (
            "Generate a concise numbered chain-of-thoughts (4-6 steps) that the reviewer should follow to evaluate the video "
            "against ALL five criteria above. The steps must be objective, game-agnostic within this genre, and applicable to any "
            f"{profile['subject_noun']}. Focus on {profile['observation_cue']}."
        ),
        "",
        "Return strict JSON only:",
        '{"evaluation_steps": ["1. ...", "2. ...", "3. ...", "4. ..."]}',
    ])
    prompt = "\n".join(lines)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1,
        "top_p": 1,
        "max_tokens": 900,
        "reasoning": {"effort": "minimal", "exclude": True},
        "response_format": {"type": "json_object"},
    }

    response_text, response_json = post_openrouter(payload, api_key)
    content = response_json["choices"][0]["message"]["content"]
    parsed = parse_json_text(content)

    steps = parsed.get("evaluation_steps") or parsed.get("steps") or []
    if isinstance(steps, str):
        steps_text = steps.strip()
    else:
        steps_text = "\n".join(str(step).strip() for step in steps if str(step).strip())

    return {
        "game_profile": profile,
        "model": model,
        "prompt": prompt,
        "raw_response_text": response_text,
        "raw_response_json": response_json,
        "parsed": parsed,
        "text": steps_text,
    }


# --------------------------------------------------------------------- #
# Multi-run evaluation: reuse the same video/prompt, sample N judgments,
# then aggregate numeric scores by mean and categorical fields by mode.
# --------------------------------------------------------------------- #
def evaluate_video_multi(
    video_path: str,
    criteria: dict,
    model: str,
    game_profile,
    evaluation_steps_text: str,
    num_runs: int = 20,
    temperature: float = 1,
    top_p: float = 1,
    concurrency: int = 1,
    on_run=None,
    max_refill_rounds: int = 2,
) -> dict:
    profile = resolve_game_profile(game_profile)
    video_data_url = encode_video_to_data_url(video_path)
    prompt = build_prompt(criteria, profile, evaluation_steps_text)

    runs: list[dict] = [None] * num_runs  # type: ignore[list-item]

    def _run(index: int) -> tuple[int, dict]:
        trace = evaluate_once(
            video_path=video_path,
            video_data_url=video_data_url,
            prompt=prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )
        trace["run_index"] = index + 1
        return index, trace

    def _execute(indices: list[int]):
        if concurrency and concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_run, i) for i in indices]
                for future in as_completed(futures):
                    index, trace = future.result()
                    runs[index] = trace
                    if on_run:
                        on_run(index + 1, num_runs, trace)
        else:
            for i in indices:
                _, trace = _run(i)
                runs[i] = trace
                if on_run:
                    on_run(i + 1, num_runs, trace)

    _execute(list(range(num_runs)))

    # Refill any slots whose parse still failed, up to `max_refill_rounds` rounds.
    # This guarantees we converge to 20 parsed runs unless the API is persistently
    # broken for that specific prompt.
    for _ in range(max_refill_rounds):
        failed = [i for i, t in enumerate(runs) if not (t and t.get("parsed_judgment"))]
        if not failed:
            break
        _execute(failed)

    parsed_list = [trace["parsed_judgment"] for trace in runs if trace.get("parsed_judgment")]
    aggregated = aggregate_judgments(parsed_list)

    return {
        "prompt": prompt,
        "game_profile": profile,
        "evaluation_steps": evaluation_steps_text,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "num_runs": num_runs,
        "num_parsed": len(parsed_list),
        "aggregated_judgment": aggregated,
        "runs": runs,
    }


def evaluate_once(
    video_path: str,
    video_data_url: str,
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
) -> dict:
    api_key = require_api_key()
    trace = {
        "video_path": str(video_path),
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "request_payload": None,
        "raw_response_text": "",
        "raw_response_json": None,
        "response_content": "",
        "parsed_judgment": None,
    }

    attempts = [
        (900, "initial"),
        (1500, "retry_larger_max_tokens"),
        (1500, "retry_after_parse_failure"),
    ]
    try:
        last_parse_exc = None
        trace["attempt_history"] = []
        for attempt_index, (max_tokens, reason) in enumerate(attempts, start=1):
            payload = build_payload(model, prompt, video_data_url, max_tokens, temperature, top_p)
            trace["request_payload"] = build_storage_payload(payload, video_path)

            response_text, response_json = post_openrouter(payload, api_key)
            trace["raw_response_text"] = response_text
            trace["raw_response_json"] = response_json

            choice = response_json["choices"][0]
            content = choice["message"]["content"]
            trace["response_content"] = normalize_content_text(content)
            finish_reason = str(choice.get("finish_reason", ""))
            native_finish_reason = str(choice.get("native_finish_reason", ""))
            trace["attempt_history"].append({
                "attempt": attempt_index,
                "reason": reason,
                "max_tokens": max_tokens,
                "finish_reason": finish_reason,
                "native_finish_reason": native_finish_reason,
            })

            try:
                parsed = parse_json_text(content)
                trace["parsed_judgment"] = normalize_judgment_schema(parsed)
                trace["attempt_max_tokens"] = max_tokens
                trace.pop("parse_error", None)
                trace.pop("error", None)
                return trace
            except Exception as parse_exc:
                last_parse_exc = parse_exc
                trace["parse_error"] = str(parse_exc)
                trace["attempt_history"][-1]["parse_error"] = str(parse_exc)
                # Always try the next attempt regardless of finish_reason.
                # Gemini sometimes returns finish_reason="stop" with truncated JSON.
                continue

        trace["error"] = (
            f"Parse failed after {len(attempts)} attempts: {last_parse_exc}"
            if last_parse_exc
            else "Model response could not be parsed after retries."
        )
        return trace
    except EvaluationTraceError as exc:
        merged = {**trace, **exc.trace}
        merged["error"] = str(exc)
        return merged
    except Exception as exc:
        trace["error"] = str(exc)
        return trace


def build_payload(
    model: str,
    prompt: str,
    video_data_url: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": video_data_url}},
                ],
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "reasoning": {
            "effort": "minimal",
            "exclude": True,
        },
        "response_format": {"type": "json_object"},
    }


def build_prompt(criteria: dict, game_profile, evaluation_steps_text: str | None = None) -> str:
    profile = resolve_game_profile(game_profile)
    experience = criteria["experience_metrics"]["metrics"]
    perception = criteria["perception_metrics"]["metrics"]
    qualitative = criteria["qualitative_reasoning"]["metrics"][0]

    lines = [
        f"You are a human gamer participating in a playtesting experiment for {profile['game_name']}.",
        f"You are watching a gameplay video, but you must imagine that YOU are the one {profile['verb']} the {profile['subject_noun']}.",
        f"Evaluate the {profile['subject_noun']} based on your honest, subjective experience as a human player.",
        "Use the evaluation criteria below and return strict JSON only.",
        "Keep every reason extremely short: one sentence, max 20 words.",
        f"Observe {profile['observation_cue']}.",
        "",
    ]

    if evaluation_steps_text:
        lines.extend([
            "Evaluation Steps (follow these while judging):",
            evaluation_steps_text,
            "",
        ])

    lines.extend([
        "Answer in this exact order:",
        "0. Briefly observe what happens in the video.",
        f"1. Who do you think made this {profile['subject_noun']}?",
        "2. How confident are you in your choice?",
        "3. What did you base your decision on? What signs or indicators guided your decision?",
        f"4. Rate your experience with the {profile['subject_noun']}.",
        "",
        "Step 0. Observation:",
        "- Write 1 or 2 short sentences describing what happens in the video.",
        "",
        "Step 1. Creator judgment:",
    ])
    for item in perception:
        options = ", ".join(str(option["value"]) for option in item["options"])
        lines.append(f"- {item['name']}: {item['survey_item']} (choose one of: {options})")

    lines.extend([
        "",
        f"Step 2. Written reason: {qualitative['name']} - {qualitative['survey_item']}",
        "",
        "Step 3. Experience ratings from 1 to 5 with short reasons:",
        f"- Empathize with the player in the video and imagine the feelings you would have while {profile['verb']}.",
        "- Use the full 1 to 5 scale when appropriate.",
    ])
    for item in experience:
        lines.append(f"- {item['name']}: {item['survey_item']} ({item['description']})")

    lines.extend([
        "",
        "Return JSON in this exact field order:",
        "{",
        '  "step0_observation": "",',
        '  "creator_belief": {"value": "", "reason": ""},',
        '  "confidence_level": {"value": null, "reason": ""},',
        '  "reasoning_for_creator_belief": "",',
        '  "enjoyment": {"score": null, "reason": ""},',
        '  "difficulty": {"score": null, "reason": ""},',
        '  "frustration": {"score": null, "reason": ""},',
        '  "novelty": {"score": null, "reason": ""},',
        '  "aesthetics": {"score": null, "reason": ""}',
        "}",
    ])
    return "\n".join(lines)


def _resolve_belief_mode(
    counts: dict[str, int],
    confidence_sums: dict[str, float],
) -> tuple[str, str]:
    """Pick the winning creator_belief with deterministic tiebreaking.

    Returns: (winner, tiebreaker_reason)
      tiebreaker_reason is one of: "majority", "confidence", "tie".
    """
    if not counts:
        return "", "majority"

    max_count = max(counts.values())
    top = [label for label, c in counts.items() if c == max_count]

    if len(top) == 1:
        return top[0], "majority"

    # Tie on vote count -> use summed confidence (higher total confidence wins).
    max_conf = max(confidence_sums.get(label, 0.0) for label in top)
    conf_top = [label for label in top if confidence_sums.get(label, 0.0) == max_conf]

    if len(conf_top) == 1:
        return conf_top[0], "confidence"

    # Still tied -> surface it explicitly so downstream analysis can exclude or handle.
    return "Tie", "tie"


def aggregate_judgments(judgments: list[dict]) -> dict | None:
    if not judgments:
        return None

    def to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    aggregated: dict = {"num_runs": len(judgments)}

    for key in ("step0_observation", "reasoning_for_creator_belief"):
        chosen = ""
        for judgment in judgments:
            candidate = str(judgment.get(key, "") or "").strip()
            if candidate:
                chosen = candidate
                break
        aggregated[key] = chosen

    belief_values: list[str] = []
    belief_confidence_sums: dict[str, float] = {}
    for judgment in judgments:
        belief = judgment.get("creator_belief")
        if not isinstance(belief, dict):
            continue
        value = str(belief.get("value") or "").strip()
        if not value:
            continue
        belief_values.append(value)

        conf_item = judgment.get("confidence_level")
        conf_val = to_float(conf_item.get("value")) if isinstance(conf_item, dict) else None
        belief_confidence_sums[value] = belief_confidence_sums.get(value, 0.0) + (conf_val or 0.0)

    belief_counts = dict(Counter(belief_values))
    belief_mode, belief_tiebreaker = _resolve_belief_mode(belief_counts, belief_confidence_sums)
    aggregated["creator_belief"] = {
        "value": belief_mode,
        "votes": belief_counts,
        "num_valid": len(belief_values),
        "confidence_sums": belief_confidence_sums,
        "tiebreaker": belief_tiebreaker,
    }

    confidence_values = []
    for judgment in judgments:
        item = judgment.get("confidence_level")
        if isinstance(item, dict):
            value = to_float(item.get("value"))
            if value is not None:
                confidence_values.append(value)
    if confidence_values:
        conf_mean = sum(confidence_values) / len(confidence_values)
        conf_std = statistics.pstdev(confidence_values) if len(confidence_values) > 1 else 0.0
    else:
        conf_mean = None
        conf_std = None
    aggregated["confidence_level"] = {
        "value": conf_mean,
        "std": conf_std,
        "num_valid": len(confidence_values),
        "values": confidence_values,
    }

    for metric in EXPERIENCE_METRICS:
        values = []
        for judgment in judgments:
            item = judgment.get(metric)
            if isinstance(item, dict):
                value = to_float(item.get("score"))
                if value is not None:
                    values.append(value)
        if values:
            mean = sum(values) / len(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
        else:
            mean = None
            std = None
        aggregated[metric] = {
            "score": mean,
            "std": std,
            "num_valid": len(values),
            "values": values,
        }

    return aggregated


def normalize_judgment_schema(judgment: dict) -> dict:
    if not isinstance(judgment, dict):
        raise ValueError("Judgment must be a JSON object.")

    normalized = dict(judgment)
    experience_block = judgment.get("experience_ratings", {})

    creator_value, creator_reason = normalize_value_and_reason(judgment.get("creator_belief"))
    confidence_value, confidence_reason = normalize_value_and_reason(judgment.get("confidence_level"))

    normalized["creator_belief"] = {"value": creator_value, "reason": creator_reason}
    normalized["confidence_level"] = {"value": confidence_value, "reason": confidence_reason}
    normalized["reasoning_for_creator_belief"] = str(judgment.get("reasoning_for_creator_belief", "") or "")
    normalized["step0_observation"] = str(judgment.get("step0_observation", "") or "")

    for metric in EXPERIENCE_METRICS:
        metric_value = judgment.get(metric)
        if metric_value is None and isinstance(experience_block, dict):
            metric_value = experience_block.get(metric)
        normalized[metric] = normalize_score_and_reason(metric_value)

    return normalized


def normalize_value_and_reason(value):
    if isinstance(value, dict):
        return value.get("value", None), str(value.get("reason", "") or "")
    return value, ""


def normalize_score_and_reason(value) -> dict:
    if isinstance(value, dict):
        return {
            "score": value.get("score", None),
            "reason": str(value.get("reason", "") or ""),
        }
    if value is None:
        return {"score": None, "reason": ""}
    return {"score": value, "reason": ""}


def encode_video_to_data_url(path: str) -> str:
    video_bytes = Path(path).read_bytes()
    encoded = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:video/mp4;base64,{encoded}"


def build_storage_payload(payload: dict, video_path: str) -> dict:
    stored = json.loads(json.dumps(payload))
    filename = Path(video_path).name

    for message in stored.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for index, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "video_url":
                content[index] = {"video": filename}

    return stored


def post_openrouter(payload: dict, api_key: str) -> tuple[str, dict]:
    request = urllib.request.Request(
        OPENROUTER_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw_text = response.read().decode("utf-8")
            return raw_text, json.loads(raw_text)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        trace = {
            "raw_response_text": error_body,
            "raw_response_json": safe_json_loads(error_body),
            "error": f"OpenRouter HTTP {exc.code}: {error_body}",
        }
        raise EvaluationTraceError(f"OpenRouter HTTP {exc.code}: {error_body}", trace) from exc


def parse_json_text(content) -> dict:
    text = normalize_content_text(content)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text.replace("json", "", 1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(text)
        if repaired is not None:
            return repaired
        raise


def _repair_truncated_json(text: str):
    """Attempt to self-heal JSON that is only missing trailing '}' or ']' brackets.

    Gemini 2.5 occasionally ends responses one closing brace short while reporting
    finish_reason='stop'. This helper balances the brackets based on what is still
    open after stripping the final trailing comma/whitespace, without touching the
    content — if any other error is present, json.loads will still fail.
    """
    stripped = text.rstrip().rstrip(",")
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in stripped:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                return None  # mismatched bracket — don't try to fix

    if in_string or not stack:
        return None

    candidate = stripped + "".join(reversed(stack))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def normalize_content_text(content) -> str:
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                chunks.append(str(part.get("text", "")))
            else:
                chunks.append(str(part))
        return "".join(chunks)
    return str(content)


def safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None
