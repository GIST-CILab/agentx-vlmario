import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def debug_log(enabled: bool, title: str, data=None) -> None:
    if not enabled:
        return
    print(f"[DEBUG] {title}", file=sys.stderr)
    if data is not None:
        if isinstance(data, (dict, list)):
            print(json.dumps(data, indent=2, ensure_ascii=False), file=sys.stderr)
        else:
            print(str(data), file=sys.stderr)
    print(file=sys.stderr)


def load_text(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def post_openrouter(payload: dict, api_key: str, debug: bool = False, label: str = "request") -> dict:
    debug_log(
        debug,
        f"{label}: payload",
        {
            **payload,
            "messages": payload.get("messages", []),
        },
    )
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            parsed = json.loads(response.read().decode("utf-8"))
            debug_log(debug, f"{label}: raw response", parsed)
            return parsed
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        debug_log(debug, f"{label}: http error body", error_body)
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {error_body}") from exc


def generate_cot_steps(
    model: str,
    source_text: str,
    candidate_text: str,
    criterion_name: str,
    criterion_desc: str,
    api_key: str,
    debug: bool = False,
) -> tuple[str, dict]:
    prompt = f"""You are designing evaluation steps for G-EVAL.

Task: Evaluate a candidate text using one metric.

Evaluation Criteria:
{criterion_name} (1-5) - {criterion_desc}

Source Text:
{source_text}

Candidate Text:
{candidate_text}

Write 3 short numbered evaluation steps that help an evaluator judge this metric.
Return only the numbered steps."""

    response = post_openrouter(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 300,
            "stream": False,
        },
        api_key,
        debug=debug,
        label="cot_generation",
    )
    content = response["choices"][0]["message"]["content"]
    steps = content.strip() if isinstance(content, str) else str(content).strip()
    return steps, response


def build_score_prompt(source_text: str, candidate_text: str, criterion_name: str, criterion_desc: str, evaluation_steps: str) -> str:
    return f"""You will be given one source text and one candidate text. Your task is to rate the candidate on one metric.

Please read and understand these instructions carefully.

Evaluation Criteria:
{criterion_name} (1-5) - {criterion_desc}

Evaluation Steps:
{evaluation_steps}

Source Text:
{source_text}

Candidate Text:
{candidate_text}

Return only one token: 1, 2, 3, 4, or 5."""


def normalize_score_token(token: str) -> int | None:
    stripped = token.strip()
    if stripped in {"1", "2", "3", "4", "5"}:
        return int(stripped)
    return None


def extract_message_text(choice: dict) -> tuple[str, str]:
    message = choice.get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content.strip(), "string"

    if isinstance(content, list):
        text_chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_chunks.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    text_chunks.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    text_chunks.append(item["content"])
        return "\n".join(chunk.strip() for chunk in text_chunks if chunk and chunk.strip()), "list"

    if content is None:
        return "", "none"

    return str(content).strip(), type(content).__name__


def extract_first_token_distribution(choice: dict) -> tuple[dict[int, float], dict | None]:
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") or []
    if not content:
        return {}, None

    first_token = content[0]
    candidates = []
    if first_token.get("token") is not None and first_token.get("logprob") is not None:
        candidates.append(
            {
                "token": first_token["token"],
                "logprob": first_token["logprob"],
            }
        )
    candidates.extend(first_token.get("top_logprobs") or [])

    best_per_score: dict[int, float] = {}
    for candidate in candidates:
        score = normalize_score_token(str(candidate.get("token", "")))
        logprob = candidate.get("logprob")
        if score is None or logprob is None:
            continue
        best_per_score[score] = max(logprob, best_per_score.get(score, float("-inf")))

    if not best_per_score:
        return {}, first_token

    max_logprob = max(best_per_score.values())
    unnormalized = {score: math.exp(lp - max_logprob) for score, lp in best_per_score.items()}
    denom = sum(unnormalized.values())
    distribution = {score: value / denom for score, value in sorted(unnormalized.items())}
    return distribution, first_token


def summarize_choice(choice: dict) -> dict:
    message = choice.get("message") or {}
    logprobs = choice.get("logprobs") or {}
    return {
        "keys": sorted(choice.keys()),
        "finish_reason": choice.get("finish_reason"),
        "message_keys": sorted(message.keys()) if isinstance(message, dict) else [],
        "message_content_type": type(message.get("content")).__name__ if isinstance(message, dict) else type(message).__name__,
        "has_logprobs": bool(logprobs),
        "logprobs_keys": sorted(logprobs.keys()) if isinstance(logprobs, dict) else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test G-EVAL weighted scoring with OpenRouter logprobs.")
    parser.add_argument("--source-file", required=True, help="Path to the source/reference text")
    parser.add_argument("--candidate-file", required=True, help="Path to the candidate text to evaluate")
    parser.add_argument("--criterion-name", default="Coherence")
    parser.add_argument(
        "--criterion-description",
        default="The candidate should be well-structured, well-organized, and should build from sentence to sentence into a coherent whole.",
    )
    parser.add_argument("--model", default="google/gemini-2.5-pro")
    parser.add_argument("--steps-file", help="Optional file containing pre-written evaluation steps")
    parser.add_argument("--skip-cot", action="store_true", help="Skip automatic CoT generation and use default steps")
    parser.add_argument("--debug", action="store_true", help="Print verbose request/response logs to stderr")
    args = parser.parse_args()

    load_dotenv(override=True)
    api_key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPEN_ROUTER_API_KEY is missing.")

    source_text = load_text(args.source_file)
    candidate_text = load_text(args.candidate_file)

    if args.steps_file:
        evaluation_steps = load_text(args.steps_file).strip()
        cot_response = None
    elif args.skip_cot:
        evaluation_steps = "\n".join(
            [
                "1. Read the source text and identify its main topic and key points.",
                "2. Read the candidate text and compare its structure and ordering to the source text.",
                "3. Output one score from 1 to 5 based on the evaluation criteria.",
            ]
        )
        cot_response = None
    else:
        evaluation_steps, cot_response = generate_cot_steps(
            model=args.model,
            source_text=source_text,
            candidate_text=candidate_text,
            criterion_name=args.criterion_name,
            criterion_desc=args.criterion_description,
            api_key=api_key,
            debug=args.debug,
        )

    prompt = build_score_prompt(
        source_text=source_text,
        candidate_text=candidate_text,
        criterion_name=args.criterion_name,
        criterion_desc=args.criterion_description,
        evaluation_steps=evaluation_steps,
    )

    response = post_openrouter(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 20,
        },
        api_key,
        debug=args.debug,
        label="score_request",
    )

    choice = response["choices"][0]
    output_text, output_content_type = extract_message_text(choice)
    distribution, first_token = extract_first_token_distribution(choice)
    weighted_score = None
    if distribution:
        weighted_score = sum(score * prob for score, prob in distribution.items())
    elif first_token and isinstance(first_token.get("token"), str) and not output_text:
        output_text = first_token["token"].strip()

    debug_log(args.debug, "source_text", source_text)
    debug_log(args.debug, "candidate_text", candidate_text)
    debug_log(args.debug, "evaluation_steps", evaluation_steps)
    if cot_response is not None:
        debug_log(args.debug, "cot_generation choice summary", summarize_choice(cot_response["choices"][0]))
    debug_log(args.debug, "score_request choice summary", summarize_choice(choice))
    debug_log(args.debug, "score_request message", choice.get("message"))
    debug_log(args.debug, "score_request logprobs", choice.get("logprobs"))
    debug_log(args.debug, "parsed first_token", first_token)
    debug_log(args.debug, "parsed score_distribution", distribution)
    debug_log(
        args.debug,
        "parsed output",
        {
            "raw_output": output_text,
            "raw_output_content_type": output_content_type,
            "supports_probability": bool(distribution),
            "weighted_score": weighted_score,
        },
    )

    result = {
        "model": response.get("model", args.model),
        "criterion": {
            "name": args.criterion_name,
            "description": args.criterion_description,
        },
        "evaluation_steps": evaluation_steps,
        "raw_output": output_text,
        "raw_output_content_type": output_content_type,
        "supports_probability": bool(distribution),
        "weighted_score": weighted_score,
        "score_distribution": distribution,
        "observed_scores": sorted(distribution.keys()),
        "missing_scores": [score for score in range(1, 6) if score not in distribution],
        "raw_first_token_logprobs": first_token,
    }

    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
