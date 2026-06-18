import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


VERTEX_API_BASE = "https://aiplatform.googleapis.com/v1/publishers/google/models"


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


def error_result(stage: str, model: str, exc: Exception, debug: bool = False) -> dict[str, Any]:
    payload = {
        "stage": stage,
        "model": model,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    debug_log(debug, f"{stage}: error", payload)
    return payload


def post_vertex_api_key(
    model: str,
    api_key: str,
    body: dict[str, Any],
    debug: bool = False,
    label: str = "request",
) -> dict[str, Any]:
    url = f"{VERTEX_API_BASE}/{model}:generateContent?key={urllib.parse.quote(api_key)}"
    debug_log(debug, f"{label}: url", url.replace(api_key, "***"))
    debug_log(debug, f"{label}: payload", body)

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
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
        raise RuntimeError(f"Vertex API HTTP {exc.code}: {error_body}") from exc


def generate_cot_steps(
    model: str,
    api_key: str,
    source_text: str,
    candidate_text: str,
    criterion_name: str,
    criterion_desc: str,
    debug: bool = False,
) -> tuple[str, dict[str, Any]]:
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

    response = post_vertex_api_key(
        model=model,
        api_key=api_key,
        body={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 300,
                "responseMimeType": "text/plain",
            },
        },
        debug=debug,
        label="cot_generation",
    )

    steps = ""
    try:
        steps = response["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        steps = ""
    return steps, response


def build_score_prompt(
    source_text: str,
    candidate_text: str,
    criterion_name: str,
    criterion_desc: str,
    evaluation_steps: str,
) -> str:
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


def extract_response_text(response: dict[str, Any]) -> str:
    try:
        return response["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return ""


def extract_first_token_distribution(candidate: dict[str, Any]) -> tuple[dict[int, float], Any]:
    logprobs_result = candidate.get("logprobsResult") or {}
    top_candidates = logprobs_result.get("topCandidates") or []
    if not top_candidates:
        return {}, None

    first_step = top_candidates[0]
    candidates = first_step.get("candidates") or []
    best_per_score: dict[int, float] = {}
    for item in candidates:
        score = normalize_score_token(str(item.get("token", "")))
        log_probability = item.get("logProbability")
        if score is None or log_probability is None:
            continue
        best_per_score[score] = max(log_probability, best_per_score.get(score, float("-inf")))

    if not best_per_score:
        return {}, first_step

    max_logprob = max(best_per_score.values())
    unnormalized = {score: math.exp(lp - max_logprob) for score, lp in best_per_score.items()}
    denom = sum(unnormalized.values())
    distribution = {score: value / denom for score, value in sorted(unnormalized.items())}
    return distribution, first_step


def summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    logprobs_result = candidate.get("logprobsResult") or {}
    return {
        "finishReason": candidate.get("finishReason"),
        "finishMessage": candidate.get("finishMessage"),
        "tokenCount": candidate.get("tokenCount"),
        "avgLogprobs": candidate.get("avgLogprobs"),
        "partCount": len(parts),
        "hasLogprobsResult": bool(logprobs_result),
        "topCandidatesSteps": len(logprobs_result.get("topCandidates") or []),
        "chosenCandidatesSteps": len(logprobs_result.get("chosenCandidates") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test G-EVAL weighted scoring with Vertex API key.")
    parser.add_argument("--source-file", required=True, help="Path to the source/reference text")
    parser.add_argument("--candidate-file", required=True, help="Path to the candidate text to evaluate")
    parser.add_argument("--criterion-name", default="Coherence")
    parser.add_argument(
        "--criterion-description",
        default="The candidate should be well-structured, well-organized, and should build from sentence to sentence into a coherent whole.",
    )
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--steps-file", help="Optional file containing pre-written evaluation steps")
    parser.add_argument("--skip-cot", action="store_true", help="Skip automatic CoT generation and use default steps")
    parser.add_argument("--debug", action="store_true", help="Print verbose request/response logs to stderr")
    args = parser.parse_args()

    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    if not api_key:
        raise RuntimeError("GOOGLE_CLOUD_API_KEY is missing.")

    source_text = load_text(args.source_file)
    candidate_text = load_text(args.candidate_file)

    if args.steps_file:
        evaluation_steps = load_text(args.steps_file).strip()
        cot_response = None
        cot_error = None
    elif args.skip_cot:
        evaluation_steps = "\n".join(
            [
                "1. Read the source text and identify its main topic and key points.",
                "2. Read the candidate text and compare its structure and ordering to the source text.",
                "3. Output one score from 1 to 5 based on the evaluation criteria.",
            ]
        )
        cot_response = None
        cot_error = None
    else:
        try:
            evaluation_steps, cot_response = generate_cot_steps(
                model=args.model,
                api_key=api_key,
                source_text=source_text,
                candidate_text=candidate_text,
                criterion_name=args.criterion_name,
                criterion_desc=args.criterion_description,
                debug=args.debug,
            )
            cot_error = None
        except Exception as exc:
            cot_response = None
            cot_error = error_result("cot_generation", args.model, exc, debug=args.debug)
            evaluation_steps = "\n".join(
                [
                    "1. Read the source text and identify its main topic and key points.",
                    "2. Read the candidate text and compare its structure and ordering to the source text.",
                    "3. Output one score from 1 to 5 based on the evaluation criteria.",
                ]
            )

    prompt = build_score_prompt(
        source_text=source_text,
        candidate_text=candidate_text,
        criterion_name=args.criterion_name,
        criterion_desc=args.criterion_description,
        evaluation_steps=evaluation_steps,
    )

    try:
        response = post_vertex_api_key(
            model=args.model,
            api_key=api_key,
            body={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 1,
                    "responseMimeType": "text/plain",
                    "responseLogprobs": True,
                    "logprobs": 20,
                },
            },
            debug=args.debug,
            label="score_request",
        )
    except Exception as exc:
        result = {
            "backend": "vertex_api_key",
            "project_id": project_id,
            "model": args.model,
            "criterion": {
                "name": args.criterion_name,
                "description": args.criterion_description,
            },
            "evaluation_steps": evaluation_steps,
            "supports_probability": False,
            "weighted_score": None,
            "score_distribution": {},
            "observed_scores": [],
            "missing_scores": [1, 2, 3, 4, 5],
            "avg_logprobs": None,
            "logprobs_result": None,
            "usage_metadata": None,
            "candidate_summary": None,
            "cot_error": cot_error,
            "score_error": error_result("score_request", args.model, exc, debug=args.debug),
        }
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return

    candidate = (response.get("candidates") or [{}])[0]
    output_text = extract_response_text(response)
    distribution, first_step = extract_first_token_distribution(candidate)
    weighted_score = None
    if distribution:
        weighted_score = sum(score * prob for score, prob in distribution.items())

    debug_log(args.debug, "project_id", project_id)
    debug_log(args.debug, "source_text", source_text)
    debug_log(args.debug, "candidate_text", candidate_text)
    debug_log(args.debug, "evaluation_steps", evaluation_steps)
    if cot_response is not None:
        debug_log(args.debug, "cot_generation candidate summary", summarize_candidate((cot_response.get("candidates") or [{}])[0]))
    debug_log(args.debug, "score_request candidate summary", summarize_candidate(candidate))
    debug_log(args.debug, "score_request usage_metadata", response.get("usageMetadata"))
    debug_log(args.debug, "score_request candidate", candidate)
    debug_log(args.debug, "parsed first_step", first_step)
    debug_log(args.debug, "parsed score_distribution", distribution)
    debug_log(
        args.debug,
        "parsed output",
        {
            "raw_output": output_text,
            "supports_probability": bool(distribution),
            "weighted_score": weighted_score,
            "avg_logprobs": candidate.get("avgLogprobs"),
        },
    )

    result = {
        "backend": "vertex_api_key",
        "project_id": project_id,
        "model": response.get("modelVersion", args.model),
        "criterion": {
            "name": args.criterion_name,
            "description": args.criterion_description,
        },
        "evaluation_steps": evaluation_steps,
        "raw_output": output_text,
        "supports_probability": bool(distribution),
        "weighted_score": weighted_score,
        "score_distribution": distribution,
        "observed_scores": sorted(distribution.keys()),
        "missing_scores": [score for score in range(1, 6) if score not in distribution],
        "avg_logprobs": candidate.get("avgLogprobs"),
        "logprobs_result": candidate.get("logprobsResult"),
        "usage_metadata": response.get("usageMetadata"),
        "candidate_summary": summarize_candidate(candidate),
        "cot_error": cot_error,
        "score_error": None,
    }

    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
