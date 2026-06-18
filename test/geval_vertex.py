import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig


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


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(exclude_none=False))
        except Exception:
            pass
    if hasattr(value, "to_json_dict"):
        try:
            return to_jsonable(value.to_json_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return to_jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def error_result(stage: str, model: str, exc: Exception, debug: bool = False) -> dict[str, Any]:
    payload = {
        "stage": stage,
        "model": model,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    debug_log(debug, f"{stage}: error", payload)
    return payload


def create_vertex_client(project: str, location: str):
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )


def generate_content(
    client: genai.Client,
    model: str,
    contents: str,
    config: GenerateContentConfig,
    debug: bool = False,
    label: str = "request",
):
    debug_log(
        debug,
        f"{label}: payload",
        {
            "model": model,
            "contents": contents,
            "config": to_jsonable(config),
        },
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    debug_log(debug, f"{label}: raw response", to_jsonable(response))
    return response


def generate_cot_steps(
    client: genai.Client,
    model: str,
    source_text: str,
    candidate_text: str,
    criterion_name: str,
    criterion_desc: str,
    debug: bool = False,
):
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

    response = generate_content(
        client=client,
        model=model,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=300,
            response_mime_type="text/plain",
        ),
        debug=debug,
        label="cot_generation",
    )
    steps = (response.text or "").strip()
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


def extract_response_text(response) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.strip()
    return ""


def extract_first_token_distribution(candidate) -> tuple[dict[int, float], Any]:
    logprobs_result = getattr(candidate, "logprobs_result", None)
    if not logprobs_result:
        return {}, None

    top_candidates = getattr(logprobs_result, "top_candidates", None) or []
    if not top_candidates:
        return {}, None

    first_step = top_candidates[0]
    candidates = getattr(first_step, "candidates", None) or []
    best_per_score: dict[int, float] = {}

    for candidate_item in candidates:
        token = getattr(candidate_item, "token", "")
        log_probability = getattr(candidate_item, "log_probability", None)
        score = normalize_score_token(str(token))
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


def summarize_candidate(candidate) -> dict[str, Any]:
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content else None
    logprobs_result = getattr(candidate, "logprobs_result", None)
    top_candidates = getattr(logprobs_result, "top_candidates", None) if logprobs_result else None
    chosen_candidates = getattr(logprobs_result, "chosen_candidates", None) if logprobs_result else None
    return {
        "finish_reason": str(getattr(candidate, "finish_reason", None)),
        "finish_message": getattr(candidate, "finish_message", None),
        "token_count": getattr(candidate, "token_count", None),
        "avg_logprobs": getattr(candidate, "avg_logprobs", None),
        "has_content": content is not None,
        "part_count": len(parts) if parts else 0,
        "has_logprobs_result": logprobs_result is not None,
        "top_candidates_steps": len(top_candidates) if top_candidates else 0,
        "chosen_candidates_steps": len(chosen_candidates) if chosen_candidates else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test G-EVAL weighted scoring with Gemini on Vertex AI.")
    parser.add_argument("--source-file", required=True, help="Path to the source/reference text")
    parser.add_argument("--candidate-file", required=True, help="Path to the candidate text to evaluate")
    parser.add_argument("--criterion-name", default="Coherence")
    parser.add_argument(
        "--criterion-description",
        default="The candidate should be well-structured, well-organized, and should build from sentence to sentence into a coherent whole.",
    )
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--project", help="Google Cloud project ID")
    parser.add_argument("--location", help="Google Cloud location, e.g. us-central1")
    parser.add_argument("--steps-file", help="Optional file containing pre-written evaluation steps")
    parser.add_argument("--skip-cot", action="store_true", help="Skip automatic CoT generation and use default steps")
    parser.add_argument("--debug", action="store_true", help="Print verbose request/response logs to stderr")
    args = parser.parse_args()

    load_dotenv(override=True)

    project = args.project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT")
    location = args.location or os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("LOCATION")

    if not project or not location:
        raise RuntimeError(
            "Vertex AI requires project and location. Set --project/--location or env GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION."
        )

    source_text = load_text(args.source_file)
    candidate_text = load_text(args.candidate_file)

    client = create_vertex_client(project=project, location=location)

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
                client=client,
                model=args.model,
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
        response = generate_content(
            client=client,
            model=args.model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1,
                response_mime_type="text/plain",
                response_logprobs=True,
                logprobs=20,
            ),
            debug=args.debug,
            label="score_request",
        )
    except Exception as exc:
        result = {
            "backend": "vertexai",
            "project": project,
            "location": location,
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
            "prompt_feedback": None,
            "candidate_summary": None,
            "cot_error": cot_error,
            "score_error": error_result("score_request", args.model, exc, debug=args.debug),
        }
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return

    candidate = response.candidates[0] if getattr(response, "candidates", None) else None
    if candidate is None:
        raise RuntimeError("No candidates returned by Vertex AI.")

    output_text = extract_response_text(response)
    distribution, first_step = extract_first_token_distribution(candidate)
    weighted_score = None
    if distribution:
        weighted_score = sum(score * prob for score, prob in distribution.items())

    debug_log(args.debug, "vertex project/location", {"project": project, "location": location})
    debug_log(args.debug, "source_text", source_text)
    debug_log(args.debug, "candidate_text", candidate_text)
    debug_log(args.debug, "evaluation_steps", evaluation_steps)
    if cot_response is not None and getattr(cot_response, "candidates", None):
        debug_log(args.debug, "cot_generation candidate summary", summarize_candidate(cot_response.candidates[0]))
    debug_log(args.debug, "score_request candidate summary", summarize_candidate(candidate))
    debug_log(args.debug, "score_request usage_metadata", to_jsonable(getattr(response, "usage_metadata", None)))
    debug_log(args.debug, "score_request prompt_feedback", to_jsonable(getattr(response, "prompt_feedback", None)))
    debug_log(args.debug, "score_request candidate content", to_jsonable(getattr(candidate, "content", None)))
    debug_log(args.debug, "score_request logprobs_result", to_jsonable(getattr(candidate, "logprobs_result", None)))
    debug_log(args.debug, "parsed first_step", to_jsonable(first_step))
    debug_log(args.debug, "parsed score_distribution", distribution)
    debug_log(
        args.debug,
        "parsed output",
        {
            "raw_output": output_text,
            "supports_probability": bool(distribution),
            "weighted_score": weighted_score,
            "avg_logprobs": getattr(candidate, "avg_logprobs", None),
        },
    )

    result = {
        "backend": "vertexai",
        "project": project,
        "location": location,
        "model": getattr(response, "model_version", None) or args.model,
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
        "avg_logprobs": getattr(candidate, "avg_logprobs", None),
        "logprobs_result": to_jsonable(getattr(candidate, "logprobs_result", None)),
        "usage_metadata": to_jsonable(getattr(response, "usage_metadata", None)),
        "prompt_feedback": to_jsonable(getattr(response, "prompt_feedback", None)),
        "candidate_summary": summarize_candidate(candidate),
        "cot_error": cot_error,
        "score_error": None,
    }

    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
