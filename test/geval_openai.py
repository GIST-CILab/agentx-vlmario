import argparse
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


def load_text(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def debug_log(enabled: bool, title: str, data=None) -> None:
    if not enabled:
        return
    print(f"[DEBUG] {title}")
    if data is not None:
        if isinstance(data, (dict, list)):
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(str(data))
    print()


def supports_log_probs(model_name: str) -> bool:
    model = model_name.lower()
    supported_prefixes = (
        "gpt-4.1",
        "gpt-4o",
        "gpt-5",
        "o3",
        "o4",
    )
    return model.startswith(supported_prefixes)


def check_if_multimodal(prompt: str) -> bool:
    return bool(re.search(r"https?://\\S+\\.(png|jpg|jpeg|webp|gif)", prompt, re.IGNORECASE))


def convert_to_multi_modal_array(input_text: str) -> list[dict[str, Any]]:
    # Very simple parser:
    # - URLs ending in image extensions become image_url blocks
    # - All other text chunks remain text blocks
    parts: list[dict[str, Any]] = []
    tokens = input_text.split()
    text_buffer: list[str] = []
    for token in tokens:
        if re.match(r"^https?://\\S+\\.(png|jpg|jpeg|webp|gif)$", token, re.IGNORECASE):
            if text_buffer:
                parts.append({"type": "text", "text": " ".join(text_buffer)})
                text_buffer = []
            parts.append({"type": "image_url", "image_url": {"url": token}})
        else:
            text_buffer.append(token)
    if text_buffer:
        parts.append({"type": "text", "text": " ".join(text_buffer)})
    return parts if parts else [{"type": "text", "text": input_text}]


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    # Placeholder estimate; exact per-model pricing differs.
    # Keep transparent and conservative for debugging purposes.
    return round((input_tokens * 0.000001) + (output_tokens * 0.000003), 8)


def create_completion(
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    top_logprobs: int,
    debug: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "max_tokens": 1,
    }
    debug_log(debug, "openai payload", payload)
    request = urllib.request.Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            parsed = json.loads(response.read().decode("utf-8"))
            debug_log(debug, "openai raw response", parsed)
            return parsed
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        debug_log(debug, "openai http error body", error_body)
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {error_body}") from exc


def extract_distribution(completion: dict[str, Any]) -> tuple[dict[int, float], Any]:
    choices = completion.get("choices") or []
    if not choices:
        return {}, None
    logprobs = choices[0].get("logprobs") or {}
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
        token = str(candidate.get("token", "")).strip()
        if token not in {"1", "2", "3", "4", "5"}:
            continue
        logprob = candidate.get("logprob")
        if logprob is None:
            continue
        score = int(token)
        best_per_score[score] = max(logprob, best_per_score.get(score, float("-inf")))

    if not best_per_score:
        return {}, first_token

    max_logprob = max(best_per_score.values())
    unnormalized = {score: math.exp(lp - max_logprob) for score, lp in best_per_score.items()}
    denom = sum(unnormalized.values())
    distribution = {score: value / denom for score, value in sorted(unnormalized.items())}
    return distribution, first_token


def calculate_final_weighted_score(distribution: dict[int, float]) -> float | None:
    if not distribution:
        return None
    # Final score is computed only from numeric tokens 1..5.
    return sum(score * prob for score, prob in distribution.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="G-EVAL OpenAI raw response/logprobs tester.")
    parser.add_argument("--prompt-file", help="Prompt file for direct raw-response call")
    parser.add_argument("--source-file", help="Optional source text for canned G-EVAL prompt")
    parser.add_argument("--candidate-file", help="Optional candidate text for canned G-EVAL prompt")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-logprobs", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_KEY or OPENAI_API_KEY is missing.")

    if not supports_log_probs(args.model):
        raise RuntimeError(
            f"Model `{args.model}` does not support `logprobs` / `top_logprobs` in this tester. "
            "Use an OpenAI model like `gpt-4.1` or `gpt-4o`."
        )

    if args.prompt_file:
        prompt = load_text(args.prompt_file)
    elif args.source_file and args.candidate_file:
        source_text = load_text(args.source_file)
        candidate_text = load_text(args.candidate_file)
        prompt = (
            "You will be given one source text and one candidate text.\n"
            "Rate coherence from 1 to 5.\n\n"
            f"Source Text:\n{source_text}\n\n"
            f"Candidate Text:\n{candidate_text}\n\n"
            "Return only one token: 1, 2, 3, 4, or 5."
        )
    else:
        raise RuntimeError("Provide either --prompt-file or both --source-file and --candidate-file.")

    is_multimodal = check_if_multimodal(prompt)
    if is_multimodal:
        content = convert_to_multi_modal_array(prompt)
    else:
        content = [{"type": "text", "text": prompt}]

    completion = create_completion(
        api_key=api_key,
        model=args.model,
        messages=[{"role": "user", "content": content}],
        temperature=args.temperature,
        top_logprobs=args.top_logprobs,
        debug=args.debug,
    )

    usage = completion.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cost = calculate_cost(input_tokens, output_tokens)
    distribution, first_token = extract_distribution(completion)
    weighted_score = calculate_final_weighted_score(distribution)

    result = {
        "model": completion.get("model", args.model),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": cost,
        "supports_probability": bool(distribution),
        "weighted_score": weighted_score,
        "final_weighted_score": weighted_score,
        "score_distribution": distribution,
        "raw_first_token_logprobs": first_token,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
