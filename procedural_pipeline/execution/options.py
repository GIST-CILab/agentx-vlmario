from argparse import ArgumentParser, Namespace

from procedural_pipeline.paths import project_path


DEFAULT_MODEL = "google/gemini-2.5-pro"
DEFAULT_LIMIT = 30
DEFAULT_NUM_RUNS = 20
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_CONCURRENCY = 4
DEFAULT_REQUEST_TIMEOUT = 180.0


def default_video_cache_dir() -> str:
    return str(project_path("outputs", "video"))


def add_video_cache_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--video-cache-dir",
        default=default_video_cache_dir(),
        help="Shared rendered-video cache root. Videos are stored as <root>/<game>/<map_id>.mp4.",
    )


def add_runtime_args(parser: ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Seconds to wait for each OpenRouter request before treating that run as failed.",
    )


def add_evaluation_steps_prompt_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--no-evaluation-steps-in-prompt",
        action="store_false",
        dest="evaluation_steps_in_prompt",
        default=True,
        help="Generate/save Auto-CoT evaluation steps, but do not include them in per-video prompts.",
    )


def add_resume_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Reuse completed raw/<map_id>.json results in the output directory (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing raw results and evaluate every selected map again.",
    )


def runtime_options(args: Namespace) -> dict:
    return {
        "model": args.model,
        "limit": args.limit,
        "num_runs": args.num_runs,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "concurrency": args.concurrency,
        "request_timeout": args.request_timeout,
    }
