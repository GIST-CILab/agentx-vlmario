import argparse

from dotenv import load_dotenv

from procedural_pipeline.execution.options import (
    add_evaluation_steps_prompt_arg,
    add_resume_arg,
    add_runtime_args,
    add_video_cache_arg,
)
from procedural_pipeline.execution.runner import run
from procedural_pipeline.games import GAMES
from procedural_pipeline.judge import EXPERIMENT_MODES


def main() -> None:
    args, game = parse_args()
    load_dotenv(override=True)
    run(game, args)


def parse_args() -> tuple[argparse.Namespace, object]:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--game",
        required=True,
        choices=sorted(GAMES.keys()),
        help="Which game module to evaluate (mario / sokoban).",
    )
    preliminary, _ = parent.parse_known_args()
    game = GAMES[preliminary.game]
    defaults = game.DEFAULTS

    parser = argparse.ArgumentParser(
        description="Unified procedural PCG video evaluation pipeline (G-Eval Auto-CoT, N-run average).",
        parents=[parent],
    )
    parser.add_argument("--maps-file", default=defaults["maps_file"])
    parser.add_argument("--criteria-file", default=defaults["criteria_file"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    add_video_cache_arg(parser)
    add_runtime_args(parser)
    add_evaluation_steps_prompt_arg(parser)
    add_resume_arg(parser)
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENT_MODES,
        default=None,
        help=(
            "Evaluation condition: creator_judgment asks creator/confidence; "
            "no_creator asks only the five experience axes; forced_creator tells "
            "the model a creator label and asks the five axes."
        ),
    )
    parser.add_argument(
        "--forced-creator",
        choices=["AI", "Human"],
        default=None,
        help="Creator label supplied to the model when --experiment forced_creator.",
    )

    # ---- Ablation / reuse options -------------------------------------- #
    parser.add_argument(
        "--blind",
        action="store_true",
        help=(
            "Backward-compatible shortcut for --experiment no_creator."
        ),
    )
    parser.add_argument(
        "--eval-steps-file",
        default=None,
        help=(
            "Reuse a previously-generated evaluation_steps_*.json instead of "
            "calling the LLM to generate Auto-CoT steps again."
        ),
    )
    parser.add_argument(
        "--video-source-dir",
        default=None,
        help=(
            "Fallback directory to reuse rendered .mp4 (and .txt) files from. "
            "Used when a map's video is missing in --output-dir but already exists "
            "in this source dir (e.g. results/v1.3/procedural_mario)."
        ),
    )
    parser.add_argument(
        "--play-only",
        action="store_true",
        help=(
            "Skip Auto-CoT step generation and LLM video evaluation; only render "
            "playthrough videos (and save per-map .txt) under --output-dir. "
            "Ignores --model, --criteria-file, --num-runs, etc."
        ),
    )

    game.add_arguments(parser)

    return parser.parse_args(), game


if __name__ == "__main__":
    main()
