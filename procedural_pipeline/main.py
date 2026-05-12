import argparse

from dotenv import load_dotenv

from procedural_pipeline.games import GAMES
from procedural_pipeline.runner import run


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
    parser.add_argument("--model", default="google/gemini-2.5-pro")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=4)

    # ---- Ablation / reuse options -------------------------------------- #
    parser.add_argument(
        "--blind",
        action="store_true",
        help=(
            "Blind mode (v2 ablation): drop creator-belief / confidence / reasoning "
            "from BOTH the prompt and the JSON output. Only the 5 experience metrics "
            "(+ a brief observation) are produced, removing the self-formed belief "
            "that drives the halo bias."
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
