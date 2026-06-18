import argparse
from pathlib import Path

from dotenv import load_dotenv

from procedural_pipeline.execution.options import (
    add_evaluation_steps_prompt_arg,
    add_resume_arg,
    add_runtime_args,
    add_video_cache_arg,
    runtime_options,
)
from procedural_pipeline.execution.runner import run
from procedural_pipeline.games import GAMES
from procedural_pipeline.judge import (
    EXPERIMENT_CREATOR_JUDGMENT,
    EXPERIMENT_FORCED_CREATOR,
    EXPERIMENT_NO_CREATOR,
    generate_evaluation_steps,
    load_criteria,
    resolve_game_profile,
)
from procedural_pipeline.paths import project_path


EXPERIMENT_PLANS = {
    "1": ("01_creator_judgment", EXPERIMENT_CREATOR_JUDGMENT, [None]),
    "creator_judgment": ("01_creator_judgment", EXPERIMENT_CREATOR_JUDGMENT, [None]),
    "2": ("02_no_creator", EXPERIMENT_NO_CREATOR, [None]),
    "no_creator": ("02_no_creator", EXPERIMENT_NO_CREATOR, [None]),
    "3": ("03_forced_creator", EXPERIMENT_FORCED_CREATOR, ["AI", "Human"]),
    "forced_creator": ("03_forced_creator", EXPERIMENT_FORCED_CREATOR, ["AI", "Human"]),
}
ORDERED_PLANS = [
    EXPERIMENT_PLANS["1"],
    EXPERIMENT_PLANS["2"],
    EXPERIMENT_PLANS["3"],
]


def main() -> None:
    args = parse_args()
    load_dotenv(override=True)

    games = sorted(GAMES.keys()) if args.games == "both" else [args.games]
    plans = ORDERED_PLANS if args.experiment == "all" else [EXPERIMENT_PLANS[args.experiment]]
    shared_steps = build_shared_evaluation_steps(games, args) if args.reuse_cot else {}

    for plan_name, experiment, forced_labels in plans:
        print("=" * 72)
        print(f"[experiments] start {plan_name} ({experiment})")
        print("=" * 72)
        for game_key in games:
            game = GAMES[game_key]
            evaluation_steps = shared_steps.get(game_key)
            if evaluation_steps is None:
                evaluation_steps = generate_game_evaluation_steps(
                    game_key,
                    context=f"{plan_name}:{game_key}",
                    model=args.model,
                )
            else:
                print(f"[experiments:{plan_name}:{game_key}] reusing shared Auto-CoT for this game")

            for forced_creator in forced_labels:
                output_dir = Path(args.output_root) / plan_name / game_key
                if forced_creator:
                    output_dir = output_dir / forced_creator.lower()

                run_args = build_run_args(
                    args=args,
                    game=game,
                    experiment=experiment,
                    forced_creator=forced_creator,
                    output_dir=output_dir,
                    evaluation_steps=evaluation_steps,
                )
                run(game, run_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the three procedural-pipeline experiment conditions across Mario/Sokoban. "
            "Experiment 3 generates one Auto-CoT per game, then reuses it for AI/Human forced labels."
        )
    )
    parser.add_argument(
        "--experiment",
        choices=["all", *EXPERIMENT_PLANS.keys()],
        default="all",
        help="Which experiment condition to run. Default runs all three.",
    )
    parser.add_argument("--games", choices=["both", *sorted(GAMES.keys())], default="both")
    parser.add_argument("--output-root", default=str(project_path("outputs", "procedural_experiments")))
    add_video_cache_arg(parser)
    add_runtime_args(parser)
    add_evaluation_steps_prompt_arg(parser)
    add_resume_arg(parser)
    parser.add_argument(
        "--reuse-cot",
        "--reuse-evaluation-steps",
        dest="reuse_cot",
        action="store_true",
        default=True,
        help="Reuse one generated Auto-CoT per game across selected experiments (default).",
    )
    parser.add_argument(
        "--no-reuse-cot",
        "--no-reuse-evaluation-steps",
        dest="reuse_cot",
        action="store_false",
        help="Generate a fresh Auto-CoT for each game within each experiment condition.",
    )
    parser.add_argument("--video-source-dir", default=None)
    parser.add_argument("--solver-power", type=int, default=None)
    return parser.parse_args()


def build_shared_evaluation_steps(games: list[str], args: argparse.Namespace) -> dict[str, dict]:
    shared_steps = {}
    for game_key in games:
        shared_steps[game_key] = generate_game_evaluation_steps(
            game_key,
            context=f"shared:{game_key}",
            model=args.model,
        )
    return shared_steps


def generate_game_evaluation_steps(game_key: str, *, context: str, model: str) -> dict:
    game = GAMES[game_key]
    criteria = load_criteria(game.DEFAULTS["criteria_file"])
    profile = resolve_game_profile(game.DEFAULTS["profile_key"])
    print(f"[experiments:{context}] generating Auto-CoT")
    return generate_evaluation_steps(criteria, profile, model)


def build_run_args(
    *,
    args: argparse.Namespace,
    game,
    experiment: str,
    forced_creator: str | None,
    output_dir: Path,
    evaluation_steps: dict,
) -> argparse.Namespace:
    data = {
        "maps_file": game.DEFAULTS["maps_file"],
        "criteria_file": game.DEFAULTS["criteria_file"],
        "output_dir": str(output_dir),
        "video_cache_dir": args.video_cache_dir,
        "blind": experiment == EXPERIMENT_NO_CREATOR,
        "experiment": experiment,
        "forced_creator": forced_creator,
        "eval_steps_file": None,
        "evaluation_steps_override": evaluation_steps,
        "evaluation_steps_in_prompt": args.evaluation_steps_in_prompt,
        "video_source_dir": args.video_source_dir,
        "play_only": False,
        "resume": args.resume,
    }
    data.update(runtime_options(args))
    if args.solver_power is not None:
        data["solver_power"] = args.solver_power
    return argparse.Namespace(**data)


if __name__ == "__main__":
    main()
