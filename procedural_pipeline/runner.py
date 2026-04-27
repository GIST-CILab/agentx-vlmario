import json
from datetime import datetime
from pathlib import Path

from procedural_pipeline.csv_output import write_csv
from procedural_pipeline.judge import (
    EvaluationTraceError,
    evaluate_video_multi,
    generate_evaluation_steps,
    load_criteria,
    resolve_game_profile,
)


BANNER = "=" * 72


def run(game, args) -> None:
    """Execute the procedural evaluation pipeline for a given game module."""
    criteria = load_criteria(args.criteria_file)
    profile = resolve_game_profile(game.DEFAULTS["profile_key"])

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output_dir / f"results_{run_timestamp}.csv"

    print(BANNER)
    print(f"[{game.KEY}] pipeline start  game={profile['game_name']}  model={args.model}")
    print(f"[{game.KEY}] output_dir={output_dir}  csv={csv_path.name}")
    print(BANNER)

    print(f"[{game.KEY}] step 1/2 - generating Auto-CoT evaluation steps (G-Eval)...")
    eval_steps = generate_evaluation_steps(criteria, profile, args.model)
    print_evaluation_steps(game.KEY, eval_steps["text"])

    steps_path = output_dir / f"evaluation_steps_{run_timestamp}.json"
    steps_path.write_text(json.dumps(eval_steps, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{game.KEY}] evaluation steps saved -> {steps_path}")

    print(f"[{game.KEY}] step 2/2 - evaluating videos with {args.num_runs} runs (temperature={args.temperature}, top_p={args.top_p})")

    results: list[dict] = []
    maps = game.load_maps(args.maps_file)[: args.limit]
    for index, (map_id, raw_map) in enumerate(maps, start=1):
        print(f"\n[{index}/{len(maps)}] {map_id}")
        result = {
            "map_index": index,
            "map_id": map_id,
            "status": "failed",
            "run_timestamp": run_timestamp,
            "game_profile": profile,
            "evaluation_steps": eval_steps["text"],
        }

        try:
            normalized_map = game.normalize_map(raw_map)
            map_path, video_path = render_or_reuse(game, normalized_map, map_id, str(output_dir), args)

            multi_trace = evaluate_video_multi(
                str(video_path),
                criteria,
                args.model,
                profile,
                eval_steps["text"],
                num_runs=args.num_runs,
                temperature=args.temperature,
                top_p=args.top_p,
                concurrency=args.concurrency,
                on_run=lambda i, total, _trace: print(f"    run {i}/{total}"),
            )

            judgment = multi_trace.get("aggregated_judgment")
            status = "ok" if judgment is not None else "failed"

            result.update({
                "status": status,
                "map_path": str(map_path),
                "video_path": str(video_path),
                "judgment": judgment,
                "num_runs": multi_trace.get("num_runs"),
                "num_parsed": multi_trace.get("num_parsed"),
                "llm": {
                    "prompt": multi_trace.get("prompt"),
                    "model": multi_trace.get("model"),
                    "temperature": multi_trace.get("temperature"),
                    "top_p": multi_trace.get("top_p"),
                    "evaluation_steps": multi_trace.get("evaluation_steps"),
                    "runs": multi_trace.get("runs"),
                },
            })
            if status == "failed":
                result["error"] = "All LLM runs failed to produce a parseable judgment."
        except EvaluationTraceError as exc:
            result["error"] = str(exc)
            result["llm"] = exc.trace
        except Exception as exc:
            result["error"] = str(exc)

        results.append(result)

        raw_path = raw_dir / f"{map_id}.json"
        raw_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        write_csv(str(csv_path), results)
        print(f"    csv updated -> {csv_path}")

    print(f"\n[{game.KEY}] done. csv={csv_path}")


def render_or_reuse(game, map_text: str, map_id: str, output_dir: str, args) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    map_path = out_dir / f"{map_id}.txt"
    video_path = out_dir / f"{map_id}.mp4"

    if video_path.exists() and video_path.stat().st_size > 0:
        if not map_path.exists():
            map_path.write_text(map_text, encoding="utf-8")
        print(f"    reusing existing video: {video_path}")
        return map_path, video_path

    return game.render_map(map_text, map_id, output_dir, args)


def print_evaluation_steps(game_key: str, steps_text: str) -> None:
    print(BANNER)
    print(f"[{game_key}] Auto-CoT Evaluation Steps:")
    print("-" * 72)
    if steps_text.strip():
        for line in steps_text.splitlines():
            print(f"  {line}" if line else "")
    else:
        print("  (empty)")
    print(BANNER)
