import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from procedural_pipeline.execution.csv_output import write_csv
from procedural_pipeline.judge import (
    EXPERIMENT_FORCED_CREATOR,
    EXPERIMENT_NO_CREATOR,
    EvaluationTraceError,
    evaluate_video_multi,
    generate_evaluation_steps,
    load_criteria,
    normalize_forced_creator,
    resolve_game_profile,
    resolve_experiment_mode,
)
from procedural_pipeline.paths import project_path


BANNER = "=" * 72


def run(game, args) -> None:
    """Execute the procedural evaluation pipeline for a given game module."""
    if getattr(args, "play_only", False):
        run_play_only(game, args)
        return

    criteria = load_criteria(args.criteria_file)
    profile = resolve_game_profile(game.DEFAULTS["profile_key"])

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    blind = bool(getattr(args, "blind", False))
    experiment = resolve_experiment_mode(getattr(args, "experiment", None), blind)
    forced_creator = _resolve_forced_creator(args, experiment)
    eval_steps_file = getattr(args, "eval_steps_file", None)
    evaluation_steps_in_prompt = bool(getattr(args, "evaluation_steps_in_prompt", True))
    video_source_dir = getattr(args, "video_source_dir", None)
    video_cache_dir = _video_cache_root(args)
    resume = bool(getattr(args, "resume", True))
    request_timeout = float(getattr(args, "request_timeout", 180))

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output_dir / f"results_{run_timestamp}.csv"
    maps = game.load_maps(args.maps_file)[: args.limit]

    print(BANNER)
    mode = _experiment_label(experiment, forced_creator)
    print(f"[{game.KEY}] pipeline start  game={profile['game_name']}  model={args.model}  mode={mode}")
    print(f"[{game.KEY}] output_dir={output_dir}  csv={csv_path.name}")
    print(f"[{game.KEY}] video_cache={video_cache_dir / game.KEY}")
    print(f"[{game.KEY}] sampling temperature={args.temperature}  top_p={args.top_p}")
    print(f"[{game.KEY}] request_timeout={request_timeout}s")
    print(f"[{game.KEY}] evaluation_steps_in_prompt={evaluation_steps_in_prompt}")
    print(f"[{game.KEY}] resume={resume}")
    if eval_steps_file:
        print(f"[{game.KEY}] reusing evaluation steps from: {eval_steps_file}")
    if video_source_dir:
        print(f"[{game.KEY}] reusing videos from:           {video_source_dir}")
    print(BANNER)

    resumed_steps_file = _latest_evaluation_steps_file(output_dir) if resume and not eval_steps_file else None
    eval_steps_override = getattr(args, "evaluation_steps_override", None)
    if resumed_steps_file:
        eval_steps = _load_evaluation_steps(str(resumed_steps_file))
        print(f"[{game.KEY}] step 1/2 - resuming Auto-CoT evaluation steps from: {resumed_steps_file}")
    elif eval_steps_override:
        eval_steps = eval_steps_override
        print(f"[{game.KEY}] step 1/2 - using provided Auto-CoT evaluation steps.")
    elif eval_steps_file:
        eval_steps = _load_evaluation_steps(eval_steps_file)
        print(f"[{game.KEY}] step 1/2 - loaded existing Auto-CoT evaluation steps.")
    else:
        print(f"[{game.KEY}] step 1/2 - generating Auto-CoT evaluation steps (G-Eval)...")
        eval_steps = generate_evaluation_steps(criteria, profile, args.model)
    print_evaluation_steps(game.KEY, eval_steps["text"])
    prompt_steps_text = eval_steps["text"] if evaluation_steps_in_prompt else ""

    steps_path = output_dir / f"evaluation_steps_{run_timestamp}.json"
    steps_path.write_text(json.dumps(eval_steps, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{game.KEY}] evaluation steps saved -> {steps_path}")

    print(f"[{game.KEY}] step 2/2 - evaluating videos with {args.num_runs} runs "
          f"(temperature={args.temperature}, top_p={args.top_p}, experiment={experiment})")

    results: list[dict] = []
    for index, (map_id, raw_map) in enumerate(maps, start=1):
        print(f"\n[{index}/{len(maps)}] {map_id}")
        resumed_result = _load_resumable_result(
            raw_dir,
            map_id,
            index,
            args,
            experiment,
            forced_creator,
            evaluation_steps_in_prompt,
        ) if resume else None
        if resumed_result:
            results.append(resumed_result)
            write_csv(str(csv_path), results)
            print(
                f"    resume: existing raw ok "
                f"({resumed_result.get('num_parsed', '')}/{resumed_result.get('num_runs', '')} parsed) -> skipped"
            )
            print(f"    csv updated -> {csv_path}")
            continue

        result = {
            "map_index": index,
            "map_id": map_id,
            "status": "failed",
            "run_timestamp": run_timestamp,
            "game_profile": profile,
            "evaluation_steps": eval_steps["text"],
            "evaluation_steps_in_prompt": evaluation_steps_in_prompt,
            "experiment": experiment,
            "blind": experiment == EXPERIMENT_NO_CREATOR,
            "forced_creator": forced_creator or "",
            "true_creator": _truth_label(index),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "request_timeout": request_timeout,
        }

        try:
            normalized_map = game.normalize_map(raw_map)
            map_path, video_path = render_or_reuse(
                game,
                normalized_map,
                map_id,
                str(output_dir),
                args,
                video_source_dir=video_source_dir,
            )

            multi_trace = evaluate_video_multi(
                str(video_path),
                criteria,
                args.model,
                profile,
                prompt_steps_text,
                num_runs=args.num_runs,
                temperature=args.temperature,
                top_p=args.top_p,
                concurrency=args.concurrency,
                request_timeout=request_timeout,
                on_run=_print_run_debug,
                blind=experiment == EXPERIMENT_NO_CREATOR,
                experiment=experiment,
                forced_creator=forced_creator,
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
                    "evaluation_steps_in_prompt": evaluation_steps_in_prompt,
                    "experiment": multi_trace.get("experiment"),
                    "forced_creator": multi_trace.get("forced_creator"),
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


def run_play_only(game, args) -> None:
    """게임 맵 → 영상(.mp4)만 생성. Auto-CoT·LLM 평가·criteria 없음."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = output_dir / f"play_only_{run_timestamp}.json"
    csv_path = output_dir / f"play_only_{run_timestamp}.csv"

    video_source_dir = getattr(args, "video_source_dir", None)
    video_cache_dir = _video_cache_root(args)
    resume = bool(getattr(args, "resume", True))

    print(BANNER)
    print(
        f"[{game.KEY}] play-only  (no G-Eval / no LLM)  "
        f"output_dir={output_dir}"
    )
    print(f"[{game.KEY}] video_cache={video_cache_dir / game.KEY}")
    print(f"[{game.KEY}] resume={resume}")
    if video_source_dir:
        print(f"[{game.KEY}] reusing videos from: {video_source_dir}")
    print(BANNER)

    results: list[dict] = []
    maps = game.load_maps(args.maps_file)[: args.limit]
    for index, (map_id, raw_map) in enumerate(maps, start=1):
        print(f"\n[{index}/{len(maps)}] {map_id}")
        if resume:
            resumed_row = _load_resumable_play_only_result(raw_dir, map_id, index)
            if resumed_row:
                results.append(resumed_row)
                _write_play_only_csv(csv_path, results)
                print("    resume: existing play-only raw ok -> skipped")
                continue

        row: dict = {
            "map_index": index,
            "map_id": map_id,
            "status": "failed",
            "run_timestamp": run_timestamp,
        }
        try:
            normalized_map = game.normalize_map(raw_map)
            map_path, video_path = render_or_reuse(
                game,
                normalized_map,
                map_id,
                str(output_dir),
                args,
                video_source_dir=video_source_dir,
            )
            row.update(
                {
                    "status": "ok",
                    "map_path": str(map_path),
                    "video_path": str(video_path),
                }
            )
            print(f"    video -> {video_path}")
        except Exception as exc:
            row["error"] = str(exc)
            print(f"    error: {exc}")

        results.append(row)
        raw_path = raw_dir / f"{map_id}.json"
        raw_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_play_only_csv(csv_path, results)

    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[{game.KEY}] play-only done. summary={summary_path}  csv={csv_path}")


def _write_play_only_csv(csv_path: Path, results: list[dict]) -> None:
    """play-only용 최소 CSV: map_id, status, video_path, error"""
    fieldnames = ("map_id", "status", "video_path", "error")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "map_id": r.get("map_id", ""),
                    "status": r.get("status", ""),
                    "video_path": r.get("video_path", ""),
                    "error": r.get("error", ""),
                }
            )


def _resolve_forced_creator(args, experiment: str) -> str | None:
    value = getattr(args, "forced_creator", None)
    if experiment == EXPERIMENT_FORCED_CREATOR:
        return normalize_forced_creator(value)
    return None


def _experiment_label(experiment: str, forced_creator: str | None) -> str:
    if experiment == EXPERIMENT_FORCED_CREATOR:
        return f"{experiment}:{forced_creator}"
    return experiment


def _truth_label(map_index: int) -> str:
    return "Human" if map_index <= 15 else "AI"


def _print_run_debug(index: int, total: int, trace: dict) -> None:
    temp_debug = trace.get("temperature_debug") or {}
    requested = temp_debug.get("requested", trace.get("temperature", ""))
    payload = temp_debug.get("payload", "")
    response = temp_debug.get("response")
    response_text = response if response is not None else "not_returned"
    print(
        f"    run {index}/{total} "
        f"temp(request={requested}, payload={payload}, response={response_text})"
    )


def render_or_reuse(
    game,
    map_text: str,
    map_id: str,
    output_dir: str,
    args,
    video_source_dir: str | None = None,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / f"{map_id}.txt"
    legacy_video_path = out_dir / f"{map_id}.mp4"
    cache_map_path, cache_video_path = _video_cache_paths(game, map_id, args)
    cache_map_path.parent.mkdir(parents=True, exist_ok=True)

    _ensure_map_file(map_path, map_text)

    if _usable_video(cache_video_path):
        _ensure_map_file(cache_map_path, map_text)
        print(f"    reusing cached video: {cache_video_path}")
        return map_path, cache_video_path

    if _usable_video(legacy_video_path):
        _ensure_map_file(cache_map_path, map_text)
        shutil.copyfile(legacy_video_path, cache_video_path)
        print(f"    cached existing output video -> {cache_video_path}")
        return map_path, cache_video_path

    if video_source_dir:
        src_dir = Path(video_source_dir)
        src_video = src_dir / f"{map_id}.mp4"
        if _usable_video(src_video):
            src_map = src_dir / f"{map_id}.txt"
            if src_map.exists():
                cached_map_text = src_map.read_text(encoding="utf-8")
                map_path.write_text(cached_map_text, encoding="utf-8")
                cache_map_path.write_text(cached_map_text, encoding="utf-8")
            else:
                _ensure_map_file(cache_map_path, map_text)
            shutil.copyfile(src_video, cache_video_path)
            print(f"    cached video from source dir -> {cache_video_path}")
            return map_path, cache_video_path

    _, rendered_video_path = game.render_map(map_text, map_id, str(cache_video_path.parent), args)
    if rendered_video_path != cache_video_path and _usable_video(rendered_video_path):
        shutil.copyfile(rendered_video_path, cache_video_path)
    _ensure_map_file(cache_map_path, map_text)
    print(f"    rendered video cached -> {cache_video_path}")
    return map_path, cache_video_path


def _video_cache_root(args) -> Path:
    return Path(getattr(args, "video_cache_dir", None) or project_path("outputs", "video"))


def _video_cache_paths(game, map_id: str, args) -> tuple[Path, Path]:
    cache_dir = _video_cache_root(args) / game.KEY
    return cache_dir / f"{map_id}.txt", cache_dir / f"{map_id}.mp4"


def _usable_video(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1024


def _ensure_map_file(path: Path, map_text: str) -> None:
    if not path.exists():
        path.write_text(map_text, encoding="utf-8")


def _load_evaluation_steps(path: str) -> dict:
    """Read a previously-saved evaluation_steps_*.json and return the same dict shape
    that generate_evaluation_steps() returns."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    text = data.get("text")
    if not text:
        parsed = data.get("parsed") or {}
        steps = parsed.get("evaluation_steps") or parsed.get("steps") or []
        if isinstance(steps, str):
            text = steps.strip()
        else:
            text = "\n".join(str(s).strip() for s in steps if str(s).strip())
    if not text:
        raise ValueError(f"No usable evaluation_steps text found in {path}")

    data["text"] = text
    data.setdefault("source_path", str(path))
    data.setdefault("reused", True)
    return data


def _latest_evaluation_steps_file(output_dir: Path) -> Path | None:
    candidates = sorted(
        output_dir.glob("evaluation_steps_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_resumable_result(
    raw_dir: Path,
    map_id: str,
    map_index: int,
    args,
    experiment: str,
    forced_creator: str | None,
    evaluation_steps_in_prompt: bool,
) -> dict | None:
    path = raw_dir / f"{map_id}.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not _is_resumable_result(
        result,
        args,
        experiment,
        forced_creator,
        evaluation_steps_in_prompt,
    ):
        return None

    result["map_index"] = map_index
    result["map_id"] = map_id
    return result


def _is_resumable_result(
    result: dict,
    args,
    experiment: str,
    forced_creator: str | None,
    evaluation_steps_in_prompt: bool,
) -> bool:
    if result.get("status") != "ok" or not result.get("judgment"):
        return False
    if result.get("experiment") and result.get("experiment") != experiment:
        return False
    existing_forced = result.get("forced_creator") or None
    if (existing_forced or None) != (forced_creator or None):
        return False
    if result.get("evaluation_steps_in_prompt") not in (None, "", evaluation_steps_in_prompt):
        return False
    if _as_int(result.get("num_runs")) < int(getattr(args, "num_runs", 0)):
        return False
    if _as_int(result.get("num_parsed")) <= 0:
        return False

    llm = result.get("llm") or {}
    model = llm.get("model")
    if model and model != getattr(args, "model", None):
        return False

    return True


def _load_resumable_play_only_result(raw_dir: Path, map_id: str, map_index: int) -> dict | None:
    path = raw_dir / f"{map_id}.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("status") != "ok":
        return None
    video_path = result.get("video_path")
    if video_path and not _usable_video(Path(video_path)):
        return None
    result["map_index"] = map_index
    result["map_id"] = map_id
    return result


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
