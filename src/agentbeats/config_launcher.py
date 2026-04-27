import argparse
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import yaml
from dotenv import load_dotenv


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORTS = {
    "green_agent": 9100,
    "designer": 9110,
    "player": 9120,
}


def load_run_toml(run_toml_path: Path) -> dict:
    if not run_toml_path.exists():
        raise FileNotFoundError(f"Run TOML not found: {run_toml_path}")
    return tomllib.loads(run_toml_path.read_text(encoding="utf-8"))


def load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML root must be a mapping.")
    return data


def get_selected_evaluation(config: dict, evaluation_name: str | None) -> dict:
    evaluations = config.get("evaluations", [])
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("config.yaml must define a non-empty 'evaluations' list.")

    if evaluation_name:
        for item in evaluations:
            if isinstance(item, dict) and item.get("name") == evaluation_name:
                return item
        raise ValueError(f"Evaluation named '{evaluation_name}' not found in config.yaml.")

    first = evaluations[0]
    if not isinstance(first, dict):
        raise ValueError("Each evaluation entry must be a mapping.")
    return first


def build_generated_scenario(config: dict, evaluation: dict) -> str:
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("'runtime' must be a mapping.")

    game = str(evaluation.get("game", "")).strip()
    if not game:
        raise ValueError("Selected evaluation is missing 'game'.")

    scenario_dir = Path("scenarios") / game
    evaluator_file = scenario_dir / f"{game}_evaluator.py"
    designer_file = scenario_dir / f"{game}_designer.py"
    player_file = scenario_dir / f"{game}_player.py"

    if not evaluator_file.exists():
        raise FileNotFoundError(f"Evaluator not found: {evaluator_file}")
    if not designer_file.exists():
        raise FileNotFoundError(f"Designer not found: {designer_file}")

    host = str(runtime.get("host", DEFAULT_HOST))
    ports = runtime.get("ports", {})
    if not isinstance(ports, dict):
        raise ValueError("'runtime.ports' must be a mapping.")

    green_port = int(ports.get("green_agent", DEFAULT_PORTS["green_agent"]))
    designer_port = int(ports.get("designer", DEFAULT_PORTS["designer"]))
    player_port = int(ports.get("player", DEFAULT_PORTS["player"]))

    green_cmd = runtime.get(
        "green_cmd",
        f"uv run python {evaluator_file.as_posix()} --host {host} --port {green_port}",
    )
    designer_cmd = runtime.get(
        "designer_cmd",
        f"uv run python {designer_file.as_posix()} --host {host} --port {designer_port}",
    )

    use_player = bool(runtime.get("use_player", True))
    player_cmd = runtime.get(
        "player_cmd",
        f"uv run python {player_file.as_posix()} --host {host} --port {player_port}",
    )

    designer_model = runtime.get("designer_model")
    output_root = str(runtime.get("output_root", "outputs"))
    output_dir = str(evaluation.get("output_dir", f"{output_root}/{game}"))
    num_maps = int(evaluation.get("iterations", 1))
    top_k = int(evaluation.get("top_k", num_maps))

    lines = [
        "[green_agent]",
        f'endpoint = "http://{host}:{green_port}"',
        f'cmd = "{green_cmd}"',
        "",
        "[[participants]]",
        'role = "designer"',
        f'endpoint = "http://{host}:{designer_port}"',
        f'cmd = "{designer_cmd}"',
    ]

    if designer_model:
        escaped_model = str(designer_model).replace('"', '\\"')
        lines.append(f'env = {{ MODEL = "{escaped_model}" }}')

    if use_player:
        if not player_file.exists():
            raise FileNotFoundError(f"Player is enabled but missing: {player_file}")
        lines.extend(
            [
                "",
                "[[participants]]",
                'role = "player"',
                f'endpoint = "http://{host}:{player_port}"',
                f'cmd = "{player_cmd}"',
            ]
        )

    lines.extend(
        [
            "",
            "[config]",
            f"num_maps = {num_maps}",
            f"top_k = {top_k}",
            f'jar_output_dir = "{output_dir}"',
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch AgentBeats using root TOML + YAML config.")
    parser.add_argument("run_toml", nargs="?", default="run.toml", help="Path to root run TOML")
    parser.add_argument("--evaluation", help="Optional evaluation name from config.yaml")
    parser.add_argument("--show-logs", action="store_true", help="Show agent stdout/stderr")
    parser.add_argument("--serve-only", action="store_true", help="Start agents only without running evaluation")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_toml_path = Path(args.run_toml)
    if not run_toml_path.is_absolute():
        run_toml_path = repo_root / run_toml_path

    run_toml = load_run_toml(run_toml_path)

    env_file_value = run_toml.get("env_file")
    if env_file_value:
        env_path = Path(str(env_file_value))
        if not env_path.is_absolute():
            env_path = repo_root / env_path
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)

    config_path_value = run_toml.get("config")
    if not config_path_value:
        raise ValueError("run.toml must define 'config'.")

    config_path = Path(str(config_path_value))
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    config = load_yaml_config(config_path)
    evaluation = get_selected_evaluation(config, args.evaluation or run_toml.get("evaluation"))
    generated_scenario = build_generated_scenario(config, evaluation)
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), dict) else {}
    show_logs = args.show_logs or bool(runtime.get("show_logs", False))
    serve_only = args.serve_only or bool(runtime.get("serve_only", False))

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as tmp:
        tmp.write(generated_scenario)
        temp_scenario_path = Path(tmp.name)

    try:
        cmd = [sys.executable, "-m", "agentbeats.run_scenario", str(temp_scenario_path)]
        if show_logs:
            cmd.append("--show-logs")
        if serve_only:
            cmd.append("--serve-only")

        subprocess.run(
            cmd,
            cwd=repo_root,
            env=os.environ.copy(),
            check=True,
        )
    finally:
        try:
            temp_scenario_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
