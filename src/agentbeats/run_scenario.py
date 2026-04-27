import argparse
import asyncio
import os, sys, time, subprocess, shlex, signal
import re
from pathlib import Path
import tomllib
import httpx
from dotenv import load_dotenv

from a2a.client import A2ACardResolver


load_dotenv(override=True)

LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def is_local_host(host: str) -> bool:
    return (host or "").strip().lower() in LOCAL_HOSTS


def get_listening_pids(host: str, port: int) -> list[int]:
    """Return PIDs listening on the given port for local endpoints."""
    if not is_local_host(host):
        return []

    if os.name == "nt":
        proc = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            check=False,
        )
        pids: set[int] = set()
        for line in proc.stdout.splitlines():
            if f":{port}" not in line or "LISTENING" not in line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            local_addr = parts[1]
            state = parts[3]
            pid_text = parts[4]
            if state != "LISTENING":
                continue
            if not (
                local_addr.endswith(f":{port}")
                or local_addr.lower().endswith(f"]:{port}")
            ):
                continue
            if pid_text.isdigit():
                pids.add(int(pid_text))
        return sorted(pids)

    proc = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted(
        {
            int(match.group(0))
            for match in re.finditer(r"\d+", proc.stdout)
            if int(match.group(0)) != os.getpid()
        }
    )


def kill_pid(pid: int) -> None:
    if pid == os.getpid():
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return


def ensure_port_free(host: str, port: int, timeout: float = 5.0) -> None:
    pids = get_listening_pids(host, port)
    if not pids:
        return

    print(f"Port {host}:{port} already in use. Terminating PID(s): {', '.join(str(pid) for pid in pids)}")
    for pid in pids:
        kill_pid(pid)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not get_listening_pids(host, port):
            print(f"Freed port {host}:{port}")
            return
        time.sleep(0.2)

    remaining = get_listening_pids(host, port)
    if remaining:
        raise RuntimeError(
            f"Failed to free port {host}:{port}. Remaining PID(s): {', '.join(str(pid) for pid in remaining)}"
        )


def cleanup_bound_ports(cfg: dict) -> None:
    targets: set[tuple[str, int]] = set()
    for participant in cfg["participants"]:
        if participant.get("cmd"):
            targets.add((participant["host"], int(participant["port"])))
    if cfg["green_agent"].get("cmd"):
        targets.add((cfg["green_agent"]["host"], int(cfg["green_agent"]["port"])))

    for host, port in sorted(targets):
        ensure_port_free(host, port)


async def wait_for_agents(cfg: dict, timeout: int = 30) -> bool:
    """Wait for all agents to be healthy and responding."""
    endpoints = []

    # Collect all endpoints to check
    for p in cfg["participants"]:
        if p.get("cmd"):  # Only check if there's a command (agent to start)
            endpoints.append(f"http://{p['host']}:{p['port']}")

    if cfg["green_agent"].get("cmd"):  # Only check if there's a command (host to start)
        endpoints.append(f"http://{cfg['green_agent']['host']}:{cfg['green_agent']['port']}")

    if not endpoints:
        return True  # No agents to wait for

    print(f"Waiting for {len(endpoints)} agent(s) to be ready...")
    start_time = time.time()

    async def check_endpoint(endpoint: str) -> bool:
        """Check if an endpoint is responding by fetching the agent card."""
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resolver = A2ACardResolver(httpx_client=client, base_url=endpoint)
                await resolver.get_agent_card()
                return True
        except Exception:
            # Any exception means the agent is not ready
            return False

    while time.time() - start_time < timeout:
        ready_count = 0
        for endpoint in endpoints:
            if await check_endpoint(endpoint):
                ready_count += 1

        if ready_count == len(endpoints):
            return True

        print(f"  {ready_count}/{len(endpoints)} agents ready, waiting...")
        await asyncio.sleep(1)

    print(f"Timeout: Only {ready_count}/{len(endpoints)} agents became ready after {timeout}s")
    return False


def parse_toml(scenario_path: str) -> dict:
    path = Path(scenario_path)
    if not path.exists():
        print(f"Error: Scenario file not found: {path}")
        sys.exit(1)

    data = tomllib.loads(path.read_text())

    def host_port(ep: str):
        s = (ep or "")
        s = s.replace("http://", "").replace("https://", "")
        s = s.split("/", 1)[0]
        host, port = s.split(":", 1)
        return host, int(port)

    green = data.get("green_agent", {})
    green_ep = green.get("endpoint", "")
    g_host, g_port = host_port(green_ep)
    green_cmd = green.get("cmd", "")

    # Optional: commands to run before starting the green agent.
    # Supports either a single string (pre_cmd) or a list of strings (pre_cmds).
    green_pre_cmd = green.get("pre_cmd", "")
    green_pre_cmds = green.get("pre_cmds", [])
    if isinstance(green_pre_cmds, str):
        green_pre_cmds = [green_pre_cmds]
    if not isinstance(green_pre_cmds, list):
        green_pre_cmds = []

    parts = []
    for p in data.get("participants", []):
        if isinstance(p, dict) and "endpoint" in p:
            h, pt = host_port(p["endpoint"])
            parts.append({
                "role": str(p.get("role", "")),
                "host": h,
                "port": pt,
                "cmd": p.get("cmd", ""),
                "env": p.get("env", {})
            })

    cfg = data.get("config", {})
    return {
        "green_agent": {
            "host": g_host,
            "port": g_port,
            "cmd": green_cmd,
            "pre_cmd": green_pre_cmd,
            "pre_cmds": green_pre_cmds,
        },
        "participants": parts,
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser(description="Run agent scenario")
    parser.add_argument("scenario", help="Path to scenario TOML file")
    parser.add_argument("--show-logs", action="store_true",
                        help="Show agent stdout/stderr")
    parser.add_argument("--serve-only", action="store_true",
                        help="Start agent servers only without running evaluation")
    args = parser.parse_args()

    cfg = parse_toml(args.scenario)
    cleanup_bound_ports(cfg)

    sink = None if args.show_logs or args.serve_only else subprocess.DEVNULL
    parent_bin = str(Path(sys.executable).parent)
    base_env = os.environ.copy()
    base_env["PATH"] = parent_bin + os.pathsep + base_env.get("PATH", "")

    repo_root = Path(__file__).resolve().parents[2]

    procs = []
    try:
        # start participant agents
        for p in cfg["participants"]:
            cmd_args = shlex.split(p.get("cmd", ""))
            if cmd_args:
                print(f"Starting {p['role']} at {p['host']}:{p['port']}")
                p_env = base_env.copy()
                # Inject custom env vars from TOML
                custom_env = p.get("env", {})
                if isinstance(custom_env, dict):
                    for k, v in custom_env.items():
                        # Simple shell-like substitution for ${VAR}
                        val = str(v)
                        if val.startswith("${") and val.endswith("}"):
                            env_name = val[2:-1]
                            val = os.getenv(env_name, val)
                        p_env[str(k)] = val

                procs.append(subprocess.Popen(
                    cmd_args,
                    env=p_env,
                    cwd=str(repo_root),
                    stdout=sink, stderr=sink,
                    text=True,
                    start_new_session=True,
                ))

        # Run pre-commands before starting the green agent
        pre_cmds: list[str] = []
        if cfg["green_agent"].get("pre_cmd"):
            pre_cmds.append(str(cfg["green_agent"]["pre_cmd"]))
        pre_cmds.extend([str(x) for x in cfg["green_agent"].get("pre_cmds", []) if x])

        for pre_cmd in pre_cmds:
            print(f"Running green_agent pre-cmd: {pre_cmd}")
            subprocess.run(
                pre_cmd,
                env=base_env,
                cwd=str(repo_root),
                shell=True,
                check=True,
            )

        # start host
        green_cmd_args = shlex.split(cfg["green_agent"].get("cmd", ""))
        if green_cmd_args:
            print(f"Starting green agent at {cfg['green_agent']['host']}:{cfg['green_agent']['port']}")
            procs.append(subprocess.Popen(
                green_cmd_args,
                env=base_env,
                cwd=str(repo_root),
                stdout=sink, stderr=sink,
                text=True,
                start_new_session=True,
            ))

        # Wait for all agents to be ready
        if not asyncio.run(wait_for_agents(cfg)):
            print("Error: Not all agents became ready. Exiting.")
            return

        print("Agents started. Press Ctrl+C to stop.")
        if args.serve_only:
            while True:
                for proc in procs:
                    if proc.poll() is not None:
                        print(f"Agent exited with code {proc.returncode}")
                        break
                    time.sleep(0.5)
        else:
            client_proc = subprocess.Popen(
                [sys.executable, "-m", "agentbeats.client_cli", args.scenario],
                env=base_env,
                cwd=str(repo_root),
                start_new_session=True,
            )
            procs.append(client_proc)
            client_proc.wait()

    except KeyboardInterrupt:
        pass

    finally:
        print("\nShutting down...")
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        time.sleep(1)
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()