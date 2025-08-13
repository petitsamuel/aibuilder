import os
import sys
import time
import json
import signal
import sqlite3
import tempfile
import subprocess
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_HOST = os.getenv("AGENT_REGISTRY_HOST", "127.0.0.1")
REGISTRY_PORT = int(os.getenv("AGENT_REGISTRY_PORT", "8000"))
AGENT_HOST = os.getenv("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8001"))


def _http_get(url: str, timeout: float = 2.0) -> tuple[int, str]:
    req = Request(url, headers={"User-Agent": "e2e-smoke-test"})
    with urlopen(req, timeout=timeout) as resp:
        code = resp.getcode()
        body = resp.read().decode("utf-8", errors="ignore")
        return code, body


def wait_for_http_ok(url: str, timeout_seconds: int = 30, require_json_key: str | None = None) -> None:
    deadline = time.time() + timeout_seconds
    last_err: str | None = None
    while time.time() < deadline:
        try:
            code, body = _http_get(url)
            if code == 200:
                if require_json_key is None:
                    return
                try:
                    data = json.loads(body)
                    if require_json_key in data:
                        return
                except json.JSONDecodeError:
                    pass
        except URLError as e:
            last_err = str(e)
        except Exception as e:
            last_err = str(e)
        time.sleep(0.3)
    raise RuntimeError(f"Timeout waiting for {url}. Last error: {last_err}")


def terminate_process(proc: subprocess.Popen, name: str, timeout_seconds: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        proc.kill()
    except Exception:
        pass


def main() -> int:
    goal = os.getenv("E2E_GOAL", "Prototype a simple app and pick a tech stack")
    registry_url = f"http://{REGISTRY_HOST}:{REGISTRY_PORT}"

    # Start Registry
    registry_cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "agent-registry" / "agentregistry.py"),
    ]
    registry_env = os.environ.copy()
    registry_env.setdefault("AGENT_REGISTRY_HOST", REGISTRY_HOST)
    registry_env.setdefault("AGENT_REGISTRY_PORT", str(REGISTRY_PORT))
    registry_proc = subprocess.Popen(
        registry_cmd,
        cwd=str(REPO_ROOT / "agent-registry"),
        env=registry_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Wait for Registry HTTP to be ready (status endpoint)
        wait_for_http_ok(f"{registry_url}/status", timeout_seconds=30, require_json_key="agents")

        # Start Agent (Tech Stack Selector)
        agent_cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "architect" / "architect.py"),
        ]
        agent_env = os.environ.copy()
        agent_env.setdefault("REGISTRY_URL", registry_url)
        agent_env.setdefault("AGENT_HOST", AGENT_HOST)
        agent_env.setdefault("AGENT_PORT", str(AGENT_PORT))
        agent_proc = subprocess.Popen(
            agent_cmd,
            cwd=str(REPO_ROOT / "architect"),
            env=agent_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            # Wait for Agent health
            wait_for_http_ok(f"http://{AGENT_HOST}:{AGENT_PORT}/health", timeout_seconds=30)

            # Wait until agent appears in registry (via heartbeat)
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    _, body = _http_get(f"{registry_url}/status")
                    data = json.loads(body)
                    agents = data.get("agents") or []
                    names = {a.get("agent_name") for a in agents if isinstance(a, dict)}
                    if "tech-stack-selector" in names:
                        break
                except Exception:
                    pass
                time.sleep(0.3)
            else:
                raise RuntimeError("Agent did not register in time")

            # Prepare temp DB for orchestrator
            tmp_dir = tempfile.mkdtemp(prefix="orch_e2e_")
            db_path = str(Path(tmp_dir) / "orchestrator_state.sqlite")

            # Run one-turn Orchestrator
            orch_cmd = [
                sys.executable,
                "-u",
                str(REPO_ROOT / "orchestrator" / "orchestrator.py"),
                "--goal",
                goal,
                "--registry-url",
                registry_url,
                "--db-path",
                db_path,
                "--project-root",
                str(REPO_ROOT),
            ]
            result = subprocess.run(
                orch_cmd,
                cwd=str(REPO_ROOT / "orchestrator"),
                text=True,
                capture_output=True,
            )

            print("\n=== Orchestrator stdout ===")
            print(result.stdout)
            print("=== Orchestrator stderr ===")
            print(result.stderr)

            if result.returncode != 0:
                raise SystemExit(result.returncode)

            # Inspect DB for a history entry to confirm a successful end-to-end call
            try:
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                with conn:
                    rows = conn.execute("SELECT id, timestamp, agent_name, result_summary FROM history ORDER BY id DESC LIMIT 5").fetchall()
                if not rows:
                    raise RuntimeError("No history entries found; orchestrator may not have completed correctly")
                latest = rows[0]
                print("\n=== E2E Result Summary ===")
                print(f"Agent: {latest['agent_name']}")
                print(f"Timestamp: {latest['timestamp']}")
                print(f"Summary: {latest['result_summary']}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        finally:
            terminate_process(agent_proc, name="agent")

    finally:
        terminate_process(registry_proc, name="registry")

    print("\nE2E smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)


