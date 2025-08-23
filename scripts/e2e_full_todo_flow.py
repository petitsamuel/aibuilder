import os
import sys
import time
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_HOST = os.getenv("AGENT_REGISTRY_HOST", "127.0.0.1")
REGISTRY_PORT = int(os.getenv("AGENT_REGISTRY_PORT", "8000"))

ARCH_HOST = os.getenv("ARCHITECT_HOST", "127.0.0.1")
ARCH_PORT = int(os.getenv("ARCHITECT_PORT", "8001"))

DEVOPS_HOST = os.getenv("DEVOPS_HOST", "127.0.0.1")
DEVOPS_PORT = int(os.getenv("DEVOPS_PORT", "8002"))

CODER_HOST = os.getenv("CODER_HOST", "127.0.0.1")
CODER_PORT = int(os.getenv("CODER_PORT", "8003"))

QA_HOST = os.getenv("QA_HOST", "127.0.0.1")
QA_PORT = int(os.getenv("QA_PORT", "8004"))


def _http_get(url: str, timeout: float = 3.0) -> tuple[int, str]:
    req = Request(url, headers={"User-Agent": "e2e-full"})
    with urlopen(req, timeout=timeout) as resp:
        code = resp.getcode()
        body = resp.read().decode("utf-8", errors="ignore")
        return code, body


def wait_for_http_ok(url: str, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    last_err: str | None = None
    while time.time() < deadline:
        try:
            code, _ = _http_get(url)
            if code == 200:
                return
        except URLError as e:
            last_err = str(e)
        except Exception as e:
            last_err = str(e)
        time.sleep(0.3)
    raise RuntimeError(f"Timeout waiting for {url}. Last error: {last_err}")


def terminate_process(
    proc: subprocess.Popen, name: str, timeout_seconds: float = 5.0
) -> None:
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


def start_registry() -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "agent-registry" / "agentregistry.py"),
    ]
    env = os.environ.copy()
    env.setdefault("AGENT_REGISTRY_HOST", REGISTRY_HOST)
    env.setdefault("AGENT_REGISTRY_PORT", str(REGISTRY_PORT))
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_until_registered(
    registry_url: str, agent_names: set[str], timeout_seconds: int = 40
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            _, body = _http_get(f"{registry_url}/status")
            data = json.loads(body)
            agents = data.get("agents") or []
            names = {a.get("agent_name") for a in agents if isinstance(a, dict)}
            if agent_names.issubset(names):
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"Agents {agent_names} did not register in time")


def start_agent(
    module: str, host: str, port: int, extra_env: dict[str, str] | None = None
) -> subprocess.Popen:
    cmd = [sys.executable, "-m", module]
    env = os.environ.copy()
    env.setdefault("REGISTRY_URL", f"http://{REGISTRY_HOST}:{REGISTRY_PORT}")
    env.setdefault("AGENT_HOST", host)
    env.setdefault("AGENT_PORT", str(port))
    if module.startswith("devopsagent"):
        env.setdefault("WORKSPACE_ROOT", str(REPO_ROOT))
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def main() -> int:
    goal = "Build me a simple to-do list app where I can add and see tasks. It needs a database."
    registry_url = f"http://{REGISTRY_HOST}:{REGISTRY_PORT}"

    # Fresh orchestrator state and temp project root
    tmp_root = REPO_ROOT / f"tmp_full_e2e_{int(time.time())}"
    project_root = tmp_root / "workspace"
    project_root.mkdir(parents=True, exist_ok=True)
    db_path = str(tmp_root / "orchestrator_state.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)

    # Start registry
    print("Starting registry...")
    registry_proc = start_registry()
    try:
        wait_for_http_ok(f"{registry_url}/status")

        # Start all agents
        print("Starting agents: architect, devops, coder, qa ...")
        # Ensure templates are available inside the ephemeral workspace for DevOps security checks
        templates_src = REPO_ROOT / "templates"
        templates_dst = project_root / "templates"
        try:
            if templates_src.exists():
                shutil.copytree(templates_src, templates_dst, dirs_exist_ok=True)
        except Exception:
            pass

        arch_proc = start_agent("architect.architect", ARCH_HOST, ARCH_PORT)
        devops_proc = start_agent(
            "devopsagent.devops",
            DEVOPS_HOST,
            DEVOPS_PORT,
            extra_env={
                # Point DevOps workspace to our ephemeral project_root
                "WORKSPACE_ROOT": str(project_root),
                # Use copied templates within the workspace
                "TEMPLATES_DIR": str(templates_dst),
                "TEMPLATE_MANIFEST": str(templates_dst / "manifest.yaml"),
            },
        )
        coder_proc = start_agent("coder.coder", CODER_HOST, CODER_PORT)
        qa_proc = start_agent("qaagent.qa", QA_HOST, QA_PORT)

        try:
            # Wait for health endpoints
            wait_for_http_ok(f"http://{ARCH_HOST}:{ARCH_PORT}/health")
            wait_for_http_ok(f"http://{DEVOPS_HOST}:{DEVOPS_PORT}/health")
            wait_for_http_ok(f"http://{CODER_HOST}:{CODER_PORT}/health")
            wait_for_http_ok(f"http://{QA_HOST}:{QA_PORT}/health")

            # Ensure all are registered
            wait_until_registered(
                registry_url,
                agent_names={
                    "tech-stack-selector",
                    "devops-agent",
                    "coder",
                    "qa-agent",
                },
            )

            # Run orchestrator for several turns
            orch_cmd = [
                sys.executable,
                "-m",
                "orchestrator.orchestrator",
                "--goal",
                goal,
                "--registry-url",
                registry_url,
                "--db-path",
                db_path,
                "--project-root",
                str(project_root),
                "--continuous",
                "--interval",
                "3",
            ]

            print("Starting orchestrator (continuous for ~20s)...")
            orch_proc = subprocess.Popen(
                orch_cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Stream history in near-real-time while orchestrator is running
            import sqlite3

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            last_printed_id = 0
            start_time = time.time()
            try:
                while True:
                    # Print any new history rows
                    try:
                        rows = conn.execute(
                            "SELECT id, timestamp, agent_name, result_summary, tool_name, tool_input FROM history WHERE id > ? ORDER BY id ASC",
                            (last_printed_id,),
                        ).fetchall()
                    except Exception:
                        rows = []
                    for r in rows:
                        ts = r["timestamp"]
                        agent = r["agent_name"]
                        summary = r["result_summary"]
                        tool = r["tool_name"] if "tool_name" in r.keys() else None
                        tinp = r["tool_input"] if "tool_input" in r.keys() else None
                        if r["id"] == last_printed_id:
                            continue
                        last_printed_id = r["id"]
                        if last_printed_id == 1:
                            print("History entries:")
                        if tool and tinp:
                            print(
                                f"- {ts} | {agent} | {summary} | tool={tool} input={tinp}"
                            )
                        else:
                            print(f"- {ts} | {agent} | {summary}")

                    # Also stream orchestrator stdout for debug
                    try:
                        assert orch_proc.stdout is not None
                        while True:
                            line = orch_proc.stdout.readline()
                            if not line:
                                break
                            sys.stdout.write(line)
                            sys.stdout.flush()
                    except Exception:
                        pass

                    # Exit if orchestrator has terminated or time budget reached
                    if orch_proc.poll() is not None:
                        break
                    if time.time() - start_time > 20:
                        break
                    time.sleep(0.5)
            finally:
                try:
                    terminate_process(orch_proc, name="orchestrator")
                except Exception:
                    pass

            # Validate DB has plan and at least one history entry
            import sqlite3

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                with conn:
                    plan_rows = conn.execute(
                        "SELECT id, description, status FROM plan ORDER BY id ASC"
                    ).fetchall()
                    hist_rows = conn.execute(
                        "SELECT id, timestamp, agent_name, task_description, result_summary FROM history ORDER BY id ASC"
                    ).fetchall()

                if not plan_rows:
                    raise RuntimeError(
                        "Plan table is empty; orchestrator did not seed plan"
                    )
                if not hist_rows:
                    raise RuntimeError(
                        "History table is empty; orchestrator did not perform any action"
                    )

                # After stopping, ensure at least one history row exists (already printed live above)

            finally:
                try:
                    conn.close()
                except Exception:
                    pass

            print("\nFull E2E Orchestrator flow completed.")

        finally:
            terminate_process(qa_proc, name="qa")
            terminate_process(coder_proc, name="coder")
            terminate_process(devops_proc, name="devops")
            terminate_process(arch_proc, name="architect")

    finally:
        terminate_process(registry_proc, name="registry")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
