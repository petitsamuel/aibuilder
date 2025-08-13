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
AGENT_HOST = os.getenv("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.getenv("DEVOPS_AGENT_PORT", os.getenv("AGENT_PORT", "8002")))


def _http_get(url: str, timeout: float = 2.0) -> tuple[int, str]:
    req = Request(url, headers={"User-Agent": "e2e-env-manager"})
    with urlopen(req, timeout=timeout) as resp:
        code = resp.getcode()
        body = resp.read().decode("utf-8", errors="ignore")
        return code, body


def _http_post_json(url: str, payload: dict, timeout: float = 15.0) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "User-Agent": "e2e-env-manager",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        code = resp.getcode()
        body = resp.read().decode("utf-8", errors="ignore")
        return code, body


def wait_for_http_ok(url: str, timeout_seconds: int = 30) -> None:
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


def main() -> int:
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
        cwd=str(REPO_ROOT),  # use repo root as source of truth
        env=registry_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        # Wait for Registry HTTP to be ready
        wait_for_http_ok(f"{registry_url}/status")

        # Start Environment Manager agent
        # Run as a module so imports like `agentkit` resolve from repo root
        agent_cmd = [
            sys.executable,
            "-m",
            "devopsagent.devops",
        ]
        agent_env = os.environ.copy()
        agent_env.setdefault("REGISTRY_URL", registry_url)
        agent_env.setdefault("AGENT_HOST", AGENT_HOST)
        agent_env.setdefault("AGENT_PORT", str(AGENT_PORT))
        agent_env.setdefault("WORKSPACE_ROOT", str(REPO_ROOT))
        agent_proc = subprocess.Popen(
            agent_cmd,
            cwd=str(REPO_ROOT),  # ensure module mode uses repo root
            env=agent_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            # Wait for agent health
            wait_for_http_ok(f"http://{AGENT_HOST}:{AGENT_PORT}/health")

            # Wait until agent appears in registry
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    _, body = _http_get(f"{registry_url}/status")
                    data = json.loads(body)
                    agents = data.get("agents") or []
                    names = {a.get("agent_name") for a in agents if isinstance(a, dict)}
                    if "devops-agent" in names:
                        break
                except Exception:
                    pass
                time.sleep(0.3)
            else:
                raise RuntimeError("Environment Manager did not register in time")

            # Prepare temporary destination inside workspace
            tmp_root = REPO_ROOT / f"tmp_env_e2e_{int(time.time())}"
            dest_dir = tmp_root / "project1"
            tmp_root.mkdir(parents=True, exist_ok=True)
            if dest_dir.exists():
                shutil.rmtree(dest_dir)

            # 1) Scaffold a project from templates/t3-app
            scaffold_url = f"http://{AGENT_HOST}:{AGENT_PORT}/scaffold_project"
            payload = {
                "template_path": "t3-app",
                "destination_path": str(dest_dir),
                "overwrite": True,
            }
            code, body = _http_post_json(scaffold_url, payload)
            if code != 200:
                raise RuntimeError(f"Scaffold failed: {code} {body}")
            result = json.loads(body)
            if result.get("status") != "ok":
                raise RuntimeError(f"Scaffold error: {result}")
            if not dest_dir.exists() or not dest_dir.is_dir():
                raise RuntimeError("Destination directory was not created")

            # 2) Run a fast dependency check (npm --version) in the project dir
            install_url = f"http://{AGENT_HOST}:{AGENT_PORT}/install_dependencies"
            payload = {
                "cwd": str(dest_dir),
                "args": ["npm", "--version"],
                "timeout": 60,
            }
            code, body = _http_post_json(install_url, payload)
            if code != 200:
                raise RuntimeError(f"Install check failed: {code} {body}")
            result = json.loads(body)
            if result.get("returncode") != 0:
                raise RuntimeError(f"npm --version failed: {result}")

            # 3) Initialize a git repository and commit files
            git_url = f"http://{AGENT_HOST}:{AGENT_PORT}/git_command"

            def git(args: list[str]) -> dict:
                c, b = _http_post_json(
                    git_url, {"cwd": str(dest_dir), "args": args, "timeout": 60}
                )
                if c != 200:
                    raise RuntimeError(f"git command failed: {c} {b}")
                return json.loads(b)

            res = git(["init"])
            if res.get("returncode") != 0:
                raise RuntimeError(f"git init failed: {res}")

            res = git(["add", "."])
            if res.get("returncode") != 0:
                raise RuntimeError(f"git add failed: {res}")

            res = git(
                [
                    "-c",
                    "user.name=E2E",
                    "-c",
                    "user.email=e2e@example.com",
                    "commit",
                    "-m",
                    "init",
                ]
            )
            if res.get("returncode") != 0:
                raise RuntimeError(f"git commit failed: {res}")

            print("\nE2E Env Manager test completed successfully.")
            print(f"Destination: {dest_dir}")

        finally:
            terminate_process(agent_proc, name="env-manager")

    finally:
        terminate_process(registry_proc, name="registry")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
