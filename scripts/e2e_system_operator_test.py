import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import urlencode


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_HOST = os.getenv("AGENT_REGISTRY_HOST", "127.0.0.1")
REGISTRY_PORT = int(os.getenv("AGENT_REGISTRY_PORT", "8000"))
AGENT_HOST = os.getenv("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.getenv("DEVOPS_AGENT_PORT", os.getenv("AGENT_PORT", "8002")))


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    req = Request(url, headers={"User-Agent": "e2e-system-operator"})
    with urlopen(req, timeout=timeout) as resp:
        code = resp.getcode()
        body = resp.read().decode("utf-8", errors="ignore")
        return code, body


def _http_post_json(url: str, payload: dict, timeout: float = 30.0) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "User-Agent": "e2e-system-operator",
            "Content-Type": "application/json",
        },
        method="POST",
    )
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
        time.sleep(0.4)
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


def print_section(title: str) -> None:
    print("\n" + "=" * 8 + f" {title} " + "=" * 8)


def stream_logs(url: str, duration_seconds: float = 8.0) -> None:
    print(f"Streaming logs for ~{duration_seconds:.0f}s from: {url}")
    start = time.time()
    try:
        req = Request(url, headers={"User-Agent": "e2e-system-operator"}, method="GET")
        with urlopen(req, timeout=duration_seconds + 5.0) as resp:
            # Iterate lines until duration reached or stream ends
            while time.time() - start < duration_seconds:
                line = resp.readline()
                if not line:
                    break
                try:
                    text = line.decode("utf-8", errors="ignore").rstrip("\n")
                except Exception:
                    text = str(line)
                print(text)
    except Exception as e:
        print(f"[logs stream error] {e}")


def wait_until_service_live(
    *,
    project_dir: Path,
    service: str = "app",
    http_host: str = "127.0.0.1",
    http_port: int = 3000,
    http_path: str = "/",
    timeout_seconds: int = 300,
) -> None:
    print(
        f"Waiting for service '{service}' to become live at http://{http_host}:{http_port}{http_path} (timeout {timeout_seconds}s)..."
    )
    start = time.time()
    last_status_print = 0.0
    status_url = f"http://{AGENT_HOST}:{AGENT_PORT}/system_operator/status"
    health_url = f"http://{AGENT_HOST}:{AGENT_PORT}/system_operator/http_health"
    while time.time() - start < timeout_seconds:
        # 1) Check docker compose status (best-effort)
        try:
            code, body = _http_post_json(
                status_url, {"cwd": str(project_dir), "service": service}
            )
            if code == 200:
                data = json.loads(body)
                if isinstance(data, dict) and data.get("service_running") is True:
                    # Service container is running; proceed to HTTP check
                    pass
        except Exception:
            pass

        # 2) HTTP health check via agent
        try:
            code, body = _http_post_json(
                health_url,
                {"host": http_host, "port": http_port, "path": http_path, "timeout": 2},
                timeout=5.0,
            )
            if code == 200:
                result = json.loads(body)
                if isinstance(result, dict) and result.get("ok"):
                    print("Service responded:", body)
                    return
        except Exception:
            pass

        # Periodic progress output
        if time.time() - last_status_print > 5.0:
            waited = int(time.time() - start)
            print(f"... still waiting ({waited}s elapsed)")
            last_status_print = time.time()

        time.sleep(1.5)
    raise RuntimeError("Service did not become live before timeout")


def main() -> int:
    registry_url = f"http://{REGISTRY_HOST}:{REGISTRY_PORT}"

    # Start Registry
    print_section("Start Agent Registry")
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
        cwd=str(REPO_ROOT),
        env=registry_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        wait_for_http_ok(f"{registry_url}/status")

        # Start DevOps Agent
        print_section("Start DevOps Agent")
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
            cwd=str(REPO_ROOT),
            env=agent_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            wait_for_http_ok(f"http://{AGENT_HOST}:{AGENT_PORT}/health")

            # Ensure agent registered
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
                time.sleep(0.25)
            else:
                raise RuntimeError("DevOps agent did not register in time")

            # Prepare destination
            print_section("Scaffold Template t3-app")
            tmp_root = REPO_ROOT / f"tmp_system_op_{int(time.time())}"
            dest_dir = tmp_root / "project1"
            tmp_root.mkdir(parents=True, exist_ok=True)
            if dest_dir.exists():
                shutil.rmtree(dest_dir)

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
            print(f"Scaffolded to: {dest_dir}")

            # Start via System Operator
            print_section("System Operator: start (docker compose up -d)")
            start_url = f"http://{AGENT_HOST}:{AGENT_PORT}/system_operator/start"
            code, body = _http_post_json(
                start_url,
                {"cwd": str(dest_dir), "build": False},
            )
            if code != 200:
                raise RuntimeError(f"system_start failed: {code} {body}")
            print("Started containers")

            # Wait until service is live (builds can take time on first run)
            print_section("Wait until service is live on :3000")
            wait_until_service_live(
                project_dir=dest_dir,
                service="app",
                http_host="127.0.0.1",
                http_port=3000,
                http_path="/",
                timeout_seconds=420,
            )

            # Status
            print_section("System Operator: status")
            status_url = f"http://{AGENT_HOST}:{AGENT_PORT}/system_operator/status"
            code, body = _http_post_json(status_url, {"cwd": str(dest_dir)})
            if code != 200:
                raise RuntimeError(f"system_status failed: {code} {body}")
            print(body)

            # Logs snapshot
            print_section("Logs snapshot (tail 100)")
            logs_url = f"http://{AGENT_HOST}:{AGENT_PORT}/system_operator/logs"
            code, body = _http_post_json(
                logs_url,
                {
                    "cwd": str(dest_dir),
                    "tail": 100,
                    "no_color": True,
                    "timestamps": True,
                },
                timeout=90.0,
            )
            if code != 200:
                raise RuntimeError(f"system_logs failed: {code} {body}")
            payload = json.loads(body)
            print(payload.get("stdout", ""))

            # Logs stream (best-effort)
            print_section("Logs stream (~8s)")
            qs = urlencode(
                {
                    "cwd": str(dest_dir),
                    "tail": 50,
                    "no_color": "true",
                    "timestamps": "true",
                }
            )
            stream_url = (
                f"http://{AGENT_HOST}:{AGENT_PORT}/system_operator/logs/stream?{qs}"
            )
            stream_logs(stream_url, duration_seconds=8.0)

            # HTTP health probe (best-effort; may still be building deps)
            print_section("HTTP health probe :3000 (best-effort)")
            health_url = f"http://{AGENT_HOST}:{AGENT_PORT}/system_operator/http_health"
            try:
                code, body = _http_post_json(
                    health_url,
                    {"host": "127.0.0.1", "port": 3000, "path": "/", "timeout": 2},
                    timeout=5.0,
                )
                print(body)
            except Exception as e:
                print(f"HTTP health check error: {e}")

            print("\nSystem Operator demo completed.")
            time.sleep(300)

        finally:
            # Stop environment (best-effort)
            try:
                stop_url = f"http://{AGENT_HOST}:{AGENT_PORT}/system_operator/stop"
                _http_post_json(
                    stop_url,
                    {"cwd": str(dest_dir), "down": True, "remove_orphans": True},
                )
            except Exception:
                pass
            terminate_process(agent_proc, name="devops-agent")

    finally:
        terminate_process(registry_proc, name="registry")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
