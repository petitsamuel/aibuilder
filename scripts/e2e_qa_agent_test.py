import os
import sys
import time
import json
import signal
import tempfile
import subprocess
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_HOST = os.getenv("AGENT_REGISTRY_HOST", "127.0.0.1")
REGISTRY_PORT = int(os.getenv("AGENT_REGISTRY_PORT", "8000"))
QA_HOST = os.getenv("QA_AGENT_HOST", "127.0.0.1")
QA_PORT = int(os.getenv("QA_AGENT_PORT", "8004"))
APP_HOST = os.getenv("TEST_APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("TEST_APP_PORT", "38080"))


def _http_get(url: str, timeout: float = 2.0) -> tuple[int, str]:
    req = Request(url, headers={"User-Agent": "e2e-qa-test"})
    with urlopen(req, timeout=timeout) as resp:
        code = resp.getcode()
        body = resp.read().decode("utf-8", errors="ignore")
        return code, body


def _http_post_json(url: str, payload: dict, timeout: float = 5.0) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "e2e-qa-test",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        code = resp.getcode()
        body = resp.read().decode("utf-8", errors="ignore")
        return code, body


def wait_for_http_ok(
    url: str, timeout_seconds: int = 30, require_json_key: str | None = None
) -> None:
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
        time.sleep(0.25)
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
    registry_cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "agent-registry" / "agentregistry.py"),
    ]
    env = os.environ.copy()
    env.setdefault("AGENT_REGISTRY_HOST", REGISTRY_HOST)
    env.setdefault("AGENT_REGISTRY_PORT", str(REGISTRY_PORT))
    proc = subprocess.Popen(
        registry_cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def start_qa_agent(registry_url: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "qaagent.qa",
    ]
    env = os.environ.copy()
    env.setdefault("REGISTRY_URL", registry_url)
    env.setdefault("AGENT_HOST", QA_HOST)
    env.setdefault("AGENT_PORT", str(QA_PORT))
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def start_sample_app() -> subprocess.Popen:
    """Start a tiny Starlette app exposing /api/tasks, /api/ping and /openapi.json."""
    code = f"""
import json
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.requests import Request
import uvicorn

async def create_task(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {{}}
    desc = data.get("description")
    if not isinstance(desc, str):
        return JSONResponse({{"error": "description_required"}}, status_code=400)
    return JSONResponse({{"id": 1, "description": desc}}, status_code=201)

async def ping(_: Request):
    return JSONResponse({{"ok": True}})

async def openapi(_: Request):
    spec = {{
        "openapi": "3.0.0",
        "info": {{"title": "Test API", "version": "1.0.0"}},
        "paths": {{
            "/api/tasks": {{
                "post": {{
                    "responses": {{
                        "201": {{
                            "description": "Created",
                            "content": {{"application/json": {{"schema": {{"type": "object"}}}}}},
                        }},
                        "400": {{"description": "Bad Request"}}
                    }}
                }}
            }},
            "/api/ping": {{
                "get": {{
                    "responses": {{
                        "200": {{"description": "OK"}}
                    }}
                }}
            }}
        }}
    }}
    return JSONResponse(spec)

app = Starlette(routes=[
    Route("/api/tasks", create_task, methods=["POST"]),
    Route("/api/ping", ping, methods=["GET"]),
    Route("/openapi.json", openapi, methods=["GET"]),
])

uvicorn.run(app, host="{APP_HOST}", port={APP_PORT}, log_level="info")
"""
    cmd = [sys.executable, "-c", code]
    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def run_case(name: str, payload: dict, expect_ok: bool) -> None:
    qa_url = f"http://{QA_HOST}:{QA_PORT}/execute_task"
    code, body = _http_post_json(qa_url, payload, timeout=10.0)
    if code != 200:
        raise AssertionError(f"{name}: QA agent responded with status {code}: {body}")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise AssertionError(f"{name}: QA agent returned non-JSON: {e}\nBody: {body}")
    ok = bool(data.get("ok"))
    if ok != expect_ok:
        raise AssertionError(
            f"{name}: expected ok={expect_ok} but got ok={ok}. Summary={data.get('summary')} Checks={data.get('checks')}"
        )
    print(f"[PASS] {name}: ok={ok}")


def main() -> int:
    registry_url = f"http://{REGISTRY_HOST}:{REGISTRY_PORT}"

    # Start Registry
    print("Starting registry...")
    registry_proc = start_registry()
    try:
        wait_for_http_ok(
            f"{registry_url}/status", timeout_seconds=30, require_json_key="agents"
        )

        print("Starting QA agent...")
        # Start QA Agent
        qa_proc = start_qa_agent(registry_url)
        try:
            wait_for_http_ok(f"http://{QA_HOST}:{QA_PORT}/health", timeout_seconds=30)

            # Wait until agent appears in registry (via heartbeat)
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    _, body = _http_get(f"{registry_url}/status")
                    data = json.loads(body)
                    agents = data.get("agents") or []
                    names = {a.get("agent_name") for a in agents if isinstance(a, dict)}
                    if "qa-agent" in names:
                        break
                except Exception:
                    pass
                time.sleep(0.3)
            else:
                raise RuntimeError("QA agent did not register in time")

            # Start Sample App
            print("Starting sample app...")
            app_proc = start_sample_app()
            try:
                wait_for_http_ok(
                    f"http://{APP_HOST}:{APP_PORT}/api/ping",
                    timeout_seconds=30,
                    require_json_key="ok",
                )

                # --- Case 1: Success (POST /api/tasks) ---
                payload_ok = {
                    "base_url": f"http://{APP_HOST}:{APP_PORT}",
                    "endpoint": "/api/tasks",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"description": "test task"},
                    "expect": {
                        "status": [201],
                        "json_fields": ["id", "description"],
                        "json_contains": {"description": "test task"},
                    },
                    "openapi_url": f"http://{APP_HOST}:{APP_PORT}/openapi.json",
                }
                run_case("success_create_task", payload_ok, expect_ok=True)

                # --- Case 2: Failure (wrong method) ---
                payload_fail_status = {
                    "base_url": f"http://{APP_HOST}:{APP_PORT}",
                    "endpoint": "/api/tasks",
                    "method": "GET",
                    "expect": {"status": [200]},
                    "openapi_url": f"http://{APP_HOST}:{APP_PORT}/openapi.json",
                }
                run_case(
                    "fail_wrong_method_status", payload_fail_status, expect_ok=False
                )

                # --- Case 3: Failure (json_contains mismatch) ---
                payload_fail_json = {
                    "base_url": f"http://{APP_HOST}:{APP_PORT}",
                    "endpoint": "/api/ping",
                    "method": "GET",
                    "expect": {"json_contains": {"missing": True}},
                }
                run_case(
                    "fail_json_contains_mismatch", payload_fail_json, expect_ok=False
                )

            finally:
                terminate_process(app_proc, name="sample_app")

        finally:
            terminate_process(qa_proc, name="qa_agent")

    finally:
        terminate_process(registry_proc, name="registry")

    print("\nE2E QA agent test completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
