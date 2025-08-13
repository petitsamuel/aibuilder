from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from pathlib import Path
import shutil
import yaml
import httpx
import json
import contextlib
from agentkit.base import AgentApp


# ------------------------------
# Configuration (env-overridable)
# ------------------------------
AGENT_NAME = os.getenv("AGENT_NAME", "devops-agent")
AGENT_HOST = os.getenv("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8002"))
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://127.0.0.1:8000")
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "60"))

# Restrict file operations to this workspace root
WORKSPACE_ROOT = os.getenv(
    "WORKSPACE_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
TEMPLATES_DIR = os.getenv(
    "TEMPLATES_DIR",
    os.path.join(WORKSPACE_ROOT, "templates"),
)
MANIFEST_PATH = os.getenv(
    "TEMPLATE_MANIFEST",
    os.path.join(TEMPLATES_DIR, "manifest.yaml"),
)


def _agent_address() -> str:
    return f"http://{AGENT_HOST}:{AGENT_PORT}"


def _resolve_path(path_like: str, base: Optional[str] = None) -> Path:
    base_dir = Path(base or WORKSPACE_ROOT).resolve()
    p = Path(path_like)
    abs_path = (base_dir / p).resolve() if not p.is_absolute() else p.resolve()
    # Ensure path is within workspace root
    workspace = Path(WORKSPACE_ROOT).resolve()
    try:
        abs_path.relative_to(workspace)
    except ValueError:
        raise ValueError("path_outside_workspace")
    return abs_path


def _load_manifest() -> dict:
    manifest_file = Path(MANIFEST_PATH)
    if not manifest_file.exists():
        raise FileNotFoundError("manifest_not_found")
    data = yaml.safe_load(manifest_file.read_text()) or {}
    return data


def _lookup_template_path(
    template_id: Optional[str], template_path: Optional[str]
) -> Path:
    if template_path:
        # If relative path, resolve against templates dir
        try:
            return _resolve_path(template_path, base=TEMPLATES_DIR)
        except ValueError:
            # If provided absolute path is already within workspace, allow it
            p = Path(template_path).resolve()
            workspace = Path(WORKSPACE_ROOT).resolve()
            try:
                p.relative_to(workspace)
            except ValueError:
                raise ValueError("template_path_outside_workspace")
            return p

    if not template_id:
        raise ValueError("template_required")

    manifest = _load_manifest()
    templates = manifest.get("templates") or []
    for entry in templates:
        if str(entry.get("id")) == template_id:
            raw_path = entry.get("path")
            if not raw_path:
                raise ValueError("template_path_missing_in_manifest")
            return _resolve_path(raw_path, base=TEMPLATES_DIR)

    raise ValueError("template_id_not_found")


async def scaffold_project(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    template_id = body.get("template_id")
    template_path = body.get("template_path")
    destination_path = body.get("destination_path")
    project_name = body.get("project_name")
    overwrite = bool(body.get("overwrite", False))

    if not isinstance(template_id, (str, type(None))) or not isinstance(
        template_path, (str, type(None))
    ):
        return JSONResponse({"error": "invalid_template_fields"}, status_code=422)

    if destination_path and not isinstance(destination_path, str):
        return JSONResponse(
            {"error": "destination_path_must_be_string"}, status_code=422
        )
    if project_name and not isinstance(project_name, str):
        return JSONResponse({"error": "project_name_must_be_string"}, status_code=422)

    if not destination_path and not project_name:
        return JSONResponse({"error": "destination_required"}, status_code=422)

    try:
        src_dir = _lookup_template_path(template_id, template_path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    if destination_path:
        try:
            dest_dir = _resolve_path(destination_path, base=WORKSPACE_ROOT)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=422)
    else:
        try:
            dest_dir = _resolve_path(project_name, base=WORKSPACE_ROOT)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=422)

    if not src_dir.exists() or not src_dir.is_dir():
        return JSONResponse(
            {"error": "template_dir_not_found", "template_dir": str(src_dir)},
            status_code=404,
        )

    if dest_dir.exists() and any(dest_dir.iterdir()) and not overwrite:
        return JSONResponse(
            {"error": "destination_exists", "destination": str(dest_dir)},
            status_code=409,
        )

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        ignore = shutil.ignore_patterns("node_modules", ".git", "__pycache__")
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True, ignore=ignore)
    except Exception as e:
        return JSONResponse(
            {"error": "copy_failed", "details": str(e)}, status_code=500
        )

    return JSONResponse(
        {
            "status": "ok",
            "source": str(src_dir),
            "destination": str(dest_dir),
            "overwrite": overwrite,
        }
    )


async def _run_process(args: List[str], cwd: Optional[str], timeout: int) -> dict:
    # Ensure cwd is within workspace
    working_dir: Optional[Path] = None
    if cwd:
        working_dir = _resolve_path(cwd, base=WORKSPACE_ROOT)
        if not working_dir.exists() or not working_dir.is_dir():
            raise FileNotFoundError("cwd_not_found")

    env = os.environ.copy()
    env.setdefault("CI", "1")
    env.setdefault("GIT_TERMINAL_PROMPT", "0")

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(working_dir) if working_dir else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        raise TimeoutError("process_timeout")

    stdout = stdout_bytes.decode(errors="replace")
    stderr = stderr_bytes.decode(errors="replace")

    # Limit output size
    max_len = 200_000
    if len(stdout) > max_len:
        stdout = stdout[-max_len:]
    if len(stderr) > max_len:
        stderr = stderr[-max_len:]

    return {
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


_ALLOWED_PACKAGE_MANAGERS = {"npm", "yarn", "pnpm", "pip", "pip3", "uv", "poetry"}


async def install_dependencies(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    cwd = body.get("cwd") or body.get("project_path")
    manager = body.get("manager")  # e.g., npm | yarn | pnpm | pip | uv | poetry
    args = body.get("args")  # optional list of args to append
    timeout = int(body.get("timeout", 1800))  # default 30 minutes

    if not isinstance(cwd, str):
        return JSONResponse({"error": "cwd_required"}, status_code=422)
    if manager and not isinstance(manager, str):
        return JSONResponse({"error": "manager_must_be_string"}, status_code=422)
    if args and not (isinstance(args, list) and all(isinstance(x, str) for x in args)):
        return JSONResponse({"error": "args_must_be_list_of_strings"}, status_code=422)

    if manager and manager not in _ALLOWED_PACKAGE_MANAGERS:
        return JSONResponse(
            {"error": "manager_not_allowed", "manager": manager}, status_code=422
        )

    # Construct default command if only manager provided
    cmd: List[str]
    if manager in {"npm", "yarn", "pnpm"}:
        cmd = [manager, "install"]
    elif manager in {"pip", "pip3"}:
        cmd = [manager, "install", "-r", "requirements.txt"]
    elif manager == "uv":
        cmd = ["uv", "pip", "install", "-r", "requirements.txt"]
    elif manager == "poetry":
        cmd = ["poetry", "install"]
    else:
        # If no manager provided, require explicit args (first arg must be allowed)
        if not args or not len(args):
            return JSONResponse({"error": "provide_manager_or_args"}, status_code=422)
        cmd = args

    if args and manager:
        # Append extra args only when the manager determined the base command
        cmd.extend([str(a) for a in args])

    # Validate the executable
    if cmd[0] not in _ALLOWED_PACKAGE_MANAGERS:
        return JSONResponse(
            {"error": "executable_not_allowed", "executable": cmd[0]}, status_code=422
        )

    try:
        result = await _run_process(cmd, cwd=cwd, timeout=timeout)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"status": "ok", "command": cmd, **result})


async def git_command(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    cwd = body.get("cwd")
    args = body.get("args")  # e.g., ["add", "."] or ["commit", "-m", "msg"]
    timeout = int(body.get("timeout", 600))

    if not isinstance(cwd, str):
        return JSONResponse({"error": "cwd_required"}, status_code=422)
    if not (isinstance(args, list) and all(isinstance(x, str) for x in args)):
        return JSONResponse({"error": "args_must_be_list_of_strings"}, status_code=422)

    cmd = ["git", *args]

    try:
        result = await _run_process(cmd, cwd=cwd, timeout=timeout)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"status": "ok", "command": cmd, **result})


# ------------------------------
# System Operator (docker compose)
# ------------------------------

_DEFAULT_COMPOSE_FILENAMES = (
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
)


def _resolve_compose_file(cwd: str, compose_file: Optional[str]) -> Optional[Path]:
    base_dir = _resolve_path(cwd, base=WORKSPACE_ROOT)
    if compose_file:
        return _resolve_path(compose_file, base=str(base_dir))
    for name in _DEFAULT_COMPOSE_FILENAMES:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return None


def _compose_cmd(compose_file: Optional[Path]) -> List[str]:
    cmd = ["docker", "compose"]
    if compose_file is not None:
        cmd.extend(["-f", str(compose_file)])
    return cmd


async def system_start(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    cwd = body.get("cwd")
    compose_file = body.get("compose_file")
    service = body.get("service")
    build = bool(body.get("build", False))
    extra_args = body.get("extra_args")
    timeout = int(body.get("timeout", 600))

    if not isinstance(cwd, str):
        return JSONResponse({"error": "cwd_required"}, status_code=422)
    if compose_file and not isinstance(compose_file, str):
        return JSONResponse({"error": "compose_file_must_be_string"}, status_code=422)
    if service and not isinstance(service, str):
        return JSONResponse({"error": "service_must_be_string"}, status_code=422)
    if extra_args and not (
        isinstance(extra_args, list) and all(isinstance(x, str) for x in extra_args)
    ):
        return JSONResponse(
            {"error": "extra_args_must_be_list_of_strings"}, status_code=422
        )

    try:
        compose_path = _resolve_compose_file(cwd, compose_file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    cmd = _compose_cmd(compose_path)
    cmd.extend(["up", "-d"])  # detached, non-blocking
    if build:
        cmd.append("--build")
    if service:
        cmd.append(service)
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = await _run_process(cmd, cwd=cwd, timeout=timeout)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"status": "ok", "command": cmd, **result})


async def system_stop(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    cwd = body.get("cwd")
    compose_file = body.get("compose_file")
    service = body.get("service")
    down = bool(body.get("down", False))
    remove_volumes = bool(body.get("remove_volumes", False))
    remove_orphans = bool(body.get("remove_orphans", True))
    timeout = int(body.get("timeout", 600))

    if not isinstance(cwd, str):
        return JSONResponse({"error": "cwd_required"}, status_code=422)
    if compose_file and not isinstance(compose_file, str):
        return JSONResponse({"error": "compose_file_must_be_string"}, status_code=422)
    if service and not isinstance(service, str):
        return JSONResponse({"error": "service_must_be_string"}, status_code=422)

    try:
        compose_path = _resolve_compose_file(cwd, compose_file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    cmd = _compose_cmd(compose_path)
    if down:
        cmd.append("down")
        if remove_volumes:
            cmd.append("-v")
        if remove_orphans:
            cmd.append("--remove-orphans")
    else:
        cmd.append("stop")
        if service:
            cmd.append(service)

    try:
        result = await _run_process(cmd, cwd=cwd, timeout=timeout)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"status": "ok", "command": cmd, **result})


async def system_restart(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    cwd = body.get("cwd")
    compose_file = body.get("compose_file")
    service = body.get("service")
    timeout = int(body.get("timeout", 600))

    if not isinstance(cwd, str):
        return JSONResponse({"error": "cwd_required"}, status_code=422)
    if compose_file and not isinstance(compose_file, str):
        return JSONResponse({"error": "compose_file_must_be_string"}, status_code=422)
    if service and not isinstance(service, str):
        return JSONResponse({"error": "service_must_be_string"}, status_code=422)

    try:
        compose_path = _resolve_compose_file(cwd, compose_file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    cmd = _compose_cmd(compose_path)
    cmd.append("restart")
    if service:
        cmd.append(service)

    try:
        result = await _run_process(cmd, cwd=cwd, timeout=timeout)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"status": "ok", "command": cmd, **result})


async def system_status(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    cwd = body.get("cwd")
    compose_file = body.get("compose_file")
    service = body.get("service")
    timeout = int(body.get("timeout", 300))

    if not isinstance(cwd, str):
        return JSONResponse({"error": "cwd_required"}, status_code=422)
    if compose_file and not isinstance(compose_file, str):
        return JSONResponse({"error": "compose_file_must_be_string"}, status_code=422)
    if service and not isinstance(service, str):
        return JSONResponse({"error": "service_must_be_string"}, status_code=422)

    try:
        compose_path = _resolve_compose_file(cwd, compose_file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    # Prefer structured output
    cmd_json = _compose_cmd(compose_path) + ["ps", "--format", "json"]
    result_json = await _run_process(cmd_json, cwd=cwd, timeout=timeout)
    running_services: List[str] = []
    services: List[Dict[str, Any]] = []
    parsed_ok = False
    if result_json["returncode"] == 0:
        try:
            data = json.loads(result_json["stdout"] or "[]")
            if isinstance(data, list):
                for entry in data:
                    name = entry.get("Service") or entry.get("Name")
                    state = entry.get("State") or entry.get("Status")
                    services.append({"service": name, "state": state, "raw": entry})
                    if state and str(state).lower().startswith("running"):
                        running_services.append(name)
                parsed_ok = True
        except Exception:
            parsed_ok = False

    if not parsed_ok:
        # Fallback to plain text output
        cmd = _compose_cmd(compose_path) + ["ps"]
        result = await _run_process(cmd, cwd=cwd, timeout=timeout)
        return JSONResponse(
            {
                "status": "ok",
                "command": cmd,
                "stdout": result.get("stdout"),
                "stderr": result.get("stderr"),
                "returncode": result.get("returncode"),
            }
        )

    status_payload: Dict[str, Any] = {
        "status": "ok",
        "command": cmd_json,
        "services": services,
        "running_services": running_services,
    }
    if service:
        status_payload["service_running"] = service in running_services
    return JSONResponse(status_payload)


async def system_logs_snapshot(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    cwd = body.get("cwd")
    compose_file = body.get("compose_file")
    service = body.get("service")
    tail = int(body.get("tail", 200))
    timestamps = bool(body.get("timestamps", True))
    no_color = bool(body.get("no_color", True))
    timeout = int(body.get("timeout", 600))

    if not isinstance(cwd, str):
        return JSONResponse({"error": "cwd_required"}, status_code=422)
    if compose_file and not isinstance(compose_file, str):
        return JSONResponse({"error": "compose_file_must_be_string"}, status_code=422)
    if service and not isinstance(service, str):
        return JSONResponse({"error": "service_must_be_string"}, status_code=422)

    try:
        compose_path = _resolve_compose_file(cwd, compose_file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    cmd = _compose_cmd(compose_path) + ["logs", "--tail", str(tail)]
    if timestamps:
        cmd.append("--timestamps")
    if no_color:
        cmd.append("--no-color")
    if service:
        cmd.append(service)

    try:
        result = await _run_process(cmd, cwd=cwd, timeout=timeout)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"status": "ok", "command": cmd, **result})


async def _compose_logs_stream_generator(
    cwd: str,
    compose_path: Optional[Path],
    service: Optional[str],
    tail: int,
    timestamps: bool,
    no_color: bool,
):
    cmd = _compose_cmd(compose_path) + ["logs", "-f", "--tail", str(tail)]
    if timestamps:
        cmd.append("--timestamps")
    if no_color:
        cmd.append("--no-color")
    if service:
        cmd.append(service)

    env = os.environ.copy()
    env.setdefault("CI", "1")
    env.setdefault("GIT_TERMINAL_PROMPT", "0")

    working_dir = _resolve_path(cwd, base=WORKSPACE_ROOT)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(working_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )

    try:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield line
    finally:
        with contextlib.suppress(ProcessLookupError):
            if proc.returncode is None:
                proc.kill()


async def system_logs_stream(request: Request) -> Response:
    params = request.query_params
    cwd = params.get("cwd")
    compose_file = params.get("compose_file")
    service = params.get("service")
    tail = int(params.get("tail", "200"))
    timestamps = params.get("timestamps", "true").lower() != "false"
    no_color = params.get("no_color", "true").lower() != "false"

    if not isinstance(cwd, str):
        return JSONResponse({"error": "cwd_required"}, status_code=422)
    if compose_file and not isinstance(compose_file, str):
        return JSONResponse({"error": "compose_file_must_be_string"}, status_code=422)
    if service and not isinstance(service, str):
        return JSONResponse({"error": "service_must_be_string"}, status_code=422)

    try:
        compose_path = _resolve_compose_file(cwd, compose_file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    generator = _compose_logs_stream_generator(
        cwd=cwd,
        compose_path=compose_path,
        service=service,
        tail=tail,
        timestamps=timestamps,
        no_color=no_color,
    )
    return StreamingResponse(generator, media_type="text/plain")


async def http_health(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    url = body.get("url")
    host = body.get("host", "127.0.0.1")
    port = body.get("port")
    path = body.get("path", "/")
    timeout_s = float(body.get("timeout", 3))

    if url and not isinstance(url, str):
        return JSONResponse({"error": "url_must_be_string"}, status_code=422)
    if not url:
        if not isinstance(port, (int, str)):
            return JSONResponse({"error": "port_required"}, status_code=422)
        url = f"http://{host}:{int(port)}{path}"

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url)
        return JSONResponse(
            {
                "status": "ok",
                "url": url,
                "http_status": resp.status_code,
                "ok": resp.status_code < 500,
            }
        )
    except Exception as e:
        return JSONResponse(
            {"status": "error", "url": url, "error": str(e)}, status_code=200
        )


routes = [
    Route("/scaffold_project", scaffold_project, methods=["POST"]),
    Route("/install_dependencies", install_dependencies, methods=["POST"]),
    Route("/git_command", git_command, methods=["POST"]),
    # System Operator
    Route("/system_operator/start", system_start, methods=["POST"]),
    Route("/system_operator/stop", system_stop, methods=["POST"]),
    Route("/system_operator/restart", system_restart, methods=["POST"]),
    Route("/system_operator/status", system_status, methods=["POST"]),
    Route("/system_operator/logs", system_logs_snapshot, methods=["POST"]),
    Route("/system_operator/logs/stream", system_logs_stream, methods=["GET"]),
    Route("/system_operator/http_health", http_health, methods=["POST"]),
]


def _make_registration_payload(agent_address: str) -> Dict[str, Any]:
    return {
        "agent_name": AGENT_NAME,
        "agent_address": agent_address,
        "capabilities": {
            "role": "environment_manager",
            "version": "0.2.0",
            "endpoints": [
                "/scaffold_project",
                "/install_dependencies",
                "/git_command",
                "/system_operator/start",
                "/system_operator/stop",
                "/system_operator/restart",
                "/system_operator/status",
                "/system_operator/logs",
                "/system_operator/logs/stream",
                "/system_operator/http_health",
            ],
            "mcp_tools": [
                {
                    "name": "scaffold_project",
                    "description": "Copy a template directory into a destination to create a new project.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "template_id": {"type": ["string", "null"]},
                            "template_path": {"type": ["string", "null"]},
                            "destination_path": {"type": ["string", "null"]},
                            "project_name": {"type": ["string", "null"]},
                            "overwrite": {"type": "boolean", "default": False},
                        },
                        "additionalProperties": False,
                    },
                    "_meta": {
                        "http": {
                            "endpoint": "/scaffold_project",
                            "method": "POST",
                            "returnType": "json",
                        }
                    },
                },
                {
                    "name": "install_dependencies",
                    "description": "Run a package manager install command in a project directory (npm, yarn, pnpm, pip, uv, poetry).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "cwd": {"type": "string"},
                            "manager": {"type": ["string", "null"]},
                            "args": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                            },
                            "timeout": {
                                "type": "integer",
                                "minimum": 1,
                                "default": 1800,
                            },
                        },
                        "required": ["cwd"],
                        "additionalProperties": False,
                    },
                    "_meta": {
                        "http": {
                            "endpoint": "/install_dependencies",
                            "method": "POST",
                            "returnType": "json",
                        }
                    },
                },
                {
                    "name": "git_command",
                    "description": "Run a git command in the specified directory (e.g., add/commit/push).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "cwd": {"type": "string"},
                            "args": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "timeout": {
                                "type": "integer",
                                "minimum": 1,
                                "default": 600,
                            },
                        },
                        "required": ["cwd", "args"],
                        "additionalProperties": False,
                    },
                    "_meta": {
                        "http": {
                            "endpoint": "/git_command",
                            "method": "POST",
                            "returnType": "json",
                        }
                    },
                },
                {
                    "name": "system_start",
                    "description": "Start the application using docker compose in detached mode.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "cwd": {"type": "string"},
                            "compose_file": {"type": ["string", "null"]},
                            "service": {"type": ["string", "null"]},
                            "build": {"type": "boolean", "default": False},
                            "extra_args": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                            },
                            "timeout": {
                                "type": "integer",
                                "minimum": 1,
                                "default": 600,
                            },
                        },
                        "required": ["cwd"],
                        "additionalProperties": False,
                    },
                    "_meta": {
                        "http": {
                            "endpoint": "/system_operator/start",
                            "method": "POST",
                            "returnType": "json",
                        }
                    },
                },
                {
                    "name": "system_stop",
                    "description": "Stop or bring down the application using docker compose.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "cwd": {"type": "string"},
                            "compose_file": {"type": ["string", "null"]},
                            "service": {"type": ["string", "null"]},
                            "down": {"type": "boolean", "default": False},
                            "remove_volumes": {"type": "boolean", "default": False},
                            "remove_orphans": {"type": "boolean", "default": True},
                            "timeout": {
                                "type": "integer",
                                "minimum": 1,
                                "default": 600,
                            },
                        },
                        "required": ["cwd"],
                        "additionalProperties": False,
                    },
                    "_meta": {
                        "http": {
                            "endpoint": "/system_operator/stop",
                            "method": "POST",
                            "returnType": "json",
                        }
                    },
                },
                {
                    "name": "system_restart",
                    "description": "Restart the application service using docker compose.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "cwd": {"type": "string"},
                            "compose_file": {"type": ["string", "null"]},
                            "service": {"type": ["string", "null"]},
                            "timeout": {
                                "type": "integer",
                                "minimum": 1,
                                "default": 600,
                            },
                        },
                        "required": ["cwd"],
                        "additionalProperties": False,
                    },
                    "_meta": {
                        "http": {
                            "endpoint": "/system_operator/restart",
                            "method": "POST",
                            "returnType": "json",
                        }
                    },
                },
                {
                    "name": "system_status",
                    "description": "Return docker compose status for running services.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "cwd": {"type": "string"},
                            "compose_file": {"type": ["string", "null"]},
                            "service": {"type": ["string", "null"]},
                            "timeout": {
                                "type": "integer",
                                "minimum": 1,
                                "default": 300,
                            },
                        },
                        "required": ["cwd"],
                        "additionalProperties": False,
                    },
                    "_meta": {
                        "http": {
                            "endpoint": "/system_operator/status",
                            "method": "POST",
                            "returnType": "json",
                        }
                    },
                },
                {
                    "name": "system_logs",
                    "description": "Fetch recent docker compose logs (stdout/stderr).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "cwd": {"type": "string"},
                            "compose_file": {"type": ["string", "null"]},
                            "service": {"type": ["string", "null"]},
                            "tail": {"type": "integer", "minimum": 0, "default": 200},
                            "timestamps": {"type": "boolean", "default": True},
                            "no_color": {"type": "boolean", "default": True},
                            "timeout": {
                                "type": "integer",
                                "minimum": 1,
                                "default": 600,
                            },
                        },
                        "required": ["cwd"],
                        "additionalProperties": False,
                    },
                    "_meta": {
                        "http": {
                            "endpoint": "/system_operator/logs",
                            "method": "POST",
                            "returnType": "json",
                        }
                    },
                },
                {
                    "name": "http_health",
                    "description": "Perform an HTTP GET to check if the development server responds.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": {"type": ["string", "null"]},
                            "host": {"type": "string", "default": "127.0.0.1"},
                            "port": {"type": ["integer", "string", "null"]},
                            "path": {"type": "string", "default": "/"},
                            "timeout": {"type": ["integer", "number"], "default": 3},
                        },
                        "additionalProperties": False,
                    },
                    "_meta": {
                        "http": {
                            "endpoint": "/system_operator/http_health",
                            "method": "POST",
                            "returnType": "json",
                        }
                    },
                },
            ],
        },
    }


_agent = AgentApp(
    agent_name=AGENT_NAME,
    host=AGENT_HOST,
    port=AGENT_PORT,
    registry_url=REGISTRY_URL,
    heartbeat_interval=HEARTBEAT_INTERVAL,
    extra_routes=routes,
    make_registration_payload=_make_registration_payload,
)

app = _agent.app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=AGENT_HOST, port=AGENT_PORT, log_level="info")
