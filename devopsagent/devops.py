from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from pathlib import Path
import shutil
import yaml
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


def _lookup_template_path(template_id: Optional[str], template_path: Optional[str]) -> Path:
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

    if not isinstance(template_id, (str, type(None))) or not isinstance(template_path, (str, type(None))):
        return JSONResponse({"error": "invalid_template_fields"}, status_code=422)

    if destination_path and not isinstance(destination_path, str):
        return JSONResponse({"error": "destination_path_must_be_string"}, status_code=422)
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
        return JSONResponse({"error": "template_dir_not_found", "template_dir": str(src_dir)}, status_code=404)

    if dest_dir.exists() and any(dest_dir.iterdir()) and not overwrite:
        return JSONResponse({"error": "destination_exists", "destination": str(dest_dir)}, status_code=409)

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        ignore = shutil.ignore_patterns("node_modules", ".git", "__pycache__")
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True, ignore=ignore)
    except Exception as e:
        return JSONResponse({"error": "copy_failed", "details": str(e)}, status_code=500)

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
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
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
        return JSONResponse({"error": "manager_not_allowed", "manager": manager}, status_code=422)

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
        return JSONResponse({"error": "executable_not_allowed", "executable": cmd[0]}, status_code=422)

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


routes = [
    Route("/scaffold_project", scaffold_project, methods=["POST"]),
    Route("/install_dependencies", install_dependencies, methods=["POST"]),
    Route("/git_command", git_command, methods=["POST"]),
]

def _make_registration_payload(agent_address: str) -> Dict[str, Any]:
    return {
        "agent_name": AGENT_NAME,
        "agent_address": agent_address,
        "capabilities": {
            "role": "environment_manager",
            "version": "0.1.0",
            "endpoints": [
                "/scaffold_project",
                "/install_dependencies",
                "/git_command",
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
                            "timeout": {"type": "integer", "minimum": 1, "default": 1800},
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
                            "timeout": {"type": "integer", "minimum": 1, "default": 600},
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
