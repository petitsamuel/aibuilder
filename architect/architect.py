from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route


# ------------------------------
# Configuration (env-overridable)
# ------------------------------
AGENT_NAME = os.getenv("AGENT_NAME", "tech-stack-selector")
AGENT_HOST = os.getenv("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8001"))
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://127.0.0.1:8000")
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "60"))

DEFAULT_TEMPLATES: List[str] = [
    "react-express-sqlite",
    "vue-fastapi-postgres",
    "nextjs-supabase",
]


def _agent_address() -> str:
    return f"http://{AGENT_HOST}:{AGENT_PORT}"


# ------------------------------
# Selection logic
# ------------------------------


def score_template(prompt: str, template: str) -> int:
    """Deterministic keyword scoring between prompt and template name."""
    p = prompt.lower()
    t = template.lower()

    score = 0
    # Direct matches
    for token in ["react", "express", "sqlite", "vue", "fastapi", "postgres", "next", "nextjs", "supabase"]:
        if token in p and token in t:
            score += 3

    # Synonyms / implications
    if ("node" in p or "javascript" in p or "js" in p) and ("express" in t or "react" in t):
        score += 2
    if ("python" in p) and ("fastapi" in t):
        score += 2
    if ("database" in p or "db" in p or "sql" in p) and ("postgres" in t or "sqlite" in t or "supabase" in t):
        score += 2
    if ("modern frontend" in p or "modern ui" in p or "ssr" in p) and ("next" in t or "nextjs" in t):
        score += 2
    if ("simple" in p or "lightweight" in p) and ("sqlite" in t):
        score += 1
    if ("real-time" in p or "realtime" in p or "auth" in p or "authentication" in p) and ("supabase" in t):
        score += 2

    return score


def select_template(prompt: str, templates: List[str]) -> str:
    if not templates:
        templates = DEFAULT_TEMPLATES
    best = max(templates, key=lambda tpl: (score_template(prompt, tpl), -templates.index(tpl)))
    # Ties are broken by template order due to the secondary key
    return best


# ------------------------------
# HTTP Handlers
# ------------------------------


async def health(_: Request) -> Response:
    return PlainTextResponse("ok")


async def execute_task(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    prompt = body.get("prompt")
    templates = body.get("templates", DEFAULT_TEMPLATES)

    if not isinstance(prompt, str) or not prompt.strip():
        return JSONResponse({"error": "prompt_required"}, status_code=422)
    if not isinstance(templates, list) or not all(isinstance(x, str) for x in templates):
        return JSONResponse({"error": "templates_must_be_list_of_strings"}, status_code=422)

    choice = select_template(prompt, templates)
    # Output must be only the string name of the chosen template
    return PlainTextResponse(choice)


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/execute_task", execute_task, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes)


# ------------------------------
# Heartbeat / Registration
# ------------------------------


_heartbeat_task: asyncio.Task | None = None


async def _heartbeat_loop() -> None:
    payload: Dict[str, Any] = {
        "agent_name": AGENT_NAME,
        "agent_address": _agent_address(),
        "capabilities": {
            "role": "tech_stack_selector",
            "endpoints": ["/execute_task"],
            "version": "0.1.0",
            # Self-described MCP tool(s) this agent offers
            "mcp_tools": [
                {
                    # MCP Tool schema (mcp.types.Tool)
                    "name": "select_tech_stack",
                    "description": "Analyze a prompt and return the best template name.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "templates": {
                                "type": "array",
                                "items": {"type": "string"},
                                "nullable": True,
                            },
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                    # output is plain text string, so omit outputSchema
                    # Transport hint for registry proxy
                    "_meta": {
                        "http": {
                            "endpoint": "/execute_task",
                            "method": "POST",
                            "returnType": "text",
                        }
                    },
                }
            ],
        },
    }

    # Register immediately, then heartbeat periodically
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                await client.post(f"{REGISTRY_URL}/register", json=payload)
            except Exception:
                # Silent retry; registry might be down during startup
                pass
            await asyncio.sleep(HEARTBEAT_INTERVAL)


@app.on_event("startup")
async def _on_startup() -> None:
    global _heartbeat_task
    _heartbeat_task = asyncio.create_task(_heartbeat_loop())


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    global _heartbeat_task
    if _heartbeat_task is not None:
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=AGENT_HOST, port=AGENT_PORT, log_level="info")