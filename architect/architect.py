from __future__ import annotations

import os
import sys
import re
import json
from typing import Any, Dict, List, Optional, Set

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, PlainTextResponse
from starlette.routing import Route

from pathlib import Path
import yaml
import httpx

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agentkit import AgentApp
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# Configuration (env-overridable)
# ------------------------------
AGENT_NAME = os.getenv("AGENT_NAME", "tech-stack-selector")
AGENT_HOST = os.getenv("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8001"))
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://127.0.0.1:8000")
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "60"))

# Workspace and manifest paths (mirror devops agent behavior)
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

# Optional OpenRouter LLM configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5-nano")
ARCHITECT_USE_LLM = os.getenv("ARCHITECT_USE_LLM", "false").lower() in {
    "1",
    "true",
    "yes",
}
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "aibuilder-tech-stack-selector")


# ------------------------------
# Manifest loading and selection logic
# ------------------------------


def _load_manifest() -> Dict[str, Any]:
    manifest_file = Path(MANIFEST_PATH)
    if not manifest_file.exists():
        raise FileNotFoundError("manifest_not_found")
    data = yaml.safe_load(manifest_file.read_text()) or {}
    return data


def _get_all_templates() -> List[Dict[str, Any]]:
    manifest = _load_manifest()
    templates = manifest.get("templates") or []
    # Normalize structure just in case
    normalized: List[Dict[str, Any]] = []
    for entry in templates:
        if not isinstance(entry, dict):
            continue
        if not entry.get("id"):
            continue
        normalized.append(
            {
                "id": str(entry.get("id")),
                "name": entry.get("name") or str(entry.get("id")),
                "path": entry.get("path"),
                "tags": list(entry.get("tags") or []),
                "description": entry.get("description") or "",
                "system_instructions": entry.get("system_instructions") or {},
            }
        )
    return normalized


def _filter_templates_by_ids(
    candidate_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    all_templates = _get_all_templates()
    if not candidate_ids:
        return all_templates
    candidate_set = {str(x) for x in candidate_ids}
    filtered = [t for t in all_templates if t.get("id") in candidate_set]
    return filtered


_SYNONYMS: Dict[str, Set[str]] = {
    "react": {"react"},
    "nextjs": {"nextjs", "next", "next.js"},
    "tailwind": {"tailwind", "tailwindcss"},
    "vue": {"vue", "vuejs", "vue.js"},
    "express": {"express", "expressjs", "express.js"},
    "nodejs": {"node", "nodejs", "node.js", "javascript", "js"},
    "fastapi": {"fastapi", "python"},
    "postgres": {"postgres", "postgresql", "psql"},
    "sqlite": {"sqlite", "sql"},
    "supabase": {"supabase", "auth", "authentication", "realtime", "real-time"},
    "static": {"static", "static site", "frontend only", "frontend-only", "no backend"},
    "api": {"api", "backend", "server"},
    "database": {"database", "db", "persistence"},
    "seo": {"seo", "search", "search engine"},
}


def _normalize_tokens(text: str) -> Set[str]:
    if not text:
        return set()
    # Lowercase and split on non-word characters
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z0-9\.\-\+]+", lowered))
    # Expand common bigrams/phrases
    normalized: Set[str] = set(tokens)
    phrases = [
        ("next.js", "nextjs"),
        ("front-end", "frontend"),
        ("front end", "frontend"),
        ("back-end", "backend"),
        ("back end", "backend"),
        ("real time", "real-time"),
    ]
    for a, b in phrases:
        if a in lowered:
            normalized.add(b)
    return normalized


def _score_template(prompt: str, template: Dict[str, Any]) -> float:
    prompt_tokens = _normalize_tokens(prompt)

    t_id = str(template.get("id", ""))
    t_name = str(template.get("name", ""))
    t_tags = [str(x).lower() for x in (template.get("tags") or [])]
    t_desc = str(template.get("description") or "").lower()

    score = 0.0

    # 1) Tag direct matches
    for tag in t_tags:
        if tag in prompt_tokens:
            score += 5.0

    # 2) Synonym matches that map to tags present in the template
    for canonical, syns in _SYNONYMS.items():
        if canonical in t_tags and any(s in prompt_tokens for s in syns):
            score += 3.0

    # 3) Direct mention in template id or name
    t_surface = f"{t_id} {t_name}".lower()
    for token in list(prompt_tokens):
        if token and token in t_surface:
            score += 1.5

    # 4) Description keyword hints
    # Reward if the prompt contains words present in the description
    # (light weight to keep tags primary)
    desc_hits = sum(1 for tok in prompt_tokens if tok in t_desc)
    score += min(desc_hits, 10) * 0.3

    # 5) Heuristics: prefer static site for static-only requests
    wants_static = any(s in prompt_tokens for s in _SYNONYMS["static"])
    has_api = ("api" in t_tags) or ("backend" in t_tags)
    if wants_static and has_api:
        score -= 3.0
    if wants_static and ("static" in t_tags):
        score += 2.0

    # Database desires
    wants_db = any(s in prompt_tokens for s in _SYNONYMS["database"]) or any(
        s in prompt_tokens for s in {"postgres", "sqlite", "supabase"}
    )
    if wants_db and any(
        tag in t_tags for tag in ["sqlite", "postgres", "supabase", "database"]
    ):
        score += 2.0
    if not wants_db and any(
        tag in t_tags for tag in ["database", "postgres", "sqlite", "supabase"]
    ):
        score -= 0.5

    # Language/framework hints
    wants_python = "python" in prompt_tokens
    wants_node = any(s in prompt_tokens for s in _SYNONYMS["nodejs"])
    if wants_python and "fastapi" in t_tags:
        score += 2.0
    if wants_node and any(tag in t_tags for tag in ["react", "express", "nextjs"]):
        score += 1.5

    return score


def _select_template_deterministic(
    prompt: str, templates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    # Break ties by manifest order
    scored = [(_score_template(prompt, t), idx, t) for idx, t in enumerate(templates)]
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return scored[0][2]


async def _select_template_llm(
    prompt: str,
    templates: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not OPENROUTER_API_KEY:
        return None
    model_name = model or OPENROUTER_MODEL

    # Keep the context compact to avoid large payloads
    shortlist = [
        {
            "id": t.get("id"),
            "name": t.get("name"),
            "tags": t.get("tags"),
            "description": (t.get("description") or "")[:500],
        }
        for t in templates
    ]

    system_msg = (
        "You are a precise tech stack selector. Given a user prompt and a set of templates, "
        "choose the single best template id. Respond with ONLY the id string, nothing else."
    )
    user_content = {
        "prompt": prompt,
        "templates": shortlist,
        "instructions": "Return only the id of the best template as plain text.",
    }

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_HTTP_REFERER,
            "X-Title": OPENROUTER_TITLE,
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_content)},
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 16,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        text = (
            ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        ).strip()
        text = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", text)
        if not text:
            return None
        for t in templates:
            if str(t.get("id")) == text:
                return t
        return None
    except Exception:
        return None


# ------------------------------
# HTTP Handlers
# ------------------------------


def _coerce_str_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    return None


async def _select_template(
    prompt: str, candidate_ids: Optional[List[str]], use_llm: bool, model: Optional[str]
) -> Dict[str, Any]:
    candidates = _filter_templates_by_ids(candidate_ids)
    if not candidates:
        raise ValueError("no_templates_available")

    # Try LLM first if enabled
    if use_llm:
        picked = await _select_template_llm(prompt, candidates, model=model)
        if picked is not None:
            return picked

    # Fallback to deterministic
    return _select_template_deterministic(prompt, candidates)


async def execute_task(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    prompt = body.get("prompt")
    candidate_ids = _coerce_str_list(body.get("candidate_ids"))
    # Back-compat: "templates" behaves like candidate_ids list of ids
    if candidate_ids is None:
        candidate_ids = _coerce_str_list(body.get("templates"))

    # LLM flags: env default can be overridden per request
    use_llm = bool(body.get("use_llm", ARCHITECT_USE_LLM))
    model = body.get("model") if isinstance(body.get("model"), str) else None

    if not isinstance(prompt, str) or not prompt.strip():
        return JSONResponse({"error": "prompt_required"}, status_code=422)

    try:
        picked = await _select_template(prompt, candidate_ids, use_llm, model)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        return JSONResponse(
            {"error": "selection_failed", "details": str(e)}, status_code=500
        )

    # Output must be only the string id of the chosen template
    return PlainTextResponse(str(picked.get("id")))


routes = [
    Route("/execute_task", execute_task, methods=["POST"]),
]


def _make_registration_payload(agent_address: str) -> Dict[str, Any]:
    return {
        "agent_name": AGENT_NAME,
        "agent_address": agent_address,
        "capabilities": {
            "role": "tech_stack_selector",
            "endpoints": ["/execute_task"],
            "version": "0.2.0",
            "mcp_tools": [
                {
                    "name": "select_tech_stack",
                    "description": "Analyze a prompt and return the best template id. Uses manifest.yaml, optional LLM via OpenRouter.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "candidate_ids": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                            },
                            "templates": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                            },
                            "use_llm": {"type": "boolean", "default": False},
                            "model": {"type": ["string", "null"]},
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
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
