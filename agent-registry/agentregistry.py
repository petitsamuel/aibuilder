from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Tuple

from mcp.server.fastmcp import FastMCP
import httpx
from pydantic import BaseModel, ValidationError, field_validator
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from mcp.types import Tool as MCPTool


# ------------------------------
# Configuration (env-overridable)
# ------------------------------

def _get_int_env(var_names: list[str], default: int) -> int:
    for name in var_names:
        raw = os.getenv(name)
        if raw is None:
            continue
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return default


STALE_THRESHOLD_SECONDS: int = _get_int_env(
    ["AGENT_REGISTRY_STALE_THRESHOLD", "STALE_THRESHOLD_SECONDS"], 120
)
PRUNE_INTERVAL_SECONDS: int = _get_int_env(
    ["AGENT_REGISTRY_PRUNE_INTERVAL", "PRUNE_INTERVAL_SECONDS"], 10
)


# ------------------------------
# Data models
# ------------------------------


class RegistrationPayload(BaseModel):
    agent_name: str
    agent_address: str
    capabilities: Any

    @field_validator("agent_name")
    @classmethod
    def _validate_agent_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("agent_name must be a non-empty string")
        return value.strip()

    @field_validator("agent_address")
    @classmethod
    def _validate_agent_address(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("agent_address must be a non-empty string")
        return value.strip()


@dataclass
class AgentRecord:
    agent_name: str
    agent_address: str
    capabilities: Any
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ------------------------------
# In-memory registry and locking
# ------------------------------


AGENT_REGISTRY: Dict[str, AgentRecord] = {}
REGISTRY_LOCK: asyncio.Lock = asyncio.Lock()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_agent(record: AgentRecord) -> dict[str, Any]:
    age_seconds = max(0.0, (_now_utc() - record.last_seen).total_seconds())
    return {
        "agent_name": record.agent_name,
        "agent_address": record.agent_address,
        "capabilities": record.capabilities,
        "last_seen": record.last_seen.isoformat(),
        "age_seconds": age_seconds,
    }


async def _prune_stale_agents_loop() -> None:
    """Background task: periodically remove agents older than the stale threshold."""
    while True:
        await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
        cutoff = _now_utc() - timedelta(seconds=STALE_THRESHOLD_SECONDS)
        async with REGISTRY_LOCK:
            stale_names = [name for name, rec in AGENT_REGISTRY.items() if rec.last_seen < cutoff]
            for name in stale_names:
                del AGENT_REGISTRY[name]


@asynccontextmanager
async def registry_lifespan(app: FastMCP):
    """Start background prune loop on server start and cancel on shutdown."""
    task = asyncio.create_task(_prune_stale_agents_loop())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


"""
Dynamic MCP aggregation

Agents register with `capabilities` that may include a list of MCP-style tools under
`capabilities.mcp_tools`. Each entry should include at least: name, endpoint, and
optionally title, description, method (default POST), returnType ("text"|"json"),
inputSchema, and outputSchema.

The registry exposes a dynamic list of tools by aggregating these capabilities
and namespacing them as `<agent_name>.<tool_name>` to avoid conflicts.
"""


def _iter_agent_tools() -> Iterable[Tuple[str, AgentRecord, dict[str, Any]]]:
    cutoff = _now_utc() - timedelta(seconds=STALE_THRESHOLD_SECONDS)
    for rec in AGENT_REGISTRY.values():
        if rec.last_seen < cutoff:
            continue
        caps = rec.capabilities if isinstance(rec.capabilities, dict) else {}
        tools: List[dict[str, Any]] = caps.get("mcp_tools", []) or []
        for tool in tools:
            tool_name = tool.get("name")
            if not tool_name:
                continue
            namespaced = f"{rec.agent_name}.{tool_name}"
            yield namespaced, rec, tool


class AgentRegistryMCP(FastMCP):
    async def list_tools(self) -> list[MCPTool]:
        # Start with any statically defined tools on this server
        base_tools = await super().list_tools()
        dynamic_tools: list[MCPTool] = []
        async with REGISTRY_LOCK:
            for namespaced, rec, tool in _iter_agent_tools():
                # Use the MCP Tool schema provided by the agent; namespace the name
                tool_dict = dict(tool)
                tool_dict["name"] = namespaced
                try:
                    dynamic_tools.append(MCPTool.model_validate(tool_dict))
                except Exception:
                    # Fallback minimal mapping if agent-provided schema is incomplete
                    dynamic_tools.append(
                        MCPTool(
                            name=namespaced,
                            title=tool.get("title") or f"{tool.get('name')} ({rec.agent_name})",
                            description=tool.get("description") or f"Tool from {rec.agent_name}",
                            inputSchema=tool.get("inputSchema")
                            or {"type": "object", "additionalProperties": True},
                            outputSchema=tool.get("outputSchema"),
                            annotations=None,
                        )
                    )
        return [*base_tools, *dynamic_tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> List[dict[str, Any]] | dict[str, Any]:
        # If name matches a dynamic capability, proxy the call; otherwise fallback
        target: Tuple[AgentRecord, dict[str, Any]] | None = None
        async with REGISTRY_LOCK:
            for namespaced, rec, tool in _iter_agent_tools():
                if namespaced == name:
                    target = (rec, tool)
                    break

        if target is None:
            # Fallback to regular tool manager
            return await super().call_tool(name, arguments)

        rec, tool = target
        meta = tool.get("_meta") or {}
        http_info = meta.get("http") or {}
        # Backwards compatibility with older capability shape
        endpoint: str = http_info.get("endpoint") or tool.get("endpoint")
        method: str = (http_info.get("method") or tool.get("method") or "POST").upper()
        return_type: str = (http_info.get("returnType") or tool.get("returnType") or "text").lower()

        url = f"{rec.agent_address.rstrip('/')}/{endpoint.lstrip('/')}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "POST":
                resp = await client.post(url, json=arguments)
            elif method == "GET":
                resp = await client.get(url, params=arguments)
            else:
                resp = await client.request(method, url, json=arguments)

        if resp.status_code != 200:
            raise RuntimeError(f"Agent call failed with status {resp.status_code}: {resp.text}")

        if return_type == "json":
            return resp.json()
        # default: convert to text content blocks
        return [{"type": "text", "text": resp.text}]


mcp = AgentRegistryMCP(name="agent-registry", lifespan=registry_lifespan)


@mcp.custom_route("/register", methods=["POST"])
async def register(request: Request) -> Response:
    """Register or heartbeat an agent. Idempotent upsert on `agent_name`."""
    try:
        raw = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    try:
        payload = RegistrationPayload.model_validate(raw)
    except ValidationError as e:
        return JSONResponse({"error": "validation_error", "details": e.errors()}, status_code=422)

    now = _now_utc()
    async with REGISTRY_LOCK:
        existing = payload.agent_name in AGENT_REGISTRY
        if existing:
            rec = AGENT_REGISTRY[payload.agent_name]
            rec.agent_address = payload.agent_address
            rec.capabilities = payload.capabilities
            rec.last_seen = now
        else:
            AGENT_REGISTRY[payload.agent_name] = AgentRecord(
                agent_name=payload.agent_name,
                agent_address=payload.agent_address,
                capabilities=payload.capabilities,
                last_seen=now,
            )

        record = AGENT_REGISTRY[payload.agent_name]

    return JSONResponse(
        {
            "status": "ok",
            "existing": existing,
            "stale_threshold_seconds": STALE_THRESHOLD_SECONDS,
            "record": _serialize_agent(record),
        }
    )


@mcp.custom_route("/status", methods=["GET"])
async def status(_: Request) -> Response:
    """Report current live registry state."""
    async with REGISTRY_LOCK:
        agents = [_serialize_agent(rec) for rec in AGENT_REGISTRY.values()]

    return JSONResponse(
        {
            "now": _now_utc().isoformat(),
            "stale_threshold_seconds": STALE_THRESHOLD_SECONDS,
            "prune_interval_seconds": PRUNE_INTERVAL_SECONDS,
            "agents": agents,
        }
    )


if __name__ == "__main__":
    # Transport and HTTP settings
    mcp.settings.streamable_http_path = "/mcp"
    mcp.settings.mount_path = "/mcp"
    mcp.settings.host = os.getenv("AGENT_REGISTRY_HOST", "127.0.0.1")
    mcp.settings.port = int(os.getenv("AGENT_REGISTRY_PORT", "8000"))
    mcp.run(transport="streamable-http")