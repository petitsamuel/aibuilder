from __future__ import annotations

import asyncio
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route


class AgentApp:
    def __init__(
        self,
        *,
        agent_name: str,
        host: str | None = None,
        port: int | None = None,
        registry_url: str | None = None,
        heartbeat_interval: int | None = None,
        extra_routes: Optional[List[Route]] = None,
        make_registration_payload: Optional[Callable[[str], Dict[str, Any]]] = None,
        debug: bool = False,
    ) -> None:
        self.agent_name = agent_name or os.getenv("AGENT_NAME", "agent")
        self.host = (host or os.getenv("AGENT_HOST", "127.0.0.1")).strip()
        self.port = int(port or int(os.getenv("AGENT_PORT", "0") or 0) or 0)
        if self.port <= 0:
            # allow per-agent default via env; fallback to 0 (let uvicorn choose)
            self.port = int(os.getenv(f"{self.agent_name.upper().replace('-','_')}_PORT", os.getenv("AGENT_PORT", "8000")))
        self.registry_url = (registry_url or os.getenv("REGISTRY_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.heartbeat_interval = int(heartbeat_interval or int(os.getenv("HEARTBEAT_INTERVAL", "60")))

        self._extra_routes = extra_routes or []
        self._make_registration_payload = make_registration_payload

        self._heartbeat_task: Optional[asyncio.Task] = None

        routes: List[Route] = [
            Route("/health", self._health, methods=["GET"]),
            *self._extra_routes,
        ]
        self.app = Starlette(debug=debug, routes=routes)
        self.app.add_event_handler("startup", self._on_startup)
        self.app.add_event_handler("shutdown", self._on_shutdown)

    def address(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def _health(self, _: Request) -> Response:
        return PlainTextResponse("ok")

    async def _heartbeat_loop(self) -> None:
        payload = self._build_registration_payload()
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                try:
                    await client.post(f"{self.registry_url}/register", json=payload)
                except Exception:
                    pass
                await asyncio.sleep(self.heartbeat_interval)

    def _build_registration_payload(self) -> Dict[str, Any]:
        if self._make_registration_payload is not None:
            return self._make_registration_payload(self.address())
        return {
            "agent_name": self.agent_name,
            "agent_address": self.address(),
            "capabilities": {
                "role": "generic_agent",
                "version": "0.1.0",
                "endpoints": [r.path for r in self._extra_routes],
                "mcp_tools": [],
            },
        }

    async def _on_startup(self) -> None:
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _on_shutdown(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass


