from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


@dataclass
class PlanItem:
    id: int
    description: str
    status: str
    parent_task_id: Optional[int]


@dataclass
class HistoryItem:
    id: int
    timestamp: str
    agent_name: str
    task_description: str
    result_summary: str
    full_result: str


class StateStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS plan (
                    id INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending','in_progress','completed','failed')),
                    parent_task_id INTEGER NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    agent_name TEXT NOT NULL,
                    task_description TEXT NOT NULL,
                    result_summary TEXT,
                    full_result TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get_plan(self) -> List[PlanItem]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, description, status, parent_task_id FROM plan ORDER BY id ASC"
            ).fetchall()
            return [
                PlanItem(
                    id=row["id"],
                    description=row["description"],
                    status=row["status"],
                    parent_task_id=row["parent_task_id"],
                )
                for row in rows
            ]
        finally:
            conn.close()

    def add_plan_item(self, description: str, status: str = "pending", parent_task_id: Optional[int] = None) -> int:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO plan (description, status, parent_task_id) VALUES (?, ?, ?)",
                (description, status, parent_task_id),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def update_plan_status(self, plan_id: int, status: str) -> None:
        conn = self._connect()
        try:
            conn.execute("UPDATE plan SET status=? WHERE id=?", (status, plan_id))
            conn.commit()
        finally:
            conn.close()

    def get_next_pending(self) -> Optional[PlanItem]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, description, status, parent_task_id FROM plan WHERE status='pending' ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return PlanItem(
                id=row["id"],
                description=row["description"],
                status=row["status"],
                parent_task_id=row["parent_task_id"],
            )
        finally:
            conn.close()

    def has_any(self) -> bool:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(1) FROM plan").fetchone()
            return bool(row[0])
        finally:
            conn.close()

    def add_history(
        self,
        agent_name: str,
        task_description: str,
        result_summary: str,
        full_result: str,
    ) -> int:
        conn = self._connect()
        try:
            ts = datetime.now(timezone.utc).isoformat()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO history (timestamp, agent_name, task_description, result_summary, full_result) VALUES (?, ?, ?, ?, ?)",
                (ts, agent_name, task_description, result_summary, full_result),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def recent_history(self, limit: int = 10) -> List[HistoryItem]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, timestamp, agent_name, task_description, result_summary, full_result FROM history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                HistoryItem(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    agent_name=row["agent_name"],
                    task_description=row["task_description"],
                    result_summary=row["result_summary"],
                    full_result=row["full_result"],
                )
                for row in rows
            ]
        finally:
            conn.close()


class MCPRegistryClient:
    def __init__(self, registry_url: str) -> None:
        url = registry_url.rstrip("/")
        if not url.endswith("/mcp") and not url.endswith("/mcp/"):
            url = f"{url}/mcp/"
        elif url.endswith("/mcp"):
            url = f"{url}/"
        self._transport = StreamableHttpTransport(url=url)
        self._client: Optional[Client] = None

    async def __aenter__(self) -> "MCPRegistryClient":
        self._client = Client(transport=self._transport)
        await self._client.__aenter__()
        await self._client.ping()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc, tb)
            self._client = None

    async def list_tools(self) -> List[Dict[str, Any]]:
        assert self._client is not None
        tools = await self._client.list_tools()
        normalized: List[Dict[str, Any]] = []
        for t in tools:
            try:
                tool_dict = t.model_dump()  # type: ignore[attr-defined]
            except Exception:
                tool_dict = dict(t)
            normalized.append(tool_dict)
        return normalized

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        assert self._client is not None
        return await self._client.call_tool(name, arguments)


def list_project_files(root: Path, max_files: int = 200, max_depth: int = 3) -> List[str]:
    ignored_dirs = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}
    results: List[str] = []

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth or len(results) >= max_files:
            return
        for entry in sorted(current.iterdir(), key=lambda p: p.name):
            if entry.name in ignored_dirs:
                continue
            rel = str(entry.relative_to(root))
            if entry.is_dir():
                results.append(rel + "/")
                _walk(entry, depth + 1)
            else:
                results.append(rel)
            if len(results) >= max_files:
                break

    _walk(root, 0)
    return results


def _trim(text: str, max_len: int = 800) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_decision_prompt(
    goal: str,
    plan: List[PlanItem],
    recent: List[HistoryItem],
    files: List[str],
    tools: List[Dict[str, Any]],
) -> str:
    plan_lines = [f"- [{p.status}] (#{p.id}) {p.description}" for p in plan]
    hist_lines = [
        f"- {h.timestamp} | {h.agent_name} | {h.result_summary or _trim(h.full_result, 120)}" for h in recent
    ]
    tool_lines = []
    for t in tools:
        name = t.get("name")
        desc = t.get("description") or ""
        tool_lines.append(f"- {name}: {desc}")

    file_listing = "\n".join(files[:100])

    instruction = (
        "You are the Main Orchestrator. Your goal is to build a web application based on the user's request. "
        "You must operate in a loop: 1) Assess the current state of the project and the plan. "
        "2) Choose the single most important next step. 3) Delegate that step to the correct agent. "
        "After the Coder Agent makes changes, you MUST instruct the Environment Manager to commit the code. "
        "After a feature is supposedly complete, you MUST use the System Operator and QA Agent to verify it."
    )

    prompt = (
        f"System Instructions:\n{instruction}\n\n"
        f"Overall Goal:\n{goal}\n\n"
        f"Plan Status:\n" + "\n".join(plan_lines) + "\n\n"
        f"Recent History (last {len(recent)}):\n" + "\n".join(hist_lines) + "\n\n"
        f"Current Files (truncated):\n{file_listing}\n\n"
        f"Available Tools (namespaced):\n" + "\n".join(tool_lines) + "\n\n"
        "Return ONLY a compact JSON object with keys: agent (tool name as exposed), task (object payload), plan_task_id (optional, integer)."
    )
    return prompt


async def decide_next_action(
    goal: str,
    plan: List[PlanItem],
    recent: List[HistoryItem],
    files: List[str],
    tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    openai_key = "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"
    if openai_key:
        prompt = build_decision_prompt(goal, plan, recent, files, tools)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                print("prompt", prompt)
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openai_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": os.getenv("ORCH_OPENAI_MODEL", "gpt-4o-mini"),
                        "messages": [
                            {"role": "system", "content": "You are a precise planner that outputs strict JSON only."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                print("data", data)
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "agent" in parsed and "task" in parsed:
                    return parsed
        except Exception as e:
            print(f"[orchestrator] LLM decision failed; using heuristics. Error: {e}", file=sys.stderr)

    next_item: Optional[PlanItem] = None
    for p in plan:
        if p.status == "pending":
            next_item = p
            break

    tool_names = [t.get("name", "") for t in tools]
    selected_tool: Optional[str] = None
    arguments: Dict[str, Any] = {}

    def exists_contains(keyword: str) -> Optional[str]:
        for name in tool_names:
            if keyword in name.lower():
                return name
        return None

    if (
        next_item is None
        or "stack" in next_item.description.lower()
        or "tech" in next_item.description.lower()
        or "template" in next_item.description.lower()
    ):
        t = exists_contains("select_tech_stack") or exists_contains("tech") or exists_contains("stack")
        if t:
            selected_tool = t
            arguments = {"prompt": goal}

    if selected_tool is None and next_item and "commit" in next_item.description.lower():
        t = exists_contains("commit") or exists_contains("env")
        if t:
            selected_tool = t
            arguments = {"message": f"Auto-commit by orchestrator for task #{next_item.id}: {next_item.description}"}

    if (
        selected_tool is None
        and next_item
        and (
            "qa" in next_item.description.lower()
            or "verify" in next_item.description.lower()
            or "test" in next_item.description.lower()
        )
    ):
        t = exists_contains("qa") or exists_contains("verify") or exists_contains("test")
        if t:
            selected_tool = t
            arguments = {"scope": "smoke"}

    if selected_tool is None and tool_names:
        selected_tool = tool_names[0]
        arguments = {}

    plan_task_id = next_item.id if next_item else None
    return {"agent": selected_tool, "task": arguments, "plan_task_id": plan_task_id}


class Orchestrator:
    def __init__(
        self,
        registry_url: str,
        db_path: str,
        project_root: str,
        goal: str,
    ) -> None:
        self.registry_url = registry_url
        self.db = StateStore(db_path)
        self.project_root = Path(project_root).resolve()
        self.goal = goal

    def _ensure_initial_plan(self) -> None:
        if self.db.has_any():
            return
        self.db.add_plan_item("Determine the optimal tech stack/template for the app", status="pending")
        # self.db.add_plan_item("Bootstrap the project from the chosen template", status="pending")
        # self.db.add_plan_item("Implement the first core feature", status="pending")
        # self.db.add_plan_item("Commit current changes", status="pending")
        # self.db.add_plan_item("Run QA smoke tests", status="pending")

    async def one_turn(self) -> None:
        print("one_turn")
        self._ensure_initial_plan()

        plan = self.db.get_plan()
        recent = self.db.recent_history(limit=10)
        files = list_project_files(self.project_root, max_files=200, max_depth=3)

        async with MCPRegistryClient(self.registry_url) as mcp:
            tools = await mcp.list_tools()
            print("tools", tools)
            decision = await decide_next_action(self.goal, plan, recent, files, tools)
            print("decision", decision)
            agent_name = decision.get("agent")
            task_payload = decision.get("task") or {}
            plan_task_id = decision.get("plan_task_id")

            if not agent_name:
                print("[orchestrator] No agent selected; skipping turn.")
                return

            if isinstance(plan_task_id, int):
                try:
                    self.db.update_plan_status(plan_task_id, "in_progress")
                except Exception:
                    pass

            try:
                result = await mcp.call_tool(agent_name, task_payload)
                success = True
            except Exception as e:
                result = {"error": str(e)}
                success = False

            if isinstance(result, list):
                try:
                    texts = [blk.get("text", "") for blk in result if isinstance(blk, dict)]
                    result_text = "\n".join(filter(None, texts)) or json.dumps(result)
                except Exception:
                    result_text = json.dumps(result)
            elif isinstance(result, (dict, str)):
                result_text = result if isinstance(result, str) else json.dumps(result)
            else:
                result_text = str(result)

            summary = result_text[:200].replace("\n", " ")
            self.db.add_history(
                agent_name=agent_name.split(".")[0] if "." in agent_name else agent_name,
                task_description=f"Call {agent_name} with payload {json.dumps(task_payload)}",
                result_summary=summary,
                full_result=result_text,
            )

            if isinstance(plan_task_id, int):
                self.db.update_plan_status(plan_task_id, "completed" if success else "failed")

            maybe_commit_tool = None
            for t in tools:
                name = (t.get("name") or "").lower()
                if "commit" in name or ("env" in name and "commit" in name):
                    maybe_commit_tool = t.get("name")
                    break
            if maybe_commit_tool:
                have_pending_commit = any(
                    p for p in self.db.get_plan() if p.status == "pending" and "commit" in p.description.lower()
                )
                if not have_pending_commit:
                    self.db.add_plan_item("Commit current changes", status="pending")

    async def run(self, continuous: bool, interval_seconds: int) -> None:
        if continuous:
            print("[orchestrator] Starting continuous loop...")
            while True:
                await self.one_turn()
                await asyncio.sleep(interval_seconds)
        else:
            await self.one_turn()


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def parse_args(argv: List[str]) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="Orchestrator delegator loop")
    parser.add_argument("--goal", type=str, required=True, help="Overall project goal")
    parser.add_argument(
        "--registry-url",
        type=str,
        default=_env("REGISTRY_URL", "http://127.0.0.1:8000"),
        help="Base URL of the Agent Registry MCP server (without /mcp)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=_env("ORCH_DB_PATH", str(Path("./orchestrator_state.sqlite").resolve())),
        help="Path to SQLite database for plan and history",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=_env("PROJECT_ROOT", str(Path.cwd().resolve())),
        help="Root directory of the project to scan",
    )
    parser.add_argument("--continuous", action="store_true", help="Run in a continuous loop")
    parser.add_argument(
        "--interval", type=int, default=int(_env("ORCH_INTERVAL", "10")), help="Seconds between turns in continuous mode"
    )

    ns = parser.parse_args(argv)
    return vars(ns)


async def _amain(argv: List[str]) -> int:
    args = parse_args(argv)
    orch = Orchestrator(
        registry_url=args["registry_url"],
        db_path=args["db_path"],
        project_root=args["project_root"],
        goal=args["goal"],
    )
    await orch.run(continuous=args["continuous"], interval_seconds=args["interval"])
    return 0


def main() -> None:
    try:
        exit_code = asyncio.run(_amain(sys.argv[1:]))
    except KeyboardInterrupt:
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
