from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from dotenv import load_dotenv

load_dotenv()


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
                    full_result TEXT,
                    tool_name TEXT,
                    tool_input TEXT
                )
                """
            )
            # Ensure meta key-value store exists
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

            # Migrate history table to include new columns if missing
            info = conn.execute("PRAGMA table_info(history)").fetchall()
            existing_cols = {row[1] for row in info}
            if "tool_name" not in existing_cols:
                cur.execute("ALTER TABLE history ADD COLUMN tool_name TEXT")
            if "tool_input" not in existing_cols:
                cur.execute("ALTER TABLE history ADD COLUMN tool_input TEXT")
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

    def add_plan_item(
        self,
        description: str,
        status: str = "pending",
        parent_task_id: Optional[int] = None,
    ) -> int:
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
        *,
        tool_name: Optional[str] = None,
        tool_input: Optional[str] = None,
    ) -> int:
        conn = self._connect()
        try:
            ts = datetime.now(timezone.utc).isoformat()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO history (timestamp, agent_name, task_description, result_summary, full_result, tool_name, tool_input) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    agent_name,
                    task_description,
                    result_summary,
                    full_result,
                    tool_name,
                    tool_input,
                ),
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

    # Simple key-value meta storage for orchestrator state
    def set_meta(self, key: str, value: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            conn.commit()
        finally:
            conn.close()

    def get_meta(self, key: str) -> Optional[str]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row[0] if row else None
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


def list_project_files(
    root: Path, max_files: int = 200, max_depth: int = 3
) -> List[str]:
    ignored_dirs = {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
    }
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
        f"- {h.timestamp} | {h.agent_name} | {h.result_summary or _trim(h.full_result, 120)}"
        for h in recent
    ]
    tool_lines = []
    for t in tools:
        name = t.get("name")
        desc = t.get("description") or ""
        tool_lines.append(f"- {name}: {desc}")

    file_listing = "\n".join(files[:100])

    instruction = (
        "You are the Main Orchestrator. You are a delegator and planner, not an implementer. "
        "Your goal is to deliver the best end-user application that fulfills the user's goal.\n\n"
        "Operate in a strict loop: (1) Assess current plan/state/history/files. "
        "(2) Choose ONE most important next step. (3) Delegate ONLY via available agents. "
        "(4) Record the result and update the plan.\n\n"
        "Never run commands yourself or guess missing details. Always rely on agents:\n"
        "- Tech Stack Selector: pick the best template from the manifest for the goal.\n"
        "- Environment Manager: scaffold a project, manage dependencies, and perform git add/commit.\n"
        "- Coder: make a single, precise code edit; provide exact file path and content; keep changes minimal.\n"
        "- System Operator: start/stop/restart the app, fetch status and logs. Start the server BEFORE QA.\n"
        "- QA Agent: validate behavior by calling HTTP endpoints (optionally via OpenAPI), and report pass/fail.\n\n"
        "For NEW projects: (1) call Tech Stack Selector to choose a template; (2) ask Environment Manager to scaffold "
        "and install dependencies; (3) instruct Environment Manager to git commit the scaffold; (4) start the app via "
        "System Operator; (5) run QA checks; (6) iterate: Coder -> commit -> QA.\n\n"
        "After ANY code change by Coder, you MUST immediately instruct the Environment Manager to git commit with a clear message.\n"
        "Before QA testing, you MUST ensure the app is running via System Operator.\n\n"
        "Return ONLY a compact JSON object specifying the next tool invocation."
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
    # Determine next pending plan item first
    next_item: Optional[PlanItem] = None
    for p in plan:
        if p.status == "pending":
            next_item = p
            break

    # Only consult LLM when there is an actual pending plan item

    if next_item is not None:
        prompt = build_decision_prompt(goal, plan, recent, files, tools)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                print("prompt", prompt)
                print(
                    "openrouter_api_key",
                    os.getenv("OPENROUTER_API_KEY"),
                    "openrouter_model",
                    os.getenv("OPENROUTER_MODEL"),
                )
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-5"),
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a precise planner that outputs strict JSON only.",
                            },
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
                # Normalize and patch missing required arguments
                if isinstance(parsed, dict) and "agent" in parsed:
                    agent_name = str(parsed.get("agent") or "")
                    task_payload = parsed.get("task")
                    if not isinstance(task_payload, dict):
                        task_payload = {}
                    if (
                        "select_tech_stack" in agent_name
                        and "prompt" not in task_payload
                    ):
                        task_payload["prompt"] = goal
                    parsed["task"] = task_payload
                    return parsed
        except Exception as e:
            print(
                f"[orchestrator] LLM decision failed; using heuristics. Error: {e}",
                file=sys.stderr,
            )

    tool_names = [t.get("name", "") for t in tools]
    selected_tool: Optional[str] = None
    arguments: Dict[str, Any] = {}

    def exists_contains(keyword: str) -> Optional[str]:
        for name in tool_names:
            if keyword in name.lower():
                return name
        return None

    if next_item is not None and (
        "stack" in next_item.description.lower()
        or "tech" in next_item.description.lower()
        or "template" in next_item.description.lower()
    ):
        t = (
            exists_contains("select_tech_stack")
            or exists_contains("tech")
            or exists_contains("stack")
        )
        if t:
            selected_tool = t
            arguments = {"prompt": goal}

    if (
        selected_tool is None
        and next_item is not None
        and (
            "scaffold" in next_item.description.lower()
            or "bootstrap" in next_item.description.lower()
        )
    ):
        t = exists_contains("scaffold_project") or exists_contains("scaffold")
        if t:
            selected_tool = t
            arguments = {}

    if (
        selected_tool is None
        and next_item is not None
        and (
            "install" in next_item.description.lower()
            and "depend" in next_item.description.lower()
        )
    ):
        t = exists_contains("install_dependencies") or exists_contains("install")
        if t:
            selected_tool = t
            arguments = {"manager": os.getenv("DEFAULT_PKG_MANAGER", "npm")}

    if (
        selected_tool is None
        and next_item is not None
        and (
            "start" in next_item.description.lower()
            or "system" in next_item.description.lower()
        )
    ):
        t = exists_contains("system_start") or exists_contains("start")
        if t:
            selected_tool = t
            arguments = {"build": False}

    if (
        selected_tool is None
        and next_item
        and "commit" in next_item.description.lower()
    ):
        t = exists_contains("git_command") or exists_contains("git")
        if t:
            selected_tool = t
            arguments = {
                "args": ["commit", "-m", f"Auto-commit for task #{next_item.id}"]
            }

    if (
        selected_tool is None
        and next_item
        and (
            "qa" in next_item.description.lower()
            or "verify" in next_item.description.lower()
            or "test" in next_item.description.lower()
        )
    ):
        t = exists_contains("qa_http_check") or exists_contains("qa")
        if t:
            selected_tool = t
            base_url = os.getenv("APP_BASE_URL", "http://127.0.0.1:3000")
            arguments = {"base_url": base_url, "endpoint": "/"}

    if selected_tool is None and tool_names and next_item is not None:
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
        self.max_failures_per_task: int = int(
            os.getenv("ORCH_MAX_FAILURES_PER_TASK", "2")
        )

    def _get_int_meta(self, key: str, default: int = 0) -> int:
        try:
            v = self.db.get_meta(key)
            return int(v) if v is not None else default
        except Exception:
            return default

    def _set_int_meta(self, key: str, value: int) -> None:
        try:
            self.db.set_meta(key, str(value))
        except Exception:
            pass

    def _patch_task_payload(
        self,
        agent_name: str,
        payload: Dict[str, Any],
        tools: List[Dict[str, Any]],
        recent: List[HistoryItem],
    ) -> Dict[str, Any]:
        name_l = (agent_name or "").lower()
        patched = dict(payload or {})

        # Find tool definition to consult its input schema
        tool_def: Optional[Dict[str, Any]] = None
        for t in tools:
            try:
                if str(t.get("name") or "").lower() == name_l:
                    tool_def = t
                    break
            except Exception:
                continue

        input_schema: Dict[str, Any] = {}
        required_fields: List[str] = []
        properties: Dict[str, Any] = {}
        if isinstance(tool_def, dict):
            input_schema = tool_def.get("inputSchema") or {}
            if isinstance(input_schema, dict):
                req = input_schema.get("required")
                if isinstance(req, list):
                    required_fields = [str(x) for x in req]
                props = input_schema.get("properties")
                if isinstance(props, dict):
                    properties = props

        # Provide default prompt if required by schema
        if "prompt" in required_fields and not isinstance(patched.get("prompt"), str):
            patched["prompt"] = self.goal

        # Provide cwd from persisted meta if required by schema
        if "cwd" in required_fields and (
            not isinstance(patched.get("cwd"), str) or not patched.get("cwd")
        ):
            project_path = self.db.get_meta("project_path")
            if isinstance(project_path, str) and project_path.strip():
                patched["cwd"] = project_path

        # Normalize synonymous directory fields to cwd if schema allows
        if "cwd" in (required_fields or []) or "cwd" in properties:
            for alias in ("directory", "project_directory", "project_path"):
                v = patched.get(alias)
                if isinstance(v, str) and v.strip() and not patched.get("cwd"):
                    patched["cwd"] = v
                    break

        # Provide template_id from meta if supported by schema and not provided
        if "template_id" in properties and not isinstance(
            patched.get("template_id"), str
        ):
            selected_template = self.db.get_meta("selected_template_id")
            if isinstance(selected_template, str) and selected_template.strip():
                patched["template_id"] = selected_template

        # If schema supports project_name/destination_path and neither is provided, set a sensible default
        supports_project_name = "project_name" in properties
        supports_destination = "destination_path" in properties
        has_dest = (
            isinstance(patched.get("destination_path"), str)
            and str(patched.get("destination_path")).strip()
        )
        has_name = (
            isinstance(patched.get("project_name"), str)
            and str(patched.get("project_name")).strip()
        )
        if (supports_project_name or supports_destination) and not (
            has_dest or has_name
        ):
            if supports_project_name:
                patched.setdefault("project_name", "todo-app")

        # If schema supports overwrite and a project_path is already known, default to overwrite True to avoid conflicts
        if "overwrite" in properties and self.db.get_meta("project_path"):
            patched.setdefault("overwrite", True)

        # For install operations: default a manager when neither manager nor args is provided
        if ("install_dependencies" in name_l) and (
            "manager" in properties or "args" in properties
        ):
            has_manager = (
                isinstance(patched.get("manager"), str)
                and patched.get("manager").strip()
            )
            has_args = (
                isinstance(patched.get("args"), list)
                and len(patched.get("args", [])) > 0
            )
            if not (has_manager or has_args):
                patched["manager"] = os.getenv("DEFAULT_PKG_MANAGER", "npm")

        return patched

    @staticmethod
    def _extract_text_from_result(obj: Any) -> Optional[str]:
        # 1) If it's a CallToolResult-like with `.content`
        content = getattr(obj, "content", None)
        if isinstance(content, list) and content:
            texts: list[str] = []
            for blk in content:
                if isinstance(blk, dict):
                    txt = blk.get("text")
                else:
                    txt = getattr(blk, "text", None)
                if isinstance(txt, str) and txt.strip():
                    texts.append(txt.strip())
            if texts:
                return "\n".join(texts).strip()
        # 2) If it's a list of blocks
        if isinstance(obj, list) and obj:
            texts: list[str] = []
            for blk in obj:
                if isinstance(blk, dict):
                    txt = blk.get("text")
                else:
                    txt = getattr(blk, "text", None)
                if isinstance(txt, str) and txt.strip():
                    texts.append(txt.strip())
            if texts:
                return "\n".join(texts).strip()
        # 3) If it's a plain string
        if isinstance(obj, str):
            return obj.strip()
        # 4) If it's a dict with an id or text
        if isinstance(obj, dict):
            if isinstance(obj.get("id"), str):
                return obj.get("id").strip()
            if isinstance(obj.get("text"), str):
                return obj.get("text").strip()
        # 5) Fallback: regex from repr
        s = str(obj)
        m = re.search(r"text=['\"]([^'\"]+)['\"]", s)
        if m:
            return m.group(1).strip()
        return None

    def _ensure_initial_plan(self) -> None:
        if self.db.has_any():
            return
        self.db.add_plan_item(
            "Determine the optimal tech stack/template for the app", status="pending"
        )

    def _has_plan_item_containing(
        self, keyword: str, statuses: Optional[List[str]] = None
    ) -> bool:
        statuses = statuses or ["pending", "in_progress"]
        for p in self.db.get_plan():
            if p.status in statuses and keyword.lower() in p.description.lower():
                return True
        return False

    def _sync_plan_with_state(self, tools: List[Dict[str, Any]]) -> None:
        """Ensure the plan contains the next appropriate tasks based on meta/state."""
        have_select = True if self.db.get_meta("selected_template_id") else False
        have_project = True if self.db.get_meta("project_path") else False
        deps_installed = self.db.get_meta("deps_installed") == "1"
        system_started = self.db.get_meta("system_started") == "1"
        qa_ok = self.db.get_meta("qa_ok") == "1"

        # Reconcile: mark already-satisfied tasks as completed
        try:
            for p in self.db.get_plan():
                d = p.description.lower()
                if p.status in ("completed", "failed"):
                    continue
                if have_select and ("stack" in d or "template" in d):
                    self.db.update_plan_status(p.id, "completed")
                elif have_project and ("scaffold" in d or "bootstrap" in d):
                    self.db.update_plan_status(p.id, "completed")
                elif deps_installed and ("install" in d and "depend" in d):
                    self.db.update_plan_status(p.id, "completed")
                elif system_started and ("start" in d or "system" in d):
                    self.db.update_plan_status(p.id, "completed")
                elif qa_ok and ("qa" in d or "verify" in d or "test" in d):
                    self.db.update_plan_status(p.id, "completed")
        except Exception:
            pass

        # 1) Ensure tech stack selection if not yet chosen
        if not have_select and not self._has_plan_item_containing("stack"):
            self.db.add_plan_item(
                "Select optimal tech stack/template", status="pending"
            )
            return

        # 2) Scaffold after selection
        if (
            have_select
            and not have_project
            and not self._has_plan_item_containing("scaffold")
        ):
            self.db.add_plan_item(
                "Scaffold project from selected template", status="pending"
            )
            return

        # 3) Install dependencies after scaffold
        if (
            have_project
            and not deps_installed
            and not self._has_plan_item_containing("install dependencies")
        ):
            self.db.add_plan_item("Install dependencies in project", status="pending")
            return

        # 4) Commit scaffold changes (optional but recommended)
        if (
            have_project
            and deps_installed
            and not self._has_plan_item_containing("commit")
        ):
            self.db.add_plan_item("Commit current changes", status="pending")
            return

        # 5) Start the system
        if (
            have_project
            and deps_installed
            and not system_started
            and not self._has_plan_item_containing("start")
        ):
            self.db.add_plan_item("Start system services", status="pending")
            return

        # 6) QA smoke test
        if (
            have_project
            and deps_installed
            and system_started
            and not qa_ok
            and not self._has_plan_item_containing("qa")
        ):
            self.db.add_plan_item("Run QA smoke tests", status="pending")
            return

    async def one_turn(self) -> None:
        print("one_turn")
        self._ensure_initial_plan()

        plan = self.db.get_plan()
        recent = self.db.recent_history(limit=10)
        files = list_project_files(self.project_root, max_files=200, max_depth=3)

        async with MCPRegistryClient(self.registry_url) as mcp:
            tools = await mcp.list_tools()
            print("tools", tools)
            # Sync plan with current state before deciding
            self._sync_plan_with_state(tools)
            plan = self.db.get_plan()
            decision = await decide_next_action(self.goal, plan, recent, files, tools)
            print("[orch] decision", decision)
            agent_name = decision.get("agent")
            task_payload = decision.get("task") or {}
            # Patch minimal payload defaults based on tool definitions
            task_payload = self._patch_task_payload(
                agent_name, task_payload, tools, recent
            )
            print(f"[orch] call: agent={agent_name} payload={task_payload}")

            # If the selected tool expects template_id/path but they're missing, call the Tech Stack Selector tool now
            try:
                tool_def: Optional[Dict[str, Any]] = None
                for t in tools:
                    try:
                        if (
                            str(t.get("name") or "").lower()
                            == (agent_name or "").lower()
                        ):
                            tool_def = t
                            break
                    except Exception:
                        continue
                props = (
                    (tool_def.get("inputSchema", {}) or {}).get("properties", {})
                    if isinstance(tool_def, dict)
                    else {}
                )
                needs_template = isinstance(props, dict) and (
                    "template_id" in props or "template_path" in props
                )
                missing_template = needs_template and not (
                    isinstance(task_payload.get("template_id"), str)
                    or isinstance(task_payload.get("template_path"), str)
                )
                if missing_template:
                    # Find a selection tool
                    sel_tool: Optional[str] = None
                    for t in tools:
                        name = (t.get("name") or "").lower()
                        if "select_tech_stack" in name:
                            sel_tool = t.get("name")
                            break
                    if sel_tool:
                        try:
                            sel_result = await mcp.call_tool(
                                sel_tool, {"prompt": self.goal}
                            )
                            chosen_text = (
                                self._extract_text_from_result(sel_result) or ""
                            )
                            if chosen_text:
                                chosen_id = (
                                    chosen_text.strip()
                                    .splitlines()[0]
                                    .strip()
                                    .strip('"')
                                    .strip("'")
                                )
                                if chosen_id:
                                    task_payload["template_id"] = chosen_id
                                    # Persist selected template for future steps
                                    try:
                                        self.db.set_meta(
                                            "selected_template_id", chosen_id
                                        )
                                    except Exception:
                                        pass
                        except Exception as e:
                            # Proceed without template (agent will respond with an error which we will record)
                            print(f"[orchestrator] select_tech_stack failed: {e}")
            except Exception:
                pass
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
            print(f"[orch] result: success={success} agent={agent_name}")

            # If this tool likely set/created a project path, persist it (prefer outputs, then inputs)
            try:
                tool_def: Optional[Dict[str, Any]] = None
                for t in tools:
                    try:
                        if (
                            str(t.get("name") or "").lower()
                            == (agent_name or "").lower()
                        ):
                            tool_def = t
                            break
                    except Exception:
                        continue
                props = (
                    (tool_def.get("inputSchema", {}) or {}).get("properties", {})
                    if isinstance(tool_def, dict)
                    else {}
                )
                if success and isinstance(props, dict):
                    # Prefer parsing a JSON body with destination if present
                    try:
                        if isinstance(result, (dict, str)):
                            rj = (
                                result
                                if isinstance(result, dict)
                                else json.loads(result)
                            )
                            dest = (
                                rj.get("destination") if isinstance(rj, dict) else None
                            )
                            if isinstance(dest, str) and dest.strip():
                                self.db.set_meta("project_path", dest)
                    except Exception:
                        pass
                    # Fallback to inputs
                    if not self.db.get_meta("project_path"):
                        if (
                            isinstance(task_payload.get("destination_path"), str)
                            and "destination_path" in props
                        ):
                            self.db.set_meta(
                                "project_path",
                                str(task_payload.get("destination_path")),
                            )
                        elif (
                            isinstance(task_payload.get("project_name"), str)
                            and "project_name" in props
                        ):
                            self.db.set_meta(
                                "project_path", str(task_payload.get("project_name"))
                            )
            except Exception:
                pass

            # If tech stack was selected directly, persist the chosen template id
            try:
                if (
                    success
                    and isinstance(agent_name, str)
                    and "select_tech_stack" in agent_name.lower()
                ):
                    chosen_text = self._extract_text_from_result(result)
                    if not chosen_text and isinstance(result, (dict, str)):
                        try:
                            parsed = (
                                result
                                if isinstance(result, dict)
                                else json.loads(result)
                            )
                            if isinstance(parsed, dict) and isinstance(
                                parsed.get("id"), str
                            ):
                                chosen_text = parsed.get("id")
                        except Exception:
                            pass
                    if isinstance(chosen_text, str) and chosen_text.strip():
                        chosen_id = (
                            chosen_text.strip()
                            .splitlines()[0]
                            .strip()
                            .strip('"')
                            .strip("'")
                        )
                        if chosen_id:
                            self.db.set_meta("selected_template_id", chosen_id)
            except Exception:
                pass

            if isinstance(result, list):
                try:
                    texts = [
                        blk.get("text", "") for blk in result if isinstance(blk, dict)
                    ]
                    result_text = "\n".join(filter(None, texts)) or json.dumps(result)
                except Exception:
                    result_text = json.dumps(result)
            elif isinstance(result, (dict, str)):
                result_text = result if isinstance(result, str) else json.dumps(result)
            else:
                result_text = str(result)

            # Generic success override if a JSON body with top-level ok=false is detected
            try:
                parsed = (
                    json.loads(result_text) if isinstance(result_text, str) else None
                )
                if isinstance(parsed, dict) and isinstance(parsed.get("ok"), bool):
                    success = parsed.get("ok")
            except Exception:
                pass

            summary = result_text[:200].replace("\n", " ")
            self.db.add_history(
                agent_name=(
                    agent_name.split(".")[0] if "." in agent_name else agent_name
                ),
                task_description=f"Call {agent_name} with payload {json.dumps(task_payload)}",
                result_summary=summary,
                full_result=result_text,
                tool_name=agent_name,
                tool_input=json.dumps(task_payload),
            )

            if isinstance(plan_task_id, int):
                self.db.update_plan_status(
                    plan_task_id, "completed" if success else "failed"
                )

            # Failure tracking and halt condition
            try:
                key: Optional[str] = None
                if isinstance(plan_task_id, int):
                    key = f"failures.plan.{plan_task_id}"
                elif isinstance(agent_name, str) and agent_name:
                    key = f"failures.agent.{agent_name}"

                if key:
                    if success:
                        self._set_int_meta(key, 0)
                    else:
                        cnt = self._get_int_meta(key, 0) + 1
                        self._set_int_meta(key, cnt)
                        if cnt >= self.max_failures_per_task:
                            # Signal orchestrator to halt further work
                            self.db.set_meta("halt", "1")
            except Exception:
                pass

            # Update meta flags based on successful tool invocation
            try:
                tool_name_l = (agent_name or "").lower()
                if success:
                    if "install_dependencies" in tool_name_l:
                        self.db.set_meta("deps_installed", "1")
                    if "system_start" in tool_name_l or "/start" in tool_name_l:
                        self.db.set_meta("system_started", "1")
                    if "qa_http_check" in tool_name_l:
                        self.db.set_meta("qa_ok", "1")
                else:
                    # Reset flags cautiously on failure
                    if "qa_http_check" in tool_name_l:
                        self.db.set_meta("qa_ok", "0")
            except Exception:
                pass

    async def run(self, continuous: bool, interval_seconds: int) -> None:
        if continuous:
            print("[orchestrator] Starting continuous loop...")
            while True:
                # Stop if a prior turn signaled halt
                try:
                    if self.db.get_meta("halt") == "1":
                        print("[orchestrator] Halting due to repeated failures.")
                        break
                except Exception:
                    pass
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
        default=_env(
            "ORCH_DB_PATH", str(Path("./orchestrator_state.sqlite").resolve())
        ),
        help="Path to SQLite database for plan and history",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=_env("PROJECT_ROOT", str(Path.cwd().resolve())),
        help="Root directory of the project to scan",
    )
    parser.add_argument(
        "--continuous", action="store_true", help="Run in a continuous loop"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(_env("ORCH_INTERVAL", "10")),
        help="Seconds between turns in continuous mode",
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
