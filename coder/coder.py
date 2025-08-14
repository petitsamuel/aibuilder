from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, PlainTextResponse
from starlette.routing import Route

from pathlib import Path
import json
import re
import httpx

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agentkit import AgentApp
from dotenv import load_dotenv

load_dotenv()


# ------------------------------
# Configuration (env-overridable)
# ------------------------------
AGENT_NAME = os.getenv("AGENT_NAME", "coder")
AGENT_HOST = os.getenv("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8003"))
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://127.0.0.1:8000")
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "60"))

# ------------------------------
# OpenRouter configuration
# ------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5-mini")
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.3"))
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "32000"))
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "aibuilder-coder")


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        # remove leading fence line and trailing fence
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    # Also try to extract the largest fenced block if present elsewhere
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            # content is typically in parts[1]
            inner = parts[1]
            # If language tag present on first line, drop it
            first_nl = inner.find("\n")
            if first_nl != -1 and len(inner[:first_nl].strip()) <= 20:
                inner = inner[first_nl + 1 :]
            return inner.strip()
    return s


def _detect_framework(file_path: str, file_content: str) -> str:
    """
    Best-effort, lightweight framework detection based only on the provided
    file path and content. Returns a short framework key used to choose the
    corresponding system instruction markdown file.
    """
    path_lower = (file_path or "").lower()
    content_lower = (file_content or "").lower()

    # Explicit override via env
    env_override = os.getenv("CODER_FRAMEWORK", "").strip().lower()
    if env_override:
        return env_override

    # Heuristics (extendable)
    # Next.js / React (App Router often under src/app or app/)
    if any(seg in path_lower for seg in ["/src/app/", "/app/", "/pages/"]):
        if any(ext in path_lower for ext in [".tsx", ".ts", ".jsx", ".js"]):
            return "nextjs"
    if 'from "next' in content_lower or "next/router" in content_lower:
        return "nextjs"

    # Generic React/JS/TS
    if any(path_lower.endswith(ext) for ext in [".tsx", ".jsx"]):
        return "react"

    # Python web frameworks
    if any(path_lower.endswith(ext) for ext in [".py"]):
        if "flask" in content_lower:
            return "flask"
        if "fastapi" in content_lower:
            return "fastapi"
        if "django" in content_lower:
            return "django"
        return "python"

    # Node/JS (non-React)
    if any(path_lower.endswith(ext) for ext in [".js", ".ts"]):
        if "express" in content_lower:
            return "express"
        return "node"

    return "default"


def _load_system_instructions(framework_key: str) -> str:
    """
    Load per-framework system instructions from a markdown file located next to
    this module. Falls back to a default file or a minimal built-in string.
    """
    base_dir = Path(__file__).resolve().parent
    # Order of attempts: exact framework file -> default file -> built-in
    candidates = [
        base_dir / f"si_{framework_key}.md",
        base_dir / "si_default.md",
    ]
    for path in candidates:
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                continue
    # Built-in minimal fallback
    return (
        "You are a world-class software developer. Prefer returning a unified diff for the single target file. "
        "If a diff is impractical, return the full, updated file content with no explanations."
    )


def _extract_unified_diff(text: str) -> Optional[str]:
    """
    Try to extract a unified diff from the model output.
    Accepts either fenced blocks (```diff ...```), or raw unified diff that
    includes --- / +++ headers and @@ hunks. Returns the diff text or None.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    s = text.strip()

    print("Extracting unified diff")
    print(s)

    # If the content is inside code fences, try to extract the largest one
    inner = _strip_code_fences(s)
    # Heuristic: if stripping fences removed them, prefer inner; otherwise s
    candidate_texts = [inner, s] if inner != s else [s]

    for candidate in candidate_texts:
        lines = candidate.strip().splitlines()
        if not lines:
            continue
        # Quick check for unified diff markers
        has_header = any(l.startswith("--- ") for l in lines) and any(
            l.startswith("+++ ") for l in lines
        )
        has_hunk = any(l.lstrip().startswith("@@") for l in lines)
        if has_header and has_hunk:
            # Trim leading/trailing non-diff chatter if present
            start_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("--- "):
                    start_idx = i
                    break
            # Return from first header line
            return "\n".join(lines[start_idx:]).strip()
    print("No diff found")
    return None


def _apply_unified_diff(original_text: str, diff_text: str, expected_path: str) -> str:
    """
    Apply a single-file unified diff to the provided original_text.
    Supports two modes:
      1) Standard unified diff hunks with line ranges ("@@ -x,y +a,b @@").
      2) Fuzzy hunks where the hunk header is just "@@" and lines may omit
         the leading space for context. In this mode we locate the original
         block by content and replace it.
    Raises ValueError on failure so the caller can fall back to full-file mode.
    """
    had_trailing_nl = original_text.endswith("\n")
    diff_lines = [ln.rstrip("\n\r") for ln in diff_text.strip().splitlines()]

    # Determine if we have standard numeric hunks
    has_numeric_hunk = any(re.match(r"^@@ -\d+", ln.lstrip()) for ln in diff_lines)

    if has_numeric_hunk:
        # Standard unified diff application (range-based)
        orig_lines = original_text.split("\n")
        idx = 0
        # Skip headers/preamble
        while idx < len(diff_lines) and not diff_lines[idx].lstrip().startswith("@@"):
            idx += 1
        if idx >= len(diff_lines):
            raise ValueError("No hunk found in diff")

        result_lines: List[str] = []
        orig_index_1based = 1
        hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

        while idx < len(diff_lines):
            line = diff_lines[idx]
            header_match = hunk_re.match(line)
            if not header_match:
                # Reached something unexpected; stop processing
                break
            idx += 1

            orig_start = int(header_match.group(1))

            # Copy unchanged lines before this hunk
            pre_copy_count = max(0, orig_start - orig_index_1based)
            if pre_copy_count > 0:
                start_zero = orig_index_1based - 1
                result_lines.extend(
                    orig_lines[start_zero : start_zero + pre_copy_count]
                )
                orig_index_1based += pre_copy_count

            # Apply hunk body
            while idx < len(diff_lines) and not diff_lines[idx].lstrip().startswith(
                "@@"
            ):
                line = diff_lines[idx]
                if not line:
                    result_lines.append("")
                    idx += 1
                    continue
                prefix = line[0]
                payload = line[1:] if len(line) > 0 else ""
                if prefix == " ":
                    if orig_index_1based - 1 < len(orig_lines):
                        result_lines.append(orig_lines[orig_index_1based - 1])
                    else:
                        result_lines.append(payload)
                    orig_index_1based += 1
                elif prefix == "-":
                    orig_index_1based += 1
                elif prefix == "+":
                    result_lines.append(payload)
                elif prefix == "\\":
                    pass
                else:
                    # Treat unknown marker as context
                    result_lines.append(line)
                idx += 1

        # Append remainder
        if orig_index_1based - 1 < len(orig_lines):
            result_lines.extend(orig_lines[orig_index_1based - 1 :])

        updated_text = "\n".join(result_lines)
        if had_trailing_nl and not updated_text.endswith("\n"):
            updated_text += "\n"
        return updated_text

    # Fallback: fuzzy application based on content within hunks
    current_text = original_text

    def _apply_fuzzy_hunk(text: str, hunk_body_lines: List[str]) -> str:
        original_block_lines: List[str] = []
        new_block_lines: List[str] = []
        for raw in hunk_body_lines:
            if not raw:
                original_block_lines.append("")
                new_block_lines.append("")
                continue
            marker = raw[0]
            payload = raw[1:] if len(raw) > 0 else ""
            if marker == " ":
                original_block_lines.append(payload)
                new_block_lines.append(payload)
            elif marker == "-":
                original_block_lines.append(payload)
            elif marker == "+":
                new_block_lines.append(payload)
            elif marker == "\\":
                continue
            else:
                original_block_lines.append(raw)
                new_block_lines.append(raw)

        original_block = "\n".join(original_block_lines)
        new_block = "\n".join(new_block_lines)

        pos = text.find(original_block)
        if pos == -1:
            removed_only = "\n".join(
                [ln[1:] for ln in hunk_body_lines if ln and ln[0] == "-"]
            )
            if removed_only:
                pos = text.find(removed_only)
                if pos != -1:
                    end_pos = pos + len(removed_only)
                    return text[:pos] + new_block + text[end_pos:]
            raise ValueError("Failed to locate hunk position for fuzzy apply")

        end_pos = pos + len(original_block)
        return text[:pos] + new_block + text[end_pos:]

    # Iterate hunks and apply sequentially
    i = 0
    while i < len(diff_lines) and not diff_lines[i].lstrip().startswith("@@"):
        i += 1
    if i >= len(diff_lines):
        raise ValueError("No hunk found in diff (fuzzy mode)")

    while i < len(diff_lines):
        if not diff_lines[i].lstrip().startswith("@@"):
            i += 1
            continue
        i += 1  # move past '@@'
        hunk_body: List[str] = []
        while i < len(diff_lines) and not diff_lines[i].lstrip().startswith("@@"):
            if diff_lines[i].startswith("--- ") or diff_lines[i].startswith("+++ "):
                i += 1
                continue
            hunk_body.append(diff_lines[i])
            i += 1

        current_text = _apply_fuzzy_hunk(current_text, hunk_body)

    if had_trailing_nl and not current_text.endswith("\n"):
        current_text += "\n"
    return current_text


def _maybe_apply_diff(
    file_path: str, original_text: str, model_output: str
) -> Optional[str]:
    """
    If the model returned a unified diff, apply it and return the updated text.
    Otherwise, return None to indicate the caller should fall back to full-file mode.
    """
    diff_text = _extract_unified_diff(model_output)
    if not diff_text:
        print("No diff found")
        return None
    try:
        return _apply_unified_diff(original_text, diff_text, file_path)
    except Exception:
        import traceback

        print("Failed to apply diff")
        traceback.print_exc()
        return None


async def _openrouter_chat(messages: List[Dict[str, str]]) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-Title"] = OPENROUTER_APP_NAME

    body: Dict[str, Any] = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": OPENROUTER_TEMPERATURE,
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        # OpenRouter follows OpenAI-like schema
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(content, str):
            content = str(content)
        return content


async def execute_task(request: Request) -> Response:
    print("Executing task!")
    # Input: { "task": str, "file_path": str, "file_content": str }
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    task = body.get("task")
    file_path = body.get("file_path")
    file_content = body.get("file_content")

    if not isinstance(task, str) or not task.strip():
        return JSONResponse({"error": "task_required"}, status_code=422)
    if not isinstance(file_path, str) or not file_path.strip():
        return JSONResponse({"error": "file_path_required"}, status_code=422)
    if not isinstance(file_content, str):
        return JSONResponse({"error": "file_content_required"}, status_code=422)

    # Load per-framework instructions from local markdown files
    framework_key = _detect_framework(file_path, file_content)
    system_prompt = _load_system_instructions(framework_key)

    user_prompt = (
        f"File path: {file_path}\n"
        f"Task: {task}\n\n"
        f"File content (do not rewrite unrelated parts):\n<BEGIN_FILE>\n{file_content}\n<END_FILE>\n\n"
        f"Output preference: Provide a unified diff that transforms the given file. "
        f"If a diff is impractical, return ONLY the complete updated file content with no explanations."
    )

    try:
        print("Sending request to OpenRouter")
        print(system_prompt)
        print(user_prompt)
        content = await _openrouter_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        print("Received response from OpenRouter")
        print(content)
    except httpx.HTTPStatusError as e:
        return JSONResponse(
            {
                "error": "openrouter_http_error",
                "status": e.response.status_code,
                "message": str(e),
            },
            status_code=502,
        )
    except Exception as e:
        return JSONResponse(
            {"error": "openrouter_error", "message": str(e)}, status_code=500
        )

    # First try to interpret the response as a unified diff and apply it
    print("Applying diff")
    updated = _maybe_apply_diff(file_path, file_content, content)
    print("Updated", updated)
    if updated is None:
        # Fall back to interpreting the output as full updated file content
        updated = _strip_code_fences(content)
        if not isinstance(updated, str) or not updated.strip():
            return JSONResponse({"error": "empty_model_output"}, status_code=502)

    return PlainTextResponse(updated)


routes = [
    Route("/execute_task", execute_task, methods=["POST"]),
]


def _make_registration_payload(agent_address: str) -> Dict[str, Any]:
    return {
        "agent_name": AGENT_NAME,
        "agent_address": agent_address,
        "capabilities": {
            "role": "coder",
            "endpoints": ["/execute_task"],
            "version": "0.1.0",
            "mcp_tools": [
                {
                    "name": "apply_code_task",
                    "description": "Use an LLM via OpenRouter to apply a precise code edit to a single provided file. Returns full updated file content as text.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "file_path": {"type": "string"},
                            "file_content": {"type": "string"},
                        },
                        "required": ["task", "file_path", "file_content"],
                        "additionalProperties": False,
                    },
                    "_meta": {
                        "http": {
                            "endpoint": "/execute_task",
                            "method": "POST",
                            "returnType": "text",
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
