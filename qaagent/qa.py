from __future__ import annotations

import os
import sys
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from pathlib import Path
import httpx

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agentkit import AgentApp
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# Configuration (env-overridable)
# ------------------------------
AGENT_NAME = os.getenv("AGENT_NAME", "qa-agent")
AGENT_HOST = os.getenv("AGENT_HOST", "127.0.0.1")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8004"))
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://127.0.0.1:8000")
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "60"))


def _normalize_url(base_url: str, endpoint: str) -> str:
    base = (base_url or "").rstrip("/")
    path = endpoint if (endpoint or "").startswith("/") else f"/{endpoint or ''}"
    return f"{base}{path}"


def _coerce_dict(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return None


def _to_int_list(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        out: List[int] = []
        for v in value:
            try:
                out.append(int(v))
            except Exception:
                return None
        return out
    try:
        return [int(value)]
    except Exception:
        return None


def _safe_json_parse(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _partial_contains(expected: Any, actual: Any) -> bool:
    """
    Return True if 'actual' contains the structure in 'expected'.
    - Dict: every expected key must be present in actual and match recursively
    - List: for each expected element, there must exist some element in actual that matches recursively
    - Scalar: equality
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for k, v in expected.items():
            if k not in actual:
                return False
            if not _partial_contains(v, actual[k]):
                return False
        return True
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        for item in expected:
            if not any(_partial_contains(item, act_item) for act_item in actual):
                return False
        return True
    return expected == actual


def _find_openapi_operation(
    spec: Dict[str, Any], endpoint: str, method: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    paths = (spec or {}).get("paths") or {}
    method_l = (method or "").lower()
    for tmpl, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        # Convert /tasks/{id} -> ^/tasks/[^/]+$
        pattern = re.sub(r"\{[^}/]+\}", r"[^/]+", str(tmpl))
        pattern = f"^{pattern}$"
        try:
            if re.match(pattern, endpoint or ""):
                op = path_item.get(method_l)
                if isinstance(op, dict):
                    return op, path_item
                else:
                    return None, path_item
        except re.error:
            continue
    return None, None


async def execute_task(request: Request) -> Response:
    """
    Input JSON schema:
    {
      "base_url": string,                      // required, e.g. "http://127.0.0.1:3000"
      "endpoint": string,                      // required, e.g. "/api/tasks"
      "method": string,                        // optional, default GET
      "headers": object|null,                  // optional
      "params": object|null,                   // optional query params
      "body": any|null,                        // optional request body; dict implies JSON
      "expect": {                              // optional expectations
        "status": int|int[],                   // expected status code(s)
        "json_fields": string[],               // required top-level JSON fields
        "json_contains": object                // partial JSON that must be contained in response
      }|null,
      "openapi_url": string|null,              // optional URL to fetch an OpenAPI spec JSON
      "openapi_spec": object|null,             // optional inline OpenAPI spec JSON
      "timeout_secs": number                   // optional, default 20
    }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    base_url = body.get("base_url")
    endpoint = body.get("endpoint")
    method = str(body.get("method", "GET")).upper()
    headers = _coerce_dict(body.get("headers")) or {}
    params = _coerce_dict(body.get("params")) or {}
    payload = body.get("body")
    expect = _coerce_dict(body.get("expect")) or {}
    timeout_secs = float(body.get("timeout_secs", 20.0))

    if not isinstance(base_url, str) or not base_url.strip():
        return JSONResponse({"error": "base_url_required"}, status_code=422)
    if not isinstance(endpoint, str) or not endpoint.strip():
        return JSONResponse({"error": "endpoint_required"}, status_code=422)

    openapi_spec: Optional[Dict[str, Any]] = None
    if isinstance(body.get("openapi_spec"), dict):
        openapi_spec = body.get("openapi_spec")
    elif isinstance(body.get("openapi_url"), str) and body.get("openapi_url").strip():
        try:
            async with httpx.AsyncClient(timeout=timeout_secs) as client:
                resp = await client.get(body.get("openapi_url"))
                resp.raise_for_status()
                openapi_spec = resp.json()
        except Exception:
            openapi_spec = None

    url = _normalize_url(base_url, endpoint)

    # Decide json vs data send
    send_json: Optional[Any] = None
    send_data: Optional[Any] = None
    content_type = (
        headers.get("Content-Type") or headers.get("content-type") or ""
    ).lower()
    if isinstance(payload, (dict, list)) or "application/json" in content_type:
        send_json = payload
        headers.setdefault("Content-Type", "application/json")
    elif payload is not None:
        send_data = payload

    # Perform HTTP request
    response = None
    request_error: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=timeout_secs) as client:
            if method == "GET":
                response = await client.get(url, headers=headers, params=params)
            elif method == "POST":
                response = await client.post(
                    url, headers=headers, params=params, json=send_json, data=send_data
                )
            elif method == "PUT":
                response = await client.put(
                    url, headers=headers, params=params, json=send_json, data=send_data
                )
            elif method == "PATCH":
                response = await client.patch(
                    url, headers=headers, params=params, json=send_json, data=send_data
                )
            elif method == "DELETE":
                response = await client.delete(
                    url, headers=headers, params=params, json=send_json, data=send_data
                )
            else:
                return JSONResponse({"error": "unsupported_method"}, status_code=422)
    except httpx.HTTPError as e:
        request_error = f"http_error: {str(e)}"
    except Exception as e:
        request_error = f"error: {str(e)}"

    checks: List[Dict[str, Any]] = []

    if request_error is not None or response is None:
        checks.append(
            {
                "name": "request_executed",
                "ok": False,
                "message": request_error or "No response returned",
            }
        )
        result = {
            "ok": False,
            "summary": "Request failed to execute",
            "checks": checks,
            "request": {
                "method": method,
                "url": url,
                "headers": headers,
                "params": params,
                "body": payload,
            },
            "response": None,
            "bug_report": {
                "endpoint": f"{method} {endpoint}",
                "reason": request_error or "request_failed",
                "expected": expect,
                "actual": None,
                "request": {
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "params": params,
                    "body": payload,
                },
            },
        }
        return JSONResponse(result, status_code=200)

    # Response snapshot
    resp_text = response.text
    resp_json: Optional[Any] = None
    try:
        resp_json = response.json()
    except Exception:
        resp_json = None

    response_snapshot = {
        "status": response.status_code,
        "headers": dict(response.headers),
        "text": resp_text[:5000],
        "json": resp_json,
        "elapsed_ms": (
            int(response.elapsed.total_seconds() * 1000) if response.elapsed else None
        ),
    }

    # OpenAPI-assisted checks
    op = None
    path_item = None
    if isinstance(openapi_spec, dict):
        op, path_item = _find_openapi_operation(openapi_spec, endpoint, method)
        if path_item is None:
            checks.append(
                {
                    "name": "openapi_path_exists",
                    "ok": False,
                    "message": "Path not found in OpenAPI spec",
                }
            )
        else:
            checks.append(
                {
                    "name": "openapi_path_exists",
                    "ok": (
                        True if op is not None or isinstance(path_item, dict) else False
                    ),
                    "message": (
                        "Found matching path template"
                        if path_item is not None
                        else "Path missing"
                    ),
                }
            )
            if op is None:
                checks.append(
                    {
                        "name": "openapi_method_allowed",
                        "ok": False,
                        "message": f"Method {method} not defined for path in spec",
                    }
                )
            else:
                # Status code defined?
                resp_defs = (op.get("responses") or {}) if isinstance(op, dict) else {}
                defined_statuses: List[int] = []
                for k in resp_defs.keys():
                    try:
                        defined_statuses.append(int(k))
                    except Exception:
                        # ignore 'default' and non-int keys
                        pass
                if defined_statuses:
                    ok_status = response.status_code in defined_statuses
                    checks.append(
                        {
                            "name": "status_in_openapi_responses",
                            "ok": ok_status,
                            "message": f"Status {response.status_code} is{'' if ok_status else ' not'} defined in spec",
                            "expected_statuses": defined_statuses,
                        }
                    )

    # Expectation checks
    exp_status_list = (
        _to_int_list(expect.get("status")) if isinstance(expect, dict) else None
    )
    if exp_status_list:
        ok_status = response.status_code in exp_status_list
        checks.append(
            {
                "name": "status_expected",
                "ok": ok_status,
                "message": f"Expected status in {exp_status_list}, got {response.status_code}",
            }
        )
    else:
        # Default success criteria if nothing else given: 2xx
        ok_status = 200 <= response.status_code < 300
        checks.append(
            {
                "name": "status_2xx_default",
                "ok": ok_status,
                "message": f"Default check 2xx; got {response.status_code}",
            }
        )

    # If expectations require JSON, ensure parse success
    exp_fields = expect.get("json_fields") if isinstance(expect, dict) else None
    exp_contains = expect.get("json_contains") if isinstance(expect, dict) else None
    requires_json = (exp_fields is not None) or (exp_contains is not None)
    if requires_json:
        checks.append(
            {
                "name": "json_parse",
                "ok": resp_json is not None,
                "message": (
                    "Response parsed as JSON"
                    if resp_json is not None
                    else "Response is not valid JSON"
                ),
            }
        )

    if (
        isinstance(exp_fields, list)
        and resp_json is not None
        and isinstance(resp_json, dict)
    ):
        missing = [k for k in exp_fields if k not in resp_json]
        ok_fields = len(missing) == 0
        checks.append(
            {
                "name": "json_fields_present",
                "ok": ok_fields,
                "message": (
                    "All expected fields present"
                    if ok_fields
                    else f"Missing fields: {missing}"
                ),
            }
        )

    if isinstance(exp_contains, (dict, list)) and resp_json is not None:
        ok_contains = _partial_contains(exp_contains, resp_json)
        checks.append(
            {
                "name": "json_contains_partial",
                "ok": ok_contains,
                "message": (
                    "Response contains expected JSON structure"
                    if ok_contains
                    else "Response missing expected JSON structure"
                ),
            }
        )

    # Summarize
    overall_ok = (
        all(ch.get("ok") for ch in checks if isinstance(ch.get("ok"), bool))
        if checks
        else True
    )

    bug_report = None
    if not overall_ok:
        failed_msgs = [ch.get("message") for ch in checks if not ch.get("ok")]
        bug_report = {
            "endpoint": f"{method} {endpoint}",
            "reason": "; ".join([m for m in failed_msgs if isinstance(m, str)])
            or "validation_failed",
            "expected": expect,
            "actual": {
                "status": response.status_code,
                "json": resp_json if resp_json is not None else None,
                "text": resp_text[:2000],
            },
            "request": {
                "method": method,
                "url": url,
                "headers": headers,
                "params": params,
                "body": payload,
            },
            "suggested_next_step": "Have Coder Agent fix the failing endpoint or update API docs.",
        }

    result = {
        "ok": bool(overall_ok),
        "summary": "All checks passed" if overall_ok else "Some checks failed",
        "checks": checks,
        "request": {
            "method": method,
            "url": url,
            "headers": headers,
            "params": params,
            "body": payload,
        },
        "response": response_snapshot,
        "bug_report": bug_report,
    }

    return JSONResponse(result, status_code=200)


routes = [
    Route("/execute_task", execute_task, methods=["POST"]),
]


def _make_registration_payload(agent_address: str) -> Dict[str, Any]:
    return {
        "agent_name": AGENT_NAME,
        "agent_address": agent_address,
        "capabilities": {
            "role": "qa",
            "endpoints": ["/execute_task"],
            "version": "0.1.0",
            "mcp_tools": [
                {
                    "name": "qa_http_check",
                    "description": "Run an HTTP check against a running app. Optionally validate using OpenAPI.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "base_url": {"type": "string"},
                            "endpoint": {"type": "string"},
                            "method": {"type": "string", "default": "GET"},
                            "headers": {"type": ["object", "null"]},
                            "params": {"type": ["object", "null"]},
                            "body": {
                                "type": [
                                    "object",
                                    "array",
                                    "string",
                                    "number",
                                    "boolean",
                                    "null",
                                ]
                            },
                            "expect": {
                                "type": ["object", "null"],
                                "properties": {
                                    "status": {
                                        "anyOf": [
                                            {"type": "integer"},
                                            {
                                                "type": "array",
                                                "items": {"type": "integer"},
                                            },
                                        ]
                                    },
                                    "json_fields": {
                                        "type": ["array", "null"],
                                        "items": {"type": "string"},
                                    },
                                    "json_contains": {
                                        "type": ["object", "array", "null"]
                                    },
                                },
                                "additionalProperties": True,
                            },
                            "openapi_url": {"type": ["string", "null"]},
                            "openapi_spec": {"type": ["object", "null"]},
                            "timeout_secs": {
                                "type": ["number", "integer"],
                                "default": 20,
                            },
                        },
                        "required": ["base_url", "endpoint"],
                        "additionalProperties": False,
                    },
                    "_meta": {
                        "http": {
                            "endpoint": "/execute_task",
                            "method": "POST",
                            "returnType": "json",
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
