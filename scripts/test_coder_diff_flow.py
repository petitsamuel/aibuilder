from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Dict

import httpx
import shutil
import subprocess
import time
import atexit


def _find_homepage_file(project_root: Path) -> Path:
    candidates: List[str] = [
        "src/app/page.tsx",
        "app/page.tsx",
        "pages/index.tsx",
        "src/pages/index.tsx",
        "src/app/page.jsx",
        "app/page.jsx",
        "pages/index.jsx",
        "src/pages/index.jsx",
    ]
    for rel in candidates:
        p = project_root / rel
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError("Could not locate a homepage file (page.tsx or index.tsx)")


def _is_healthy(url: str) -> bool:
    try:
        with httpx.Client(timeout=2.5) as client:
            r = client.get(url)
            return r.status_code == 200
    except Exception:
        return False


def _start_process(cmd: List[str], cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _ensure_agents(args, workspace_root: Path) -> Dict[str, subprocess.Popen]:
    started: Dict[str, subprocess.Popen] = {}

    launcher = args.launcher

    def run_cmd(
        module: Optional[str] = None, script: Optional[Path] = None
    ) -> List[str]:
        if module is None and script is None:
            raise ValueError("Either module or script must be provided")
        if launcher == "uv":
            if module is not None:
                print("Running:", "uv run -q -m", module, "(cwd=", workspace_root, ")")
                return ["uv", "run", "-q", "-m", module]
            else:
                print(
                    "Running:", "uv run -q", str(script), "(cwd=", workspace_root, ")"
                )
                return ["uv", "run", "-q", str(script)]
        else:
            if module is not None:
                print("Running:", "python -m", module, "(cwd=", workspace_root, ")")
                return ["python", "-m", module]
            else:
                print("Running:", "python", str(script), "(cwd=", workspace_root, ")")
                return ["python", str(script)]

    # Optionally start registry (best effort; no /health endpoint guaranteed)
    if args.start_agents and args.start_registry:
        registry_script = workspace_root / "agent-registry" / "server.py"
        if registry_script.exists():
            try:
                # Cannot use -m due to hyphen in directory name; run by script path
                proc = _start_process(
                    run_cmd(module=None, script=registry_script), cwd=workspace_root
                )
                started["registry"] = proc
                # Give it a moment to bind
                time.sleep(0.5)
            except Exception:
                pass

    # Ensure devops agent
    devops_health = f"{args.devops_url}/health"
    if not _is_healthy(devops_health):
        if not args.start_agents:
            raise SystemExit(
                f"DevOps agent not reachable at {devops_health}. Start it or pass --start-agents."
            )
        if shutil.which(launcher) is None and launcher == "uv":
            raise SystemExit(
                "'uv' not found in PATH. Install uv or use --launcher python"
            )
        proc = _start_process(run_cmd(module="devopsagent.devops"), cwd=workspace_root)
        started["devops"] = proc
        # Wait for health
        for _ in range(60):
            if _is_healthy(devops_health):
                break
            time.sleep(0.5)
        if not _is_healthy(devops_health):
            raise SystemExit("DevOps agent failed to become healthy")

    # Ensure coder agent
    coder_health = f"{args.coder_url}/health"
    print("Coder health", coder_health)
    print(f"{args.coder_url}/health")
    if not _is_healthy(coder_health):
        if not args.start_agents:
            raise SystemExit(
                f"Coder agent not reachable at {coder_health}. Start it or pass --start-agents."
            )
        if shutil.which(launcher) is None and launcher == "uv":
            raise SystemExit(
                "'uv' not found in PATH. Install uv or use --launcher python"
            )
        proc = _start_process(run_cmd(module="coder.coder"), cwd=workspace_root)
        started["coder"] = proc
        # Wait for health
        for _ in range(60):
            if _is_healthy(coder_health):
                break
            time.sleep(0.5)
        if not _is_healthy(coder_health):
            raise SystemExit("Coder agent failed to become healthy")

    # Ensure cleanup on exit
    if started:

        def _cleanup():
            for name, p in started.items():
                with contextlib.suppress(Exception):
                    p.terminate()

        import contextlib  # local import to keep global namespace slim

        atexit.register(_cleanup)

    return started


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for coder diff flow")
    parser.add_argument(
        "--start-agents",
        action="store_true",
        default=True,
        help="Automatically start required agents if not running (default: true)",
    )
    parser.add_argument(
        "--no-start-agents",
        action="store_false",
        dest="start_agents",
        help="Do not auto-start agents",
    )
    parser.add_argument(
        "--start-registry",
        action="store_true",
        default=True,
        help="Also start agent registry (best-effort)",
    )
    parser.add_argument(
        "--launcher",
        choices=["uv", "python"],
        default="uv",
        help="How to launch agents when auto-starting",
    )
    parser.add_argument(
        "--devops-url", default="http://127.0.0.1:8002", help="DevOps agent base URL"
    )
    parser.add_argument(
        "--coder-url", default="http://127.0.0.1:8003", help="Coder agent base URL"
    )
    parser.add_argument(
        "--template-path",
        default="t3-app",
        help="Template path under templates/ (relative) or absolute path",
    )
    parser.add_argument(
        "--dest",
        default="demo-t3-app",
        help="Destination project directory (relative to repo root)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite destination if it exists"
    )
    parser.add_argument(
        "--header-text",
        default="Hello Samuel!",
        help="Header text to insert on home page",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[1]

    # Ensure agents are up
    _ensure_agents(args, workspace_root)
    dest_dir = (workspace_root / args.dest).resolve()

    print(f"[1/4] Scaffolding template '{args.template_path}' to {dest_dir} ...")
    payload = {
        "template_path": args.template_path,
        "destination_path": str(dest_dir.relative_to(workspace_root)),
        "overwrite": bool(args.overwrite or True),
    }
    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{args.devops_url}/scaffold_project", json=payload)
        if r.status_code != 200:
            raise SystemExit(f"scaffold_project failed: {r.status_code} {r.text}")
        rsp = r.json()
        if rsp.get("status") != "ok":
            raise SystemExit(f"scaffold_project error: {json.dumps(rsp)}")
    print("    ✓ Project scaffolded")

    home_file = _find_homepage_file(dest_dir)
    print(f"[2/4] Target file: {home_file}")
    file_content = home_file.read_text(encoding="utf-8")

    task = (
        'Add an <h1> header at the very top of the rendered output that reads "'
        + args.header_text
        + '". Do not refactor or reformat unrelated lines. If necessary, wrap the return in a fragment.'
    )

    body = {
        "task": task,
        # Provide a repo-relative path to match prompts; the agent applies changes in-memory
        "file_path": str(home_file.relative_to(workspace_root)),
        "file_content": file_content,
    }

    print("[3/4] Asking coder agent to apply the change ...")
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{args.coder_url}/execute_task", json=body)
        if r.status_code != 200:
            # Try to parse potential error payload
            try:
                err = r.json()
            except Exception:
                err = {"raw": r.text}
            raise SystemExit(
                f"coder/execute_task failed: {r.status_code} {json.dumps(err)}"
            )
        # updated_text = r.text
        updated_text = r.text
        print("Updated text", updated_text)

    if not isinstance(updated_text, str) or not updated_text.strip():
        raise SystemExit("coder returned empty content")

    print("[4/4] Writing updated file and verifying ...")
    home_file.write_text(updated_text, encoding="utf-8")
    final_text = home_file.read_text(encoding="utf-8")
    if args.header_text not in final_text:
        print(final_text)
        raise SystemExit("Header text not found in updated file")
    print("    ✓ Update verified. Test successful.")


if __name__ == "__main__":
    main()
