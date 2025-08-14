Orchestrator

Coordinates agents via the Agent Registry's MCP interface. Maintains a lightweight plan and history in SQLite. Runs in a loop or single turn.

### Prerequisites

- Agent Registry running on `http://127.0.0.1:8000/mcp/`
- At least one agent registered (e.g., Architect, DevOps, Coder)

### Start (single turn)

```bash
uv run python -m orchestrator.orchestrator --goal "Create a Next.js landing page" --registry-url http://127.0.0.1:8000 --db-path ./orchestrator/orchestrator_state.sqlite --project-root .
```

### Start (continuous loop)

```bash
uv run python -m orchestrator.orchestrator --goal "Create a Next.js landing page" --registry-url http://127.0.0.1:8000 --db-path ./orchestrator/orchestrator_state.sqlite --project-root . --continuous --interval 10
```

CLI options:

- `--goal` (required)
- `--registry-url` (default `http://127.0.0.1:8000`)
- `--db-path` (default `./orchestrator_state.sqlite` in CWD)
- `--project-root` (default current directory)
- `--continuous` (run loop)
- `--interval` seconds between turns (default 10)


