Coder Agent

Applies precise code edits to a single provided file using an LLM via OpenRouter. Exposes a single HTTP endpoint and registers with the Agent Registry.

### Start (from repo root)

```bash
OPENROUTER_API_KEY=... uv run python coder/coder.py
```

Environment overrides:

- `AGENT_NAME` (default `coder`)
- `AGENT_HOST` (default `127.0.0.1`)
- `AGENT_PORT` (default `8003`)
- `REGISTRY_URL` (default `http://127.0.0.1:8000`)
- `HEARTBEAT_INTERVAL` seconds (default `60`)
- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL`, `OPENROUTER_TEMPERATURE`, `OPENROUTER_MAX_TOKENS`, `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME`

### Health check

```bash
curl -sS http://127.0.0.1:8003/health
```

### Apply a code task

Request body:

- `task` string (required): high-level instruction
- `file_path` string (required): path for context
- `file_content` string (required): current content of the file

```bash
curl -sS -X POST http://127.0.0.1:8003/execute_task \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "Add a new link in the header to About page",
    "file_path": "templates/t3-app/src/app/page.tsx",
    "file_content": "'""$(cat templates/t3-app/src/app/page.tsx | sed 's/"/\\"/g')""'"
  }'
```

Response: plain text with either a unified diff or the updated file content.


