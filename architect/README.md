Architect Agent (Tech Stack Selector)

Chooses the best template/stack for a project prompt. Exposes an HTTP endpoint and registers itself with the Agent Registry.

### Start (from repo root)

```bash
uv run python architect/architect.py
```

Environment overrides:

- `AGENT_NAME` (default `tech-stack-selector`)
- `AGENT_HOST` (default `127.0.0.1`)
- `AGENT_PORT` (default `8001`)
- `REGISTRY_URL` (default `http://127.0.0.1:8000`)
- `HEARTBEAT_INTERVAL` seconds (default `60`)

### Health check

```bash
curl -sS http://127.0.0.1:8001/health
```

### Select a tech stack

Request body:

- `prompt` string (required)
- `templates` array of strings (optional; defaults to built-in list)

```bash
curl -sS -X POST http://127.0.0.1:8001/execute_task \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "I want a simple blog with Next.js and Tailwind",
    "templates": ["react-express-sqlite", "nextjs-tailwind-static"]
  }'
```

Expected response: a plain-text template id, e.g. `nextjs-tailwind-static`.


