Environment Manager & System Operator (DevOps Agent)

Provides endpoints to scaffold projects from `templates/`, manage dependencies, run git commands, and operate Docker Compose services. Registers itself with the Agent Registry.

### Start (from repo root)

```bash
uv run python devopsagent/devops.py
```

Environment overrides:

- `AGENT_NAME` (default `devops-agent`)
- `AGENT_HOST` (default `127.0.0.1`)
- `AGENT_PORT` (default `8002`)
- `REGISTRY_URL` (default `http://127.0.0.1:8000`)
- `HEARTBEAT_INTERVAL` seconds (default `60`)
- `WORKSPACE_ROOT` (default repository root)
- `TEMPLATES_DIR` (default `./templates` under workspace)
- `TEMPLATE_MANIFEST` (default `templates/manifest.yaml`)

### Health check

```bash
curl -sS http://127.0.0.1:8002/health
```

### Scaffold a project

```bash
curl -sS -X POST http://127.0.0.1:8002/scaffold_project \
  -H 'Content-Type: application/json' \
  -d '{
    "template_id": "nextjs-tailwind-static",
    "destination_path": "./sandbox/next-static",
    "overwrite": true
  }' | jq .
```

### Install dependencies (Node example)

```bash
curl -sS -X POST http://127.0.0.1:8002/install_dependencies \
  -H 'Content-Type: application/json' \
  -d '{
    "cwd": "./sandbox/next-static",
    "manager": "npm"
  }' | jq .
```

### Run a git command

```bash
curl -sS -X POST http://127.0.0.1:8002/git_command \
  -H 'Content-Type: application/json' \
  -d '{
    "cwd": "./sandbox/next-static",
    "args": ["status"]
  }' | jq .
```

### System operator: start services

```bash
curl -sS -X POST http://127.0.0.1:8002/system_operator/start \
  -H 'Content-Type: application/json' \
  -d '{
    "cwd": "./sandbox/next-static",
    "build": true
  }' | jq .
```

### System operator: stop services

```bash
curl -sS -X POST http://127.0.0.1:8002/system_operator/stop \
  -H 'Content-Type: application/json' \
  -d '{
    "cwd": "./sandbox/next-static",
    "down": true,
    "remove_orphans": true
  }' | jq .
```

### System operator: status

```bash
curl -sS -X POST http://127.0.0.1:8002/system_operator/status \
  -H 'Content-Type: application/json' \
  -d '{
    "cwd": "./sandbox/next-static"
  }' | jq .
```

### System operator: logs (snapshot)

```bash
curl -sS -X POST http://127.0.0.1:8002/system_operator/logs \
  -H 'Content-Type: application/json' \
  -d '{
    "cwd": "./sandbox/next-static",
    "tail": 200
  }' | jq -r .stdout
```

### System operator: logs (stream)

```bash
curl -N 'http://127.0.0.1:8002/system_operator/logs/stream?cwd=./sandbox/next-static&tail=100'
```

### HTTP health probe

```bash
curl -sS -X POST http://127.0.0.1:8002/system_operator/http_health \
  -H 'Content-Type: application/json' \
  -d '{
    "host": "127.0.0.1",
    "port": 3000,
    "path": "/"
  }' | jq .
```


