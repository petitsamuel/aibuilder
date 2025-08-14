Agent Registry (MCP server)

The Agent Registry exposes an MCP-compatible server and two HTTP endpoints for agents to register and for operators to inspect status. Other agents heartbeat to this registry and publish their MCP tools dynamically.

```
+------------------+      +------------------+      +------------------+
|  Architect       |      |   Coder          |      |  DevOps Agent    |
| (HTTP @:8001)    |      | (HTTP @:8003)    |      | (HTTP @:8002)    |
+------------------+      +------------------+      +------------------+
         ^                      ^                       ^
         | Registers with       | Registers with        | Registers with
         |                      |                       |
         +----------------------+-------+---------------+
                                        |
                                        v
                            +-----------------------+
                            |   Agent Registry      |
                            |   (MCP @:8000/mcp)    |
                            |  + /register, /status |
                            +-----------------------+
                                        ^
                                        |
                                        |
                            +-----------------------+
                            |    Orchestrator       |
                            |   (Client of MCP)     |
                            +-----------------------+
```

### Start

Defaults: host `127.0.0.1`, port `8000`.

```bash
uv run python agent-registry/agentregistry.py
```

Environment overrides:

- `AGENT_REGISTRY_HOST` (default `127.0.0.1`)
- `AGENT_REGISTRY_PORT` (default `8000`)

The MCP transport is served under `/mcp` (e.g., `http://127.0.0.1:8000/mcp/`).

### Health and status

```bash
curl -sS http://127.0.0.1:8000/status | jq .
```

### Register/heartbeat an agent (manual test)

```bash
curl -sS -X POST http://127.0.0.1:8000/register \
  -H 'Content-Type: application/json' \
  -d '{
    "agent_name": "sample-agent",
    "agent_address": "http://127.0.0.1:9999",
    "capabilities": {
      "role": "sample",
      "version": "0.0.0",
      "endpoints": ["/health"],
      "mcp_tools": [
        {
          "name": "ping",
          "description": "Proxy GET /health on the sample agent",
          "inputSchema": {"type": "object", "additionalProperties": true},
          "_meta": {"http": {"endpoint": "/health", "method": "GET", "returnType": "text"}}
        }
      ]
    }
  }' | jq .
```

### Optional: demo test server and client

A minimal standalone MCP test server with a `greet` tool is provided for local sanity checks. This is separate from the Agent Registry.

Start the test server:

```bash
uv run python agent-registry/server.py
```

Then run the sample client (targets `http://127.0.0.1:8000/mcp/`):

```bash
uv run python agent-registry/client.py
```
