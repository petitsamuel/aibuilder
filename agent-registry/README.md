The Agent Registry MCP server


```
+------------------+      +------------------+      +------------------+
| CoderAgent.py    |      |   QA_Agent.py    |      | EnvManager.py    |
| (FastAPI @:8001) |      | (FastAPI @:8002) |      | (FastAPI @:8003) |
+------------------+      +------------------+      +------------------+
         ^                      ^                       ^
         | Registers with       | Registers with        | Registers with
         |                      |                       |
         +----------------------+-------+---------------+
                                        |
                                        v
                            +-----------------------+
                            |     MCP_Server.py     |
                            |   (FastAPI @:8000)    |
                            | [Holds Agent Registry]|
                            +-----------------------+
                                        ^
                                        | Queries & Dispatches
                                        |
                            +-----------------------+
                            |   Orchestrator.py     |
                            |  (Main Logic Loop)    |
                            +-----------------------+
```