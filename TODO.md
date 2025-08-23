
~/dev/aibuilder main ⇡1 !4 ?10 ❯ python scripts/e2e_full_todo_flow.py                                                                                                Py aibuilder 01:07:18
Starting registry...
Starting agents: architect, devops, coder, qa ...
Starting orchestrator (continuous for ~20s)...
History entries:
- 2025-08-22T23:07:24.227356+00:00 | tech-stack-selector | CallToolResult(content=[TextContent(type='text', text='t3-stack', annotations=None, meta=None)], structured_content=None, data=None, is_error=False) | tool=tech-stack-selector.select_tech_stack input={"prompt": "Build me a simple to-do list app where I can add and see tasks. It needs a database."}
- 2025-08-22T23:07:28.819165+00:00 | devops-agent | CallToolResult(content=[TextContent(type='text', text='{\n  "status": "ok",\n  "source": "/Users/sam/dev/aibuilder/templates/t3-app",\n  "destination": "/Users/sam/dev/aibuilder/todo-app",\n  "overwri | tool=devops-agent.scaffold_project input={"template_id": "t3-stack", "destination": "todo-list-app", "project_name": "todo-app"}
- 2025-08-22T23:07:33.487633+00:00 | devops-agent | CallToolResult(content=[TextContent(type='text', text='{\n  "status": "ok",\n  "source": "/Users/sam/dev/aibuilder/templates/t3-app",\n  "destination": "/Users/sam/dev/aibuilder/todo-app",\n  "overwri | tool=devops-agent.scaffold_project input={"source": "/Users/sam/dev/aibuilder/templates/t3-app", "destination": "/Users/sam/dev/aibuilder/todo-app", "project_name": "todo-app", "overwrite": true, "template_id": "t3-stack"}
- 2025-08-22T23:07:38.005925+00:00 | devops-agent | CallToolResult(content=[TextContent(type='text', text='{\n  "status": "ok",\n  "source": "/Users/sam/dev/aibuilder/templates/t3-app",\n  "destination": "/Users/sam/dev/aibuilder/todo-app",\n  "overwri | tool=devops-agent.scaffold_project input={"source": "/Users/sam/dev/aibuilder/templates/t3-app", "destination": "/Users/sam/dev/aibuilder/todo-app", "project_name": "todo-app", "overwrite": true, "template_id": "t3-stack"}
- 2025-08-22T23:07:42.662206+00:00 | devops-agent | CallToolResult(content=[TextContent(type='text', text='{\n  "status": "ok",\n  "source": "/Users/sam/dev/aibuilder/templates/t3-app",\n  "destination": "/Users/sam/dev/aibuilder/todo-app",\n  "overwri | tool=devops-agent.scaffold_project input={"source": "/Users/sam/dev/aibuilder/templates/t3-app", "destination": "/Users/sam/dev/aibuilder/todo-app", "project_name": "todo-app", "overwrite": true, "template_id": "t3-stack"}

Full E2E Orchestrator flow completed.


the e2e test is currently blocking on calling the same tool 

also make the project creation use a unique id or something
