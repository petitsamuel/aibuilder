from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="test-server")

@mcp.tool()
async def greet(name: str) -> str:
    """Greet a person."""
    return "Hello from agent-registry!"

if __name__ == "__main__":
    mcp.settings.streamable_http_path = "/mcp"
    mcp.settings.mount_path = "/mcp"
    mcp.settings.host = "127.0.0.1"
    mcp.settings.port = 8000
    mcp.run(transport='streamable-http')