import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

async def example():
    transport = StreamableHttpTransport(url="http://127.0.0.1:8000/mcp/")
    async with Client(transport=transport) as client:
        await client.ping()
        print("Connected to server")
        
        tools = await client.list_tools()
        print("Available tools:", tools)
        
        greeting = await client.call_tool("greet", {"name": "John"})
        print("Greeting:", greeting)


if __name__ == "__main__":
    asyncio.run(example())