# Create server parameters for stdio connection
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PATH = Path(__file__).resolve().parents[2] / "servers" / "math_server.py"

model = ChatOllama(model=config.OLLAMA_MODEL, api_url=config.OLLAMA_HOST)

server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(SERVER_PATH), "--transport", "stdio"],
)


async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)

            agent = create_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            return agent_response


if __name__ == "__main__":
    result = asyncio.run(run_agent())

    print("\n" + "=" * 60)
    print("AGENT RESPONSE SUMMARY")
    print("=" * 60)

    messages = result.get("messages", [])

    if messages:
        print(f"\nUSER QUERY:")
        print(f"  {messages[0].content}")

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"\nTOOL CALLS:")
                for tool_call in msg.tool_calls:
                    print(f"  Tool: {tool_call['name']}")
                    print(f"  Args: {tool_call['args']}")
                break

        final_response = None
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not msg.content.startswith("ToolMessage"):
                final_response = msg
                break

        if final_response:
            print(f"\nFINAL RESPONSE:")
            print(f"  {final_response.content}")

    print("\n" + "=" * 60)