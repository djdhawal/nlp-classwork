import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PATH = Path(__file__).resolve().parents[2] / "servers" / "math_server.py"


def render_text_result(result) -> str:
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH), "--transport", "stdio"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            result = await session.call_tool("power", {"base": 2, "exponent": 8})
            print(f"\n2^8 = {render_text_result(result)}")

            result = await session.call_tool("subtract", {"a": 100, "b": 37})
            print(f"100 - 37 = {render_text_result(result)}")


if __name__ == "__main__":
    asyncio.run(main())