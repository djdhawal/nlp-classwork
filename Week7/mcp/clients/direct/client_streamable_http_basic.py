import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def render_text_result(result) -> str:
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


async def main() -> None:
    async with streamablehttp_client(config.SERVER_HTTP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            result = await session.call_tool("add", {"a": 125, "b": 375})
            print(f"\n125 + 375 = {render_text_result(result)}")

            result = await session.call_tool("divide", {"a": 144, "b": 12})
            print(f"144 / 12 = {render_text_result(result)}")


if __name__ == "__main__":
    asyncio.run(main())