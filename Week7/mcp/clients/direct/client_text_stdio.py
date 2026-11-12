import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PATH = Path(__file__).resolve().parents[2] / "servers" / "text_server.py"


def render_text_result(result) -> str:
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH), "--transport", "stdio"],
    )

    sample_text = "  model   context   protocol makes   tools reusable  "

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            normalized = await session.call_tool(
                "normalize_whitespace", {"text": sample_text}
            )
            print(f"\nNormalized: {render_text_result(normalized)}")

            word_count = await session.call_tool("count_words", {"text": sample_text})
            print(f"Word count: {render_text_result(word_count)}")

            title_case = await session.call_tool(
                "change_case", {"text": "mcp helps standardize tool use", "mode": "title"}
            )
            print(f"Title case: {render_text_result(title_case)}")


if __name__ == "__main__":
    asyncio.run(main())