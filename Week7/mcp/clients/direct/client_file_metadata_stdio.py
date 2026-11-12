import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PATH = Path(__file__).resolve().parents[2] / "servers" / "file_metadata_server.py"


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

            summary = await session.call_tool("file_summary", {"path": str(config.SAMPLE_FILE)})
            print(f"\nFile summary:\n{render_text_result(summary)}")

            preview = await session.call_tool(
                "read_text_preview", {"path": str(config.SAMPLE_FILE), "max_chars": 120}
            )
            print(f"\nPreview:\n{render_text_result(preview)}")

            listing = await session.call_tool(
                "list_directory", {"path": str(config.SAMPLE_FILE.parent)}
            )
            print(f"\nSibling files:\n{render_text_result(listing)}")


if __name__ == "__main__":
    asyncio.run(main())