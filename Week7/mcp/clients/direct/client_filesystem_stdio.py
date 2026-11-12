"""
Third-party MCP example: Filesystem server
-------------------------------------------
Uses the official @modelcontextprotocol/server-filesystem package to expose
a local directory as MCP tools a client can use to list and read files.

Requirements:
    npx (comes with Node.js)

No API key needed. The server is downloaded and run on-demand via npx.

The allowed root is set to the week1/ folder of this course so the demo
works out of the box. Change ALLOWED_ROOT to any directory you want to
expose.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Expose the week1 directory (contains real NLP course text files).
ALLOWED_ROOT = str(config.ALLOWED_ROOT)

SERVER_PARAMS = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", ALLOWED_ROOT],
)


def render_text_result(result) -> str:
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


async def main() -> None:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- list available tools ---
            tools_result = await session.list_tools()
            print("Available tools:")
            for t in tools_result.tools:
                print(f"  - {t.name}: {t.description}")
            print()

            # --- list the week1 directory ---
            print(f"Listing: {ALLOWED_ROOT}")
            result = await session.call_tool("list_directory", {"path": ALLOWED_ROOT})
            print(render_text_result(result))
            print()

            # --- read a short preview of article.txt ---
            article = str(Path(ALLOWED_ROOT) / "article.txt")
            print(f"File info for article.txt:")
            info = await session.call_tool("get_file_info", {"path": article})
            print(render_text_result(info))
            print()

            print("First 300 characters of article.txt:")
            content = await session.call_tool("read_file", {"path": article})
            text = render_text_result(content)
            print(text[:300])
            print()

            # --- search for python source files ---
            print("Python files in week1/:")
            search = await session.call_tool(
                "search_files",
                {"path": ALLOWED_ROOT, "pattern": "*.py"},
            )
            print(render_text_result(search))


if __name__ == "__main__":
    asyncio.run(main())
