import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("TextUtilities")


@mcp.tool()
def count_words(text: str) -> int:
    """Count the number of words in the input text"""
    return len(text.split())


@mcp.tool()
def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim the result"""
    return re.sub(r"\s+", " ", text).strip()


@mcp.tool()
def change_case(text: str, mode: str = "title") -> str:
    """Convert text to upper, lower, or title case"""
    normalized_mode = mode.lower()
    if normalized_mode == "upper":
        return text.upper()
    if normalized_mode == "lower":
        return text.lower()
    if normalized_mode == "title":
        return text.title()
    raise ValueError("mode must be one of: upper, lower, title")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MCP text utility server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=config.SERVER_DEFAULT_TRANSPORT,
        help="Transport to expose for MCP clients",
    )
    parser.add_argument(
        "--mount-path",
        default=config.SERVER_MOUNT_PATH,
        help="Mount path for the streamable HTTP transport",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.transport == "streamable-http":
        mcp.run(transport=args.transport, mount_path=args.mount_path)
    else:
        mcp.run(transport=args.transport)