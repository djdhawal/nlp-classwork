# math_server.py
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract b from a"""
    return a - b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the given exponent"""
    return base ** exponent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MCP math server")
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