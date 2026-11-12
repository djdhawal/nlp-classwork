import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("FileMetadata")


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Path not found: {candidate}")
    return candidate


@mcp.tool()
def file_summary(path: str) -> str:
    """Return basic metadata about a file or directory"""
    target = resolve_path(path)
    stats = target.stat()
    kind = "directory" if target.is_dir() else "file"
    return (
        f"path: {target}\n"
        f"type: {kind}\n"
        f"size_bytes: {stats.st_size}\n"
        f"modified: {stats.st_mtime}"
    )


@mcp.tool()
def list_directory(path: str = ".") -> str:
    """List entries in a directory"""
    target = resolve_path(path)
    if not target.is_dir():
        raise ValueError(f"Path is not a directory: {target}")
    entries = sorted(child.name for child in target.iterdir())
    return "\n".join(entries) if entries else "<empty directory>"


@mcp.tool()
def read_text_preview(path: str, max_chars: int = 200) -> str:
    """Read a short preview from a UTF-8 text file"""
    target = resolve_path(path)
    if target.is_dir():
        raise ValueError(f"Path is a directory: {target}")
    return target.read_text(encoding="utf-8")[:max_chars]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MCP file metadata server")
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