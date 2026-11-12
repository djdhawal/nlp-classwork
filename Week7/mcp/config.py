"""
Shared config loader for the MCP project.

Reads config.yaml (located in the same directory as this file) and exposes
typed constants that all servers and clients import.

Usage in any file under week7/mcp/:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # N = depth to mcp/
    import config
"""

from pathlib import Path

import yaml

_MCP_ROOT = Path(__file__).resolve().parent
# Course root: mcp/ -> week7/ -> code/
_COURSE_ROOT = _MCP_ROOT.parents[1]

with open(_MCP_ROOT / "config.yaml") as _f:
    _cfg = yaml.safe_load(_f)

# --- Server ---
SERVER_HTTP_URL: str = _cfg["server"]["http_url"]
SERVER_MOUNT_PATH: str = _cfg["server"]["mount_path"]
SERVER_DEFAULT_TRANSPORT: str = _cfg["server"]["default_transport"]

# --- Ollama / LLM ---
# Single source of truth is week7/config.yaml (same model as the notebooks).
with open(_MCP_ROOT.parent / "config.yaml") as _f:
    _course_cfg = yaml.safe_load(_f)
OLLAMA_HOST: str = _course_cfg["ollama"]["url"]
OLLAMA_MODEL: str = _course_cfg["ollama"]["models"][0]

# --- File paths (resolved to absolute) ---
SAMPLE_FILE: Path = _COURSE_ROOT / _cfg["paths"]["sample_file"]
ALLOWED_ROOT: Path = _COURSE_ROOT / _cfg["paths"]["allowed_root"]
