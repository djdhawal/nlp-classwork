# MCP Examples

This folder contains small Model Context Protocol (MCP) servers plus direct and agent-based client examples.

## Files

- `servers/`: MCP servers grouped by capability.
- `clients/direct/`: Minimal MCP clients that talk to servers directly.
- `clients/agents/`: Agent framework integrations built on top of MCP tools.

Current files:

- `config.yaml`: Central configuration (server URL, mount path, file paths; the LLM host/model come from `../config.yaml`). Edit this file to change any runtime parameters.
- `config.py`: Config loader — reads `config.yaml` and exposes typed constants imported by all servers and clients.
- `servers/math_server.py`: FastMCP server exposing calculator tools.
- `servers/text_server.py`: FastMCP server exposing text utility tools.
- `servers/file_metadata_server.py`: FastMCP server exposing basic file inspection tools.
- `clients/direct/client_stdio_basic.py`: Minimal direct MCP client over `stdio`.
- `clients/direct/client_streamable_http_basic.py`: Minimal direct MCP client over `streamable-http`.
- `clients/direct/client_text_stdio.py`: Minimal direct MCP client for the text utility server.
- `clients/direct/client_file_metadata_stdio.py`: Minimal direct MCP client for the file metadata server.
- `clients/direct/client_filesystem_stdio.py`: Third-party Filesystem MCP server — lists and reads files from an allowed directory.
- `clients/direct/client_memory_stdio.py`: Third-party Memory MCP server — stores and queries entities in a local knowledge graph.
- `clients/agents/client_langchain.py`: LangChain agent that loads MCP tools over `stdio`.
- `clients/agents/client_strands.py`: Strands agent that loads MCP tools over `streamable-http`.

## Example Servers

### Math Server

The math server exposes these tools:

- `add(a, b)`
- `subtract(a, b)`
- `multiply(a, b)`
- `divide(a, b)`
- `power(base, exponent)`

### Text Utilities Server

The text server exposes these tools:

- `count_words(text)`
- `normalize_whitespace(text)`
- `change_case(text, mode)`

### File Metadata Server

The file metadata server exposes these tools:

- `file_summary(path)`
- `list_directory(path)`
- `read_text_preview(path, max_chars)`

## How To Think About The Examples

- Use `stdio` when the client should start and manage the MCP server as a subprocess.
- Use `streamable-http` when the server should stay running and clients should connect over HTTP. The URL is configured in `config.yaml` (`server.http_url`).
- Start with the direct clients first. They show the MCP flow without adding an LLM framework on top.

Recommended order:

1. `clients/direct/client_stdio_basic.py`
2. `clients/direct/client_text_stdio.py`
3. `clients/direct/client_file_metadata_stdio.py`
4. `clients/direct/client_filesystem_stdio.py`
5. `clients/direct/client_memory_stdio.py`
6. `clients/direct/client_streamable_http_basic.py`
7. `clients/agents/client_langchain.py`
8. `clients/agents/client_strands.py`

## Quick Start

Each server supports the same runtime options:

- `--transport stdio`
- `--transport streamable-http`
- `--mount-path /mcp`

Run a server over `stdio` when a client will spawn it as a subprocess:

```bash
cd week7/mcp
python servers/math_server.py --transport stdio
```

Run the server over `streamable-http` when clients will connect over HTTP:

```bash
cd week7/mcp
python servers/math_server.py --transport streamable-http
```

The direct `stdio` clients already launch their matching servers for you, so you usually do not need to start those servers manually.

## Example Runs

Math client over `stdio`:

```bash
cd week7/mcp
python clients/direct/client_stdio_basic.py
```

Text utilities client:

```bash
cd week7/mcp
python clients/direct/client_text_stdio.py
```

File metadata client:

```bash
cd week7/mcp
python clients/direct/client_file_metadata_stdio.py
```

Math client over `streamable-http`:

```bash
cd week7/mcp
python servers/math_server.py --transport streamable-http
python clients/direct/client_streamable_http_basic.py
```

LangChain example:

```bash
cd week7/mcp
python clients/agents/client_langchain.py
```

Strands example:

```bash
cd week7/mcp
python servers/math_server.py --transport streamable-http
python clients/agents/client_strands.py
```

## Typical Workflows

### Direct `stdio`

Use a direct `stdio` client when you want the simplest MCP flow:

1. The client starts the matching server as a subprocess.
2. The session is initialized.
3. The client lists tools.
4. The client calls tools and prints the results.

### Direct `streamable-http`

Use the HTTP flow when you want a long-running server process that multiple clients can connect to.

### Text Utilities

Use the text server when you want small transformation or analysis tools such as whitespace cleanup, word counting, or case conversion.

### File Metadata

Use the file metadata server when you want tools that inspect local files, preview text, or list directory contents.

## Extending The Examples

To add a new MCP tool, follow the same pattern used by the current servers:

```python
@mcp.tool()
def modulo(a: int, b: int) -> int:
    return a % b
```

Guidelines:

- Keep signatures simple and typed.
- Add a clear docstring because clients surface tool descriptions.
- Validate bad inputs inside the tool, such as divide-by-zero or missing-file cases.

## Troubleshooting

### HTTP client cannot connect

Make sure the server is already running with `--transport streamable-http` and that `server.http_url` in `config.yaml` matches the address the server is listening on.

### Tool list is empty or calls fail

Make sure the client initializes the MCP session before calling `list_tools()` or `call_tool()`.

### File preview fails

Make sure the path exists and the file is readable as UTF-8 text.

## Third-Party MCP Servers

These examples connect to official reference servers from the
[modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) repo.
They require Node.js (`npx`) but no API keys.

### Filesystem

Exposes a local directory as MCP tools for listing and reading files.
The demo points at the folder configured under `paths.allowed_root` in `config.yaml`.

```bash
cd week7/mcp
python clients/direct/client_filesystem_stdio.py
```

The client launches `npx @modelcontextprotocol/server-filesystem` automatically.
Tools exposed: `list_directory`, `read_file`, `get_file_info`, `search_files`, and more.

### Memory (knowledge graph)

Exposes a local knowledge graph as MCP tools for creating and querying entities
and relations. The demo stores NLP concepts and then searches and reads the graph.

```bash
cd week7/mcp
python clients/direct/client_memory_stdio.py
```

The client launches `npx @modelcontextprotocol/server-memory` automatically.
Tools exposed: `create_entities`, `create_relations`, `search_nodes`, `read_graph`, and more.

## Dependencies

At minimum, the direct examples need the `mcp` package installed in the Python environment you use to run them.

The file metadata example also assumes you point it at files that exist on your machine and that text previews are UTF-8 readable.

The third-party examples require Node.js ≥ 18 with `npx` available (`node --version`, `npx --version`).

The framework-specific examples also need their respective packages:

- `client_langchain.py`: LangChain, LangChain MCP adapters, and LangChain Ollama support.
- `client_strands.py`: Strands plus its MCP and Ollama integrations.
