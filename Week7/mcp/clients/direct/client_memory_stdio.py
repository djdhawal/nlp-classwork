"""
Third-party MCP example: Memory server (knowledge graph)
---------------------------------------------------------
Uses the official @modelcontextprotocol/server-memory package to persist
entities and relations in a local knowledge graph.  This shows how an agent
can build and query structured memory through MCP tools.

Requirements:
    npx (comes with Node.js)

No API key needed. Graph state is kept in-process (resets each run).
"""

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PARAMS = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-memory"],
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

            # --- create NLP course entities ---
            print("Creating entities...")
            await session.call_tool(
                "create_entities",
                {
                    "entities": [
                        {
                            "name": "Tokenization",
                            "entityType": "NLPConcept",
                            "observations": ["Splits text into tokens", "Week 1 topic"],
                        },
                        {
                            "name": "BERT",
                            "entityType": "Model",
                            "observations": [
                                "Bidirectional Encoder Representations from Transformers",
                                "Pre-trained on masked language modeling",
                                "Week 3 topic",
                            ],
                        },
                        {
                            "name": "RAG",
                            "entityType": "Technique",
                            "observations": [
                                "Retrieval-Augmented Generation",
                                "Combines retrieval with generation",
                                "Week 7 topic",
                            ],
                        },
                    ]
                },
            )
            print("  Entities created: Tokenization, BERT, RAG")
            print()

            # --- create relations ---
            print("Creating relations...")
            await session.call_tool(
                "create_relations",
                {
                    "relations": [
                        {
                            "from": "BERT",
                            "to": "Tokenization",
                            "relationType": "requires",
                        },
                        {
                            "from": "RAG",
                            "to": "BERT",
                            "relationType": "can_use",
                        },
                    ]
                },
            )
            print("  Relations created: BERT→requires→Tokenization, RAG→can_use→BERT")
            print()

            # --- search for a node ---
            print("Searching for 'BERT':")
            search = await session.call_tool("search_nodes", {"query": "BERT"})
            raw = render_text_result(search)
            try:
                parsed = json.loads(raw)
                for entity in parsed.get("entities", []):
                    print(f"  {entity['name']} ({entity['entityType']})")
                    for obs in entity.get("observations", []):
                        print(f"    - {obs}")
            except (json.JSONDecodeError, KeyError):
                print(raw)
            print()

            # --- read the full graph ---
            print("Full graph summary:")
            graph = await session.call_tool("read_graph", {})
            raw = render_text_result(graph)
            try:
                parsed = json.loads(raw)
                print(f"  Entities : {len(parsed.get('entities', []))}")
                print(f"  Relations: {len(parsed.get('relations', []))}")
            except (json.JSONDecodeError, KeyError):
                print(raw)


if __name__ == "__main__":
    asyncio.run(main())
