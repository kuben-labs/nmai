"""Test script for verifying MCP tool schemas are loaded correctly.

With the new FilteredToolset approach, schemas are preserved as-is from the
MCP server - no sanitization needed. This test just validates the tools load.
"""

import asyncio
from src.ai_accounting_agent.coordinator import load_mcp_config, get_mcp_toolset


async def test():
    toolset = get_mcp_toolset()
    all_tools = await toolset.list_tools()

    print(f"Total tools: {len(all_tools)}")

    for tool in all_tools[:5]:
        schema = tool.inputSchema
        print(f"\nTool: {tool.name}")
        print(f"  Description: {(tool.description or '')[:80]}...")
        if schema:
            props = schema.get("properties", {})
            print(f"  Properties: {list(props.keys())[:10]}")
            if "$defs" in schema:
                print(f"  Has $defs: {list(schema['$defs'].keys())[:5]}")


if __name__ == "__main__":
    asyncio.run(test())
