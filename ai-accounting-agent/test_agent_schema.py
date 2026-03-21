import asyncio
from src.ai_accounting_agent.coordinator import load_mcp_config, setup_mcp_toolsets
from src.ai_accounting_agent.rag_tool_filter import sanitize_schema

async def test():
    configs = load_mcp_config()
    toolsets = setup_mcp_toolsets(configs)
    all_tools = await toolsets[0].list_tools()
    
    for tool in all_tools:
        schema = tool.inputSchema
        if not schema:
            continue
        sanitized = sanitize_schema(schema)
        # Check if properties has string values
        props = sanitized.get("properties", {})
        for k, v in props.items():
            if not isinstance(v, dict):
                print(f"Tool {tool.name} has non-dict property {k}: {v}")

if __name__ == "__main__":
    asyncio.run(test())
