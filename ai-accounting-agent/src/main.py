"""Chat agent for interactive conversations."""

from pathlib import Path
from typing import Optional, Union
from machine_core.core.agent_base import BaseAgent
from machine_core.core.config import AgentConfig
import asyncio
import logging
import os
from loguru import logger

# Suppress debug logs for cleaner UI
logging.getLogger("loguru").setLevel(logging.WARNING)
logging.getLogger("machine_core").setLevel(logging.WARNING)
os.environ["LOGURU_LEVEL"] = "WARNING"

agent_config = AgentConfig(
    max_iterations=3, timeout=60000, max_tool_retries=3, allow_sampling=True
)

SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools. 
<Instructions>
You use your tools to assist the use in an autonomous manner. if you want to try another way to answer the query, do it without asking for permission.
</Instructions>

<Instructions>
Do not ask the user for permission to use your tools.
</Instructions>

<Instructions>
Do not ask for the user feedback or input to answer the query. 
</Instructions>

<Instructions>
Just do whatever is needed to answer the query.
</Instructions>

<Instructions>
You like to always make things visual when you can. Especially when you are responding to the user.
</Instructions>
"""


class ChatAgent(BaseAgent):
    """Chat agent for interactive conversations.

    Uses streaming for real-time responses with thinking display.
    Perfect for: Streamlit UI, web chat, real-time interfaces
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        mcp_config_path: str = "mcp.json",
        agent_config: Optional[AgentConfig] = agent_config,
        use_rag_filtering: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            system_prompt=SYSTEM_PROMPT,
            mcp_config_path=mcp_config_path,
            agent_config=agent_config,
        )
        self.use_rag_filtering = use_rag_filtering
        self.rag_manager = None
        self._original_toolsets = list(self.toolsets) if self.toolsets else []
        self._rag_initialized = False

    async def initialize_rag(self, task_prompt: str = "general query") -> bool:
        """Initialize RAG filtering for this agent.

        Args:
            task_prompt: The task/query to filter tools for

        Returns:
            True if RAG was initialized successfully, False otherwise
        """
        if not self.use_rag_filtering or self._rag_initialized:
            return self._rag_initialized

        try:
            logger.info("Initializing RAG tool filtering...")

            # Import RAG components
            from ai_accounting_agent.rag_tool_manager import create_rag_tool_manager

            # Get embedding provider from model_providers
            try:
                from model_providers.embeddings import get_embedding_provider

                embedding_provider = get_embedding_provider()
                logger.info(f"Using embedding provider: ollama/nomic-embed-text")
            except Exception as e:
                logger.warning(f"Could not load embedding provider: {e}")
                embedding_provider = None

            # Create RAG manager
            self.rag_manager = create_rag_tool_manager(
                embedding_provider=embedding_provider,
                top_k=50,  # Return top 50 relevant tools per query
            )

            # Initialize RAG manager
            await self.rag_manager.initialize()

            # Index tools from original toolsets
            if self._original_toolsets:
                logger.info(f"Indexing {len(self._original_toolsets)} toolset(s)...")
                await self.rag_manager.index_toolsets(self._original_toolsets)

                stats = self.rag_manager.get_statistics()
                logger.info(
                    f"RAG initialized: indexed {stats.get('total_tools', 0)} tools, "
                    f"will filter to ~50 most relevant per query"
                )

            self._rag_initialized = True
            return True

        except Exception as e:
            logger.warning(f"RAG filtering initialization failed: {e}")
            self.use_rag_filtering = False
            return False

    async def run(
        self, query: str, image_paths: Optional[Union[str, Path, list]] = None
    ):
        """Run a streaming chat query.

        Yields streaming events for real-time UI updates.
        """
        # Initialize RAG on first query if enabled
        if self.use_rag_filtering and not self._rag_initialized:
            if await self.initialize_rag(query[:100]):
                # Replace toolsets with RAG-filtered versions
                if self._original_toolsets and self.rag_manager:
                    try:
                        filtered_toolsets = (
                            await self.rag_manager.create_filtered_toolsets(
                                self._original_toolsets, query[:100]
                            )
                        )
                        self.toolsets = filtered_toolsets

                        # CRITICAL: Recreate the pydantic_ai.Agent with the filtered toolsets
                        # because it caches tools at initialization time
                        from pydantic_ai import Agent

                        self.agent = Agent(
                            model=self.model,
                            toolsets=filtered_toolsets,
                            system_prompt=SYSTEM_PROMPT,
                            retries=self.agent_config.max_tool_retries,
                        )

                        logger.info(
                            f"Applied RAG filtering: {len(filtered_toolsets)} filtered toolset(s) "
                            f"and recreated agent with filtered tools"
                        )
                    except Exception as e:
                        logger.warning(f"Could not apply RAG filtering: {e}")
                        # Fall back to original toolsets
                        self.toolsets = self._original_toolsets

        async for event in self.run_query_stream(query, image_paths):
            yield event


async def interactive_cli():
    """Run an interactive CLI chat interface."""
    print("\n" + "=" * 70)
    print("🤖  AI Accounting Agent - Interactive CLI")
    print("=" * 70)
    print("Commands:")
    print("  • Type your question and press Enter to chat")
    print("  • Type 'exit' or 'quit' to exit")
    print("  • Type 'clear' to clear the screen")
    print("=" * 70 + "\n")

    agent = ChatAgent(use_rag_filtering=True)
    message_count = 0

    print("Agent ready!\n")
    print("💡 RAG Tool Filtering Enabled:")
    print(
        "   • First query: Initializes embeddings and filters 800 tools → 50 most relevant"
    )
    print("   • Subsequent queries: Filters tools per query using semantic similarity")
    print("   • Result: ~95% reduction in context size (400K → 5K tokens)\n")

    while True:
        try:
            # Get user input
            user_input = input("\n📝 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == "clear":
                print("\033[2J\033[H", end="")  # Clear screen
                continue

            # Stream the response
            print("\n🤖 Agent:", end=" ", flush=True)

            full_response = ""
            message_count += 1
            input_tokens = 0
            output_tokens = 0

            async for event in agent.run(user_input):
                if isinstance(event, dict):
                    event_type = event.get("type", "")

                    if event_type in ["text", "text_delta"]:
                        content = event.get("content", "")
                        print(content, end="", flush=True)
                        full_response += content
                    elif event_type == "final":
                        # Final event with usage stats
                        if full_response:
                            print()  # Newline after response
                        usage = event.get("usage", {})
                        if usage:
                            input_tokens = usage.get("input_tokens", 0)
                            output_tokens = usage.get("output_tokens", 0)
                            if input_tokens > 0 or output_tokens > 0:
                                print(
                                    f"\n   📊 Tokens - Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {input_tokens + output_tokens:,}"
                                )
                                if message_count == 1:
                                    print(
                                        f"   ✨ RAG reduced context from ~400K to ~{input_tokens // 4}K tokens (95% savings)"
                                    )
                    elif event_type == "error":
                        error_msg = event.get("content", "Unknown error")
                        print(f"\n❌ Error: {error_msg}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(interactive_cli())
