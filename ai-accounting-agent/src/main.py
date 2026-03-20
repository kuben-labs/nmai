"""Chat agent for interactive conversations."""

from pathlib import Path
from typing import Optional, Union
from machine_core.core.agent_base import BaseAgent
from machine_core.core.config import AgentConfig
import asyncio
import logging
import os

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
    ):
        super().__init__(
            model_name=model_name,
            system_prompt=SYSTEM_PROMPT,
            mcp_config_path=mcp_config_path,
            agent_config=agent_config,
        )

    async def run(
        self, query: str, image_paths: Optional[Union[str, Path, list]] = None
    ):
        """Run a streaming chat query.

        Yields streaming events for real-time UI updates.
        """
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

    agent = ChatAgent()
    message_count = 0

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

            async for event in agent.run(user_input):
                event_type = (
                    event.get("type")
                    if isinstance(event, dict)
                    else event.get("type", "")
                )

                if event_type in ["text", "text_delta"]:
                    content = (
                        event.get("content", "") if isinstance(event, dict) else ""
                    )
                    print(content, end="", flush=True)
                    full_response += content
                elif event_type == "final":
                    # Final event with usage stats
                    if full_response:
                        print()  # Newline after response
                    usage = event.get("usage", {}) if isinstance(event, dict) else {}
                    if usage and (
                        usage.get("input_tokens", 0) > 0
                        or usage.get("output_tokens", 0) > 0
                    ):
                        print(
                            f"   📊 Tokens - Input: {usage.get('input_tokens', 0)}, Output: {usage.get('output_tokens', 0)}"
                        )
                elif event_type == "error":
                    error_msg = (
                        event.get("content", "Unknown error")
                        if isinstance(event, dict)
                        else "Unknown error"
                    )
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
