"""Main CLI entry point for the accounting agent."""

import asyncio
import sys
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


async def main():
    """Main CLI entry point for testing the agent locally."""
    from .coordinator import run_accounting_task

    logger.info("AI Accounting Agent - CLI Mode")
    logger.info("=" * 60)

    # Get task from user
    print("\nEnter accounting task (or 'quit' to exit):")
    print(
        "Example: 'Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.'"
    )
    print()

    task = input("> ").strip()

    if not task or task.lower() == "quit":
        logger.info("Exiting")
        return

    try:
        # Run the task
        logger.info(f"Task: {task}")
        logger.info("Starting accounting task workflow...")

        result = await run_accounting_task(
            prompt=task, file_content="", tripletex_credentials={}
        )

        # Print results
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)

        if result.get("success"):
            print("✓ Task completed successfully")
            print("\nPlanning phase output:")
            print(result.get("plan", "No plan"))
            print("\nExecution report:")
            print(result.get("report", "No report"))
        else:
            print("✗ Task failed")
            print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
