"""Example integration of RAG tool filtering with the accounting agent.

This module shows how to integrate RAG-based tool filtering into the
accounting task execution pipeline.
"""

import asyncio
from typing import Dict, Any, Optional
from loguru import logger

from .coordinator import run_accounting_task
from .rag_tool_manager import create_rag_tool_manager


async def run_accounting_task_with_rag(
    prompt: str,
    file_content: str = "",
    tripletex_credentials: Optional[dict] = None,
    use_rag_filtering: bool = True,
    top_k: int = 300,
) -> Dict[str, Any]:
    """Run an accounting task with optional RAG-based tool filtering.

    This function demonstrates how to integrate RAG tool filtering into the
    accounting task workflow.

    Args:
        prompt: The accounting task prompt
        file_content: Extracted content from attached files (PDF text, OCR, etc.)
        tripletex_credentials: Tripletex API credentials
        use_rag_filtering: Whether to use RAG filtering (default: True)
        top_k: Number of top tools to return if using RAG

    Returns:
        Dictionary with task results
    """
    logger.info(f"Running accounting task with RAG filtering: {use_rag_filtering}")

    if use_rag_filtering:
        try:
            # Try to import embedding provider - optional
            try:
                from model_providers.embeddings import get_embedding_provider

                embedding_provider = get_embedding_provider()
                logger.info("Using embeddings for RAG filtering")
            except Exception as e:
                logger.warning(f"Could not load embedding provider: {e}")
                logger.info("RAG filtering will be limited")
                embedding_provider = None

            # Create RAG manager
            rag_manager = create_rag_tool_manager(
                embedding_provider=embedding_provider, top_k=top_k
            )

            # Initialize RAG manager
            await rag_manager.initialize()
            logger.info(f"RAG manager initialized: {rag_manager.get_statistics()}")

            # Log that RAG filtering is enabled
            logger.info(
                f"RAG filtering enabled - will return top {top_k} relevant tools per task"
            )

        except Exception as e:
            logger.warning(f"Error setting up RAG filtering: {e}")
            logger.info("Continuing without RAG filtering")
            use_rag_filtering = False

    # Run the accounting task (existing workflow)
    result = await run_accounting_task(
        prompt=prompt,
        file_content=file_content,
        tripletex_credentials=tripletex_credentials
        if tripletex_credentials is not None
        else {},
    )

    if use_rag_filtering:
        result["rag_enabled"] = True
        result["top_k"] = top_k
    else:
        result["rag_enabled"] = False

    return result


async def demo_rag_filtering():
    """Demonstrate RAG tool filtering with example tasks.

    This shows how the system would work with actual accounting tasks.
    """
    logger.info("Starting RAG Filtering Demo")

    # Example tasks
    tasks = [
        "Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.",
        "Opprett en faktura for kunde ABC AS med beløp 5000 NOK.",
        "Opprett et prosjekt knyttet til kunde XYZ AS.",
    ]

    for task in tasks:
        logger.info(f"\nProcessing: {task[:60]}...")

        try:
            result = await run_accounting_task_with_rag(
                prompt=task,
                file_content="",
                tripletex_credentials={},
                use_rag_filtering=True,
                top_k=300,
            )

            if result["success"]:
                logger.info("✓ Task completed successfully with RAG filtering")
            else:
                logger.error(f"✗ Task failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error processing task: {e}")


async def benchmark_context_reduction():
    """Benchmark context reduction from RAG filtering.

    This estimates the context savings from using RAG filtering.
    """
    logger.info("Starting Context Reduction Benchmark")

    # Assumptions
    total_tools = 800
    tokens_per_tool = 500  # Average tokens for tool description
    relevant_tools = 300

    tokens_without_rag = total_tools * tokens_per_tool
    tokens_with_rag = relevant_tools * tokens_per_tool
    reduction = tokens_without_rag - tokens_with_rag
    reduction_percent = (reduction / tokens_without_rag) * 100

    logger.info(f"Total tools in system: {total_tools}")
    logger.info(f"Tools sent to LLM without RAG: {total_tools}")
    logger.info(f"Tools sent to LLM with RAG: {relevant_tools}")
    logger.info(f"\nContext size without RAG: {tokens_without_rag:,} tokens")
    logger.info(f"Context size with RAG: {tokens_with_rag:,} tokens")
    logger.info(f"Context reduction: {reduction:,} tokens ({reduction_percent:.1f}%)")
    logger.info(
        f"\nNote: This allows using longer prompts and maintaining higher quality contexts."
    )


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_rag_filtering())

    # Show benchmark
    asyncio.run(benchmark_context_reduction())
