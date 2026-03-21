"""Simplified RAG tool manager - handles embedding index and returns relevant tool names.

This module manages the embedding index lifecycle and provides a simple interface
to get relevant tool names for a task. The actual tool filtering is done by
pydantic-ai's native FilteredToolset, which preserves full parameter schemas.
"""

from typing import List, Any, Optional, Set
from loguru import logger

from .rag_tool_filter import (
    ToolVectorStore,
    ToolEmbedder,
    index_mcp_tools,
    get_relevant_tool_names,
)


class RAGToolManager:
    """Manages RAG-based tool filtering for an agent.

    This manager handles:
    - Lazy initialization of embedder and vector store
    - Tool indexing and embedding (once at startup)
    - Returning relevant tool names per task (fast lookup)
    """

    def __init__(
        self,
        embedding_provider: Optional[Any] = None,
        vector_store_path: str = ".tool_embeddings",
        top_k: int = 100,
    ):
        """Initialize the RAG tool manager.

        Args:
            embedding_provider: Optional embedding provider for semantic search
            vector_store_path: Path to store tool embeddings
            top_k: Number of top relevant tools to return
        """
        self.embedding_provider = embedding_provider
        self.vector_store_path = vector_store_path
        self.top_k = top_k
        self.embedder: Optional[ToolEmbedder] = None
        self.vector_store: Optional[ToolVectorStore] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize embedder and vector store."""
        if self._initialized:
            return

        logger.info("Initializing RAG tool manager...")

        try:
            self.embedder = ToolEmbedder(embedding_provider=self.embedding_provider)
            logger.info("Embedder initialized")

            self.vector_store = ToolVectorStore(db_path=self.vector_store_path)
            logger.info(
                f"Vector store initialized with {len(self.vector_store.tools)} cached tools"
            )

            self._initialized = True
            logger.info("RAG tool manager initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize RAG tool manager: {e}")
            raise

    async def index_toolsets(self, toolsets: List[Any]) -> None:
        """Index tools from MCP toolsets into the vector store.

        Args:
            toolsets: List of MCP toolsets to index
        """
        if not self._initialized:
            await self.initialize()

        if self.vector_store is None or self.embedder is None:
            logger.error("Vector store or embedder not initialized")
            return

        # Skip re-indexing if we already have tools with embeddings
        if len(self.vector_store.tools) > 0:
            tools_with_embeddings = sum(
                1 for tool in self.vector_store.tools.values() if tool.embedding
            )
            if tools_with_embeddings > 0:
                logger.info(
                    f"Skipping tool indexing: {len(self.vector_store.tools)} tools already indexed"
                )
                return

        await index_mcp_tools(
            toolsets=toolsets,
            vector_store=self.vector_store,
            embedder=self.embedder,
        )

    async def get_relevant_names(
        self, task_prompt: str, top_k: Optional[int] = None
    ) -> Set[str]:
        """Get the set of relevant tool names for a task.

        This is the main interface for tool filtering. Returns a set of tool names
        that can be used with FilteredToolset to filter tools while preserving
        full parameter schemas.

        Args:
            task_prompt: The task prompt to find relevant tools for
            top_k: Override for the number of tools to return

        Returns:
            Set of relevant tool names
        """
        if not self._initialized:
            await self.initialize()

        if self.vector_store is None or self.embedder is None:
            logger.error("Vector store or embedder not initialized")
            return set()

        k = top_k or self.top_k

        return await get_relevant_tool_names(
            vector_store=self.vector_store,
            embedder=self.embedder,
            task_prompt=task_prompt,
            top_k=k,
        )

    def get_statistics(self) -> dict:
        """Get statistics about the tool database."""
        if not self._initialized or self.vector_store is None:
            return {"status": "not initialized"}

        total_tools = len(self.vector_store.tools)
        tools_with_embeddings = sum(
            1 for tool in self.vector_store.tools.values() if tool.embedding
        )

        return {
            "status": "initialized",
            "total_tools": total_tools,
            "tools_with_embeddings": tools_with_embeddings,
            "embedding_ratio": tools_with_embeddings / total_tools
            if total_tools > 0
            else 0,
            "vector_store_path": self.vector_store_path,
        }


def create_rag_tool_manager(
    embedding_provider: Optional[Any] = None,
    vector_store_path: str = ".tool_embeddings",
    top_k: int = 100,
) -> RAGToolManager:
    """Factory function to create a RAG tool manager.

    Args:
        embedding_provider: Optional embedding provider
        vector_store_path: Path to store embeddings
        top_k: Number of top tools to return

    Returns:
        RAGToolManager instance
    """
    return RAGToolManager(
        embedding_provider=embedding_provider,
        vector_store_path=vector_store_path,
        top_k=top_k,
    )
