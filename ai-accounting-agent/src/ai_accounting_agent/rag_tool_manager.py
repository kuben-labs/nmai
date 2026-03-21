"""Integration of RAG tool filtering into the agent initialization pipeline."""

import asyncio
from typing import List, Any, Optional
from loguru import logger

from pydantic_ai.toolsets import AbstractToolset

from .rag_tool_filter import (
    RAGToolFilter,
    ToolVectorStore,
    ToolEmbedder,
    index_mcp_tools,
)


class RAGToolManager:
    """Manages RAG-based tool filtering for an agent.

    This manager handles:
    - Lazy initialization of embedder and vector store
    - Tool indexing and embedding
    - Creation of filtered toolsets per task
    """

    def __init__(
        self,
        embedding_provider: Optional[Any] = None,
        vector_store_path: str = ".tool_embeddings",
        top_k: int = 300,
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
            # Initialize embedder
            self.embedder = ToolEmbedder(embedding_provider=self.embedding_provider)
            logger.info("Embedder initialized")

            # Initialize vector store
            self.vector_store = ToolVectorStore(db_path=self.vector_store_path)
            logger.info(
                f"Vector store initialized with {len(self.vector_store.tools)} tools"
            )

            self._initialized = True
            logger.info("RAG tool manager initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize RAG tool manager: {e}")
            raise

    async def index_toolsets(self, toolsets: List[AbstractToolset[Any]]) -> None:
        """Index tools from toolsets.

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
                    f"Skipping tool indexing: {len(self.vector_store.tools)} tools already indexed "
                    f"with embeddings"
                )
                return

        await index_mcp_tools(
            toolsets=toolsets,
            vector_store=self.vector_store,
            embedder=self.embedder,
        )

    async def filter_tools_async(
        self, query: str, top_k: Optional[int] = None
    ) -> List[dict]:
        """Filter tools based on semantic relevance to a query (async).

        Args:
            query: The query to filter tools for
            top_k: Number of top tools to return (defaults to manager's top_k)

        Returns:
            List of filtered tool metadata dicts with similarity scores
        """
        if not self._initialized:
            await self.initialize()

        if self.vector_store is None or self.embedder is None:
            logger.error("Vector store or embedder not initialized")
            return []

        k = top_k or self.top_k

        # Embed the query
        query_embedding = await self.embedder.embed_text(query)
        if not query_embedding:
            logger.warning("Failed to embed query, returning random tools")
            all_tools = list(self.vector_store.tools.items())
            random_tools = all_tools[:k]
            return [
                {
                    "id": name,
                    "name": tool.name,
                    "description": tool.description[:100] + "..."
                    if len(tool.description) > 100
                    else tool.description,
                    "score": 0.0,
                }
                for name, tool in random_tools
            ]

        # Find similar tools - now returns list of (tool_name, score) tuples
        similar_tool_scores = self.vector_store.find_similar_tools(
            query_embedding, top_k=k
        )

        # Format results with actual similarity scores from LanceDB
        filtered_tools = []
        for tool_name, score in similar_tool_scores:
            if tool_name in self.vector_store.tools:
                tool = self.vector_store.tools[tool_name]
                filtered_tools.append(
                    {
                        "id": tool_name,
                        "name": tool.name,
                        "description": tool.description[:100] + "..."
                        if len(tool.description) > 100
                        else tool.description,
                        "score": score,
                    }
                )

        logger.info(f"RAG Filter: Found {len(filtered_tools)} relevant tools for query")
        return filtered_tools

    async def create_filtered_toolsets(
        self, toolsets: List[Any], task_prompt: str
    ) -> List[RAGToolFilter]:
        """Create filtered versions of toolsets for a specific task.

        Args:
            toolsets: Original MCP toolsets
            task_prompt: The task prompt to filter tools for

        Returns:
            List of RAG-filtered toolsets
        """
        if not self._initialized:
            await self.initialize()

        if self.vector_store is None or self.embedder is None:
            logger.error("Vector store or embedder not initialized")
            return []

        logger.info(f"Creating RAG-filtered toolsets for task: {task_prompt[:100]}...")

        filtered = []
        for toolset in toolsets:
            filtered_toolset = RAGToolFilter(
                wrapped_toolset=toolset,
                vector_store=self.vector_store,
                embedder=self.embedder,
                task_prompt=task_prompt,
                top_k=self.top_k,
            )
            filtered.append(filtered_toolset)

        logger.info(f"Created {len(filtered)} RAG-filtered toolsets")
        return filtered

    def get_statistics(self) -> dict:
        """Get statistics about the tool database.

        Returns:
            Dictionary with tool statistics
        """
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
    top_k: int = 300,
) -> RAGToolManager:
    """Factory function to create a RAG tool manager.

    Args:
        embedding_provider: Optional embedding provider
        vector_store_path: Path to store embeddings
        top_k: Number of top tools to return

    Returns:
        Initialized RAGToolManager instance
    """
    return RAGToolManager(
        embedding_provider=embedding_provider,
        vector_store_path=vector_store_path,
        top_k=top_k,
    )
