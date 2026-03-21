"""RAG-based tool filtering system for reducing context size.

This module provides functionality to:
1. Extract and embed all available MCP tools
2. Store embeddings in a vector database (LanceDB)
3. Filter tools based on semantic similarity to task prompts
4. Return relevant tool names (set[str]) for use with FilteredToolset

The actual tool filtering is done by pydantic-ai's native FilteredToolset,
which preserves full parameter schemas. This module only handles the
embedding index and semantic search to determine WHICH tools are relevant.
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass

from loguru import logger

try:
    import lancedb
except ImportError:
    lancedb = None


@dataclass
class ToolMetadata:
    """Metadata for a single tool."""

    name: str
    description: str
    parameters: Dict[str, Any]
    category: Optional[str] = None
    embedding: Optional[List[float]] = None


class ToolEmbedder:
    """Handles embedding of tool descriptions for semantic search."""

    def __init__(self, embedding_provider=None):
        """Initialize the tool embedder.

        Args:
            embedding_provider: Optional embedding provider (GoogleEmbeddingProvider).
                               If None, will try to get from environment.
        """
        self.embedding_provider = embedding_provider
        # If embedding_provider has a provider attribute, extract the actual provider
        if embedding_provider and hasattr(embedding_provider, "provider"):
            self.provider = embedding_provider.provider
        else:
            self.provider = embedding_provider

    async def embed_tools_batch(
        self, tools: List[ToolMetadata], batch_size: int = 32
    ) -> List[ToolMetadata]:
        """Embed multiple tools in batches for efficiency.

        Args:
            tools: List of tools to embed
            batch_size: Number of tools to embed per batch

        Returns:
            List of tools with embeddings added
        """
        if self.provider is None:
            logger.warning("No embedding provider configured, skipping embeddings")
            return tools

        embedded_tools = []
        total = len(tools)

        for i in range(0, total, batch_size):
            batch = tools[i : i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1} ({len(batch)} tools)")

            try:
                # Create text representations for embedding
                tool_texts = []
                for tool in batch:
                    # Limit parameters JSON to first 2000 chars for embedding
                    params_str = json.dumps(tool.parameters, indent=2, default=str)
                    if len(params_str) > 2000:
                        params_str = params_str[:2000] + "..."

                    tool_text = f"""
Tool: {tool.name}
Description: {tool.description}
Parameters: {params_str}
                    """.strip()
                    tool_texts.append(tool_text)

                # Get embeddings for the batch
                embeddings = await asyncio.to_thread(self.provider.embed, tool_texts)

                # Assign embeddings to tools
                for tool, embedding in zip(batch, embeddings):
                    if embedding:
                        tool.embedding = embedding
                    embedded_tools.append(tool)

            except Exception as e:
                logger.warning(f"Failed to embed batch: {e}")
                embedded_tools.extend(batch)

        return embedded_tools

    async def embed_text(self, text: str) -> List[float]:
        """Embed a text string (e.g., task prompt).

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.provider is None:
            logger.warning("No embedding provider configured")
            return []

        try:
            embeddings = await asyncio.to_thread(self.provider.embed, [text])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            return []
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            return []


class ToolVectorStore:
    """Manages tool embeddings and vector similarity search using LanceDB."""

    def __init__(self, db_path: str = ".tool_embeddings"):
        """Initialize the vector store using LanceDB.

        Args:
            db_path: Path to store the LanceDB database
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)

        if lancedb is None:
            raise ImportError(
                "lancedb is required for RAG tool filtering. Install with: uv add lancedb"
            )

        self.db = lancedb.connect(str(self.db_path))
        self.table = None
        self.tools: Dict[str, ToolMetadata] = {}

        # Load existing tools from database
        self._load_from_disk()

    def _load_from_disk(self):
        """Load tools from LanceDB if the table exists."""
        try:
            if "tools" in self.db.table_names():
                self.table = self.db.open_table("tools")
                records = self.table.search().limit(10000).to_list()

                for record in records:
                    tool = ToolMetadata(
                        name=record["name"],
                        description=record["description"],
                        parameters=record.get("parameters", {}),
                        category=record.get("category"),
                        embedding=record.get("embedding"),
                    )
                    self.tools[record["name"]] = tool

                logger.info(f"Loaded {len(self.tools)} tools from LanceDB")
            else:
                logger.debug("No existing tools table in LanceDB")

        except Exception as e:
            logger.warning(f"Failed to load tools from LanceDB: {e}")

    def _save_to_disk(self):
        """Save tools to LanceDB with embeddings."""
        try:
            records = []
            for tool_name, tool in self.tools.items():
                records.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": json.dumps(tool.parameters, default=str),
                        "category": tool.category,
                        "embedding": tool.embedding,
                    }
                )

            if not records:
                logger.warning("No tools to save")
                return

            if self.table is None:
                self.table = self.db.create_table(
                    "tools", data=records, mode="overwrite"
                )
            else:
                self.db.drop_table("tools")
                self.table = self.db.create_table(
                    "tools", data=records, mode="overwrite"
                )

            logger.info(f"Saved {len(records)} tools to LanceDB with embeddings")

        except Exception as e:
            logger.error(f"Failed to save tools to LanceDB: {e}")
            raise

    def add_tool(self, tool: ToolMetadata) -> None:
        """Add a tool to the vector store."""
        self.tools[tool.name] = tool

    def find_similar_tools(
        self, query_embedding: List[float], top_k: int = 100
    ) -> List[Tuple[str, float]]:
        """Find similar tools using LanceDB vector search.

        Args:
            query_embedding: Embedding of the query (task prompt)
            top_k: Number of top results to return

        Returns:
            List of (tool_name, similarity_score) tuples, sorted by relevance
        """
        if not query_embedding:
            logger.warning("Empty query embedding, returning all tools")
            return [(name, 0.0) for name in list(self.tools.keys())[:top_k]]

        if self.table is None:
            logger.warning("No tools indexed yet, returning all tools")
            return [(name, 0.0) for name in list(self.tools.keys())[:top_k]]

        try:
            results = self.table.search(query_embedding).limit(top_k).to_list()

            tool_scores = []
            for result in results:
                tool_name = result.get("name", "")
                distance = result.get("_distance", 0.0)
                similarity = 1.0 / (1.0 + distance)
                tool_scores.append((tool_name, similarity))

            logger.debug(f"LanceDB search found {len(tool_scores)} results")
            return tool_scores

        except Exception as e:
            logger.error(f"LanceDB search failed: {e}")
            return [(name, 0.0) for name in list(self.tools.keys())[:top_k]]

    def get_all_tool_names(self) -> Set[str]:
        """Get all tool names in the store."""
        return set(self.tools.keys())


async def index_mcp_tools(
    toolsets: List[Any],
    vector_store: ToolVectorStore,
    embedder: ToolEmbedder,
) -> None:
    """Index all tools from MCP toolsets into the vector store.

    Args:
        toolsets: List of MCP toolsets
        vector_store: The vector store to populate
        embedder: The embedder to use for creating embeddings
    """
    logger.info("Starting MCP tool indexing...")

    total_tools = 0
    for toolset in toolsets:
        try:
            tools_list = await toolset.list_tools()  # type: ignore
            logger.info(
                f"Extracted {len(tools_list)} tools from {toolset.__class__.__name__}"
            )

            # Convert to ToolMetadata objects
            tool_metadata_list = []
            for mcp_tool in tools_list:
                tool_metadata = ToolMetadata(
                    name=mcp_tool.name,
                    description=mcp_tool.description or "",
                    parameters=mcp_tool.inputSchema or {},
                )
                tool_metadata_list.append(tool_metadata)

            # Embed tools in batches
            embedded_tools = await embedder.embed_tools_batch(
                tool_metadata_list, batch_size=32
            )

            # Store all tools
            for tool in embedded_tools:
                vector_store.add_tool(tool)
                total_tools += 1

            # Save to disk
            vector_store._save_to_disk()

        except Exception as e:
            logger.error(f"Failed to extract tools from toolset: {e}")
            import traceback

            traceback.print_exc()

    logger.info(f"Tool indexing complete: {total_tools} tools indexed")


async def index_openapi_tools(
    spec: Dict[str, Any],
    vector_store: ToolVectorStore,
    embedder: ToolEmbedder,
) -> None:
    """Index tools from an OpenAPI spec into the vector store.

    This replaces index_mcp_tools for the direct-API architecture.
    Extracts tool names and descriptions from the OpenAPI spec's
    operation IDs and summaries.

    Args:
        spec: Parsed OpenAPI spec JSON
        vector_store: The vector store to populate
        embedder: The embedder to use for creating embeddings
    """
    logger.info("Starting OpenAPI tool indexing...")

    tool_metadata_list = []
    paths = spec.get("paths", {})

    for path, path_item in paths.items():
        for method in ["get", "post", "put", "delete", "patch"]:
            operation = path_item.get(method)
            if not operation:
                continue

            operation_id = operation.get("operationId", "")
            if not operation_id:
                continue

            tool_name = operation_id.strip("[]").replace(" ", "_")
            description = operation.get("summary", "") or operation.get(
                "description", ""
            )
            if not description:
                description = f"{method.upper()} {path}"

            # Build a simple representation for embedding
            params = {}
            for param in operation.get("parameters", []):
                name = param.get("name", "")
                if name:
                    params[name] = param.get("schema", {}).get("type", "string")

            tool_metadata = ToolMetadata(
                name=tool_name,
                description=description,
                parameters=params,
            )
            tool_metadata_list.append(tool_metadata)

    logger.info(f"Found {len(tool_metadata_list)} tools in OpenAPI spec")

    # Embed in batches
    embedded_tools = await embedder.embed_tools_batch(tool_metadata_list, batch_size=32)

    # Store
    for tool in embedded_tools:
        vector_store.add_tool(tool)

    vector_store._save_to_disk()
    logger.info(f"OpenAPI tool indexing complete: {len(embedded_tools)} tools indexed")


async def get_relevant_tool_names(
    vector_store: ToolVectorStore,
    embedder: ToolEmbedder,
    task_prompt: str,
    top_k: int = 100,
) -> Set[str]:
    """Get the set of relevant tool names for a task using RAG similarity search.

    This is the main entry point for tool filtering. Returns a set of tool names
    that can be used with pydantic-ai's FilteredToolset to filter the MCP toolset
    while preserving full parameter schemas.

    Args:
        vector_store: The populated vector store
        embedder: The embedder for embedding the task prompt
        task_prompt: The task prompt to find relevant tools for
        top_k: Number of top relevant tools to return

    Returns:
        Set of relevant tool names
    """
    try:
        # Embed the task prompt
        query_embedding = await embedder.embed_text(task_prompt)

        # Find similar tools
        similar_tool_scores = vector_store.find_similar_tools(
            query_embedding, top_k=top_k
        )

        # Extract just the tool names
        relevant_names = {tool_name for tool_name, score in similar_tool_scores}

        logger.info(
            f"RAG search: Found {len(relevant_names)} relevant tools "
            f"(from {len(vector_store.tools)} total) for task: {task_prompt[:80]}..."
        )

        return relevant_names

    except Exception as e:
        logger.error(f"Error getting relevant tools: {e}")
        # Fallback: return all tools
        return vector_store.get_all_tool_names()
