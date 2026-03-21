"""RAG-based tool filtering system for reducing context size.

This module provides functionality to:
1. Extract and embed all available MCP tools
2. Store embeddings in a vector database (LanceDB)
3. Filter tools based on semantic similarity to task prompts
4. Return only relevant tools to stay within context limits
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from loguru import logger
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_ai._run_context import RunContext
from pydantic_ai.tools import ToolDefinition

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
            embedding_provider: Optional embedding provider (ResolvedEmbedding from model_providers).
                               If None, will try to get from environment.
        """
        self.embedding_provider = embedding_provider
        # If embedding_provider is a ResolvedEmbedding, extract the actual provider
        if embedding_provider and hasattr(embedding_provider, "provider"):
            self.provider = embedding_provider.provider
        else:
            self.provider = embedding_provider

    async def embed_tool(self, tool: ToolMetadata) -> ToolMetadata:
        """Embed a single tool's description.

        Args:
            tool: The tool metadata to embed

        Returns:
            Tool metadata with embedding added
        """
        if self.provider is None:
            logger.warning("No embedding provider configured, skipping embeddings")
            return tool

        try:
            # Create a rich text representation of the tool
            tool_text = f"""
Tool: {tool.name}
Description: {tool.description}
Parameters: {json.dumps(tool.parameters, indent=2, default=str)}
            """.strip()

            # Get embedding using the provider's embed method
            # The embed method takes a list of strings and returns a list of embeddings
            embeddings = await asyncio.to_thread(self.provider.embed, [tool_text])
            if embeddings and len(embeddings) > 0:
                tool.embedding = embeddings[0]
            return tool

        except Exception as e:
            logger.warning(f"Failed to embed tool {tool.name}: {e}")
            return tool

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
                # Create rich text representations, limiting parameter size
                tool_texts = []
                for tool in batch:
                    # Limit parameters JSON to first 2000 chars to avoid huge requests
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
            # Use the provider's embed method
            embeddings = await asyncio.to_thread(self.provider.embed, [text])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            return []
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            return []


class ToolVectorStore:
    """Manages tool embeddings and vector similarity search using LanceDB.

    LanceDB provides:
    - Persistent vector storage on disk
    - Fast vector similarity search
    - Automatic embedding persistence
    """

    def __init__(self, db_path: str = ".tool_embeddings"):
        """Initialize the vector store using LanceDB.

        Args:
            db_path: Path to store the LanceDB database
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)

        # Initialize LanceDB
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
            # Try to open existing table
            if "tools" in self.db.table_names():
                self.table = self.db.open_table("tools")

                # Load all records from the table
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
            # Prepare data for LanceDB
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

            # Create or overwrite the table
            # LanceDB automatically creates vector index for 'embedding' column
            if self.table is None:
                self.table = self.db.create_table(
                    "tools", data=records, mode="overwrite"
                )
            else:
                # Delete old table and create new one with embeddings
                self.db.drop_table("tools")
                self.table = self.db.create_table(
                    "tools", data=records, mode="overwrite"
                )

            logger.info(f"Saved {len(records)} tools to LanceDB with embeddings")

        except Exception as e:
            logger.error(f"Failed to save tools to LanceDB: {e}")
            raise

    def add_tool(self, tool: ToolMetadata) -> None:
        """Add a tool to the vector store.

        Args:
            tool: The tool metadata to add
        """
        self.tools[tool.name] = tool

    def find_similar_tools(
        self, query_embedding: List[float], top_k: int = 300
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
            # LanceDB vector search returns results with score (distance)
            # Smaller distance = more similar
            results = self.table.search(query_embedding).limit(top_k).to_list()

            # Convert distance to similarity score (1 / (1 + distance))
            tool_scores = []
            for result in results:
                tool_name = result.get("name", "")
                distance = result.get("_distance", 0.0)
                # Convert distance to similarity: 1 / (1 + distance)
                similarity = 1.0 / (1.0 + distance)
                tool_scores.append((tool_name, similarity))

            logger.debug(f"LanceDB search found {len(tool_scores)} results")
            return tool_scores

        except Exception as e:
            logger.error(f"LanceDB search failed: {e}")
            # Fallback: return all tools without scores
            return [(name, 0.0) for name in list(self.tools.keys())[:top_k]]

    def get_all_tools(self) -> List[ToolMetadata]:
        """Get all tools in the store.

        Returns:
            List of all tools
        """
        return list(self.tools.values())


class RAGToolFilter(AbstractToolset):
    """Filters tools based on semantic similarity to task prompts.

    This wrapper reduces context size by only loading tools relevant to the
    current task, using RAG (Retrieval Augmented Generation) principles.
    """

    def __init__(
        self,
        wrapped_toolset: AbstractToolset,
        vector_store: ToolVectorStore,
        embedder: ToolEmbedder,
        task_prompt: str,
        top_k: int = 300,
    ):
        """Initialize the RAG tool filter.

        Args:
            wrapped_toolset: The underlying MCP toolset to wrap
            vector_store: The tool vector store for similarity search
            embedder: The embedder for semantic search
            task_prompt: The current task prompt to filter tools for
            top_k: Number of top relevant tools to return
        """
        self.wrapped_toolset = wrapped_toolset
        self.vector_store = vector_store
        self.embedder = embedder
        self.task_prompt = task_prompt
        self.top_k = top_k
        self._relevant_tool_names: Optional[set] = None
        self._original_class_name = wrapped_toolset.__class__.__name__

    @property
    def id(self) -> str:
        """Return a unique ID for this toolset.

        Required by AbstractToolset interface.
        """
        return f"rag-filtered-{self._original_class_name}"

    async def _get_relevant_tool_names(self) -> set:
        """Get the set of relevant tool names for this task.

        Returns:
            Set of tool names that are relevant
        """
        if self._relevant_tool_names is not None:
            return self._relevant_tool_names

        try:
            # Embed the task prompt
            query_embedding = await self.embedder.embed_text(self.task_prompt)

            # Find similar tools - now returns list of (tool_name, score) tuples
            similar_tool_scores = self.vector_store.find_similar_tools(
                query_embedding, top_k=self.top_k
            )

            # Extract just the tool names
            self._relevant_tool_names = {
                tool_name for tool_name, score in similar_tool_scores
            }

            logger.info(
                f"RAG Filter: Found {len(self._relevant_tool_names)} relevant tools "
                f"(from {len(self.vector_store.tools)} total) for task"
            )

            return self._relevant_tool_names

        except Exception as e:
            logger.error(f"Error getting relevant tools: {e}")
            # Fallback: return all tools
            return set(self.vector_store.tools.keys())

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """Get only the relevant tools for this task with minimal schemas.

        Required by AbstractToolset interface.

        Args:
            ctx: The run context

        Returns:
            Dictionary of filtered tools with minimal schemas
        """
        try:
            # Get all tools from wrapped toolset
            all_tools = await self.wrapped_toolset.get_tools(ctx)
            logger.info(
                f"RAGToolFilter.get_tools(): Received {len(all_tools)} tools from wrapped toolset"
            )

            # Get relevant tool names
            relevant_names = await self._get_relevant_tool_names()
            logger.info(
                f"RAGToolFilter.get_tools(): Filtering to {len(relevant_names)} relevant tools"
            )

            # Filter tools - keep only relevant ones
            filtered_tools = {
                name: tool for name, tool in all_tools.items() if name in relevant_names
            }

            # CRITICAL: Replace huge parameter schemas with minimal valid schemas
            # This reduces token usage while keeping tools functional
            simplified_tools = {}
            for name, tool in filtered_tools.items():
                try:
                    # Replace the tool_def's parameters_json_schema with a minimal but valid schema
                    # that still accepts the required parameters
                    if hasattr(tool, "tool_def") and hasattr(
                        tool.tool_def, "parameters_json_schema"
                    ):
                        # Keep the original to extract required fields
                        original_schema = tool.tool_def.parameters_json_schema or {}

                        # Create minimal schema with just required properties
                        required_fields = original_schema.get("required", [])
                        properties = {}

                        # Add minimal definitions for required fields
                        for field in required_fields:
                            properties[field] = {
                                "type": "string",
                                "description": f"Required parameter: {field}",
                            }

                        # Create new minimal schema
                        minimal_schema = {
                            "type": "object",
                            "properties": properties,
                            "additionalProperties": True,  # Allow other fields
                        }

                        if required_fields:
                            minimal_schema["required"] = required_fields

                        # Update the tool definition with minimal schema
                        tool.tool_def.parameters_json_schema = minimal_schema

                    simplified_tools[name] = tool
                except Exception as e:
                    logger.debug(f"Could not simplify tool {name}: {e}")
                    simplified_tools[name] = tool

            logger.info(
                f"RAGToolFilter.get_tools(): Filtered tools from {len(all_tools)} to {len(simplified_tools)} "
                f"and replaced parameter schemas with minimal schemas"
            )

            return simplified_tools

        except Exception as e:
            logger.error(f"Error in get_tools: {e}")
            import traceback

            traceback.print_exc()
            return await self.wrapped_toolset.get_tools(ctx)

    async def list_tools(self) -> list:
        """List only the relevant tools for this task with minimal schemas.

        This method is called by some agents to get tool information.
        We replace huge parameter schemas with minimal but valid schemas.

        Returns:
            List of filtered tool definitions with minimal schemas
        """
        try:
            # Get all tools from wrapped toolset
            all_tools = await self.wrapped_toolset.list_tools()
            logger.info(
                f"RAGToolFilter.list_tools(): Received {len(all_tools)} tools from wrapped toolset"
            )

            # Get relevant tool names
            relevant_names = await self._get_relevant_tool_names()
            logger.info(
                f"RAGToolFilter.list_tools(): Filtering to {len(relevant_names)} relevant tools"
            )

            # Filter tools - keep only relevant ones
            filtered_tools = [tool for tool in all_tools if tool.name in relevant_names]

            # CRITICAL: Replace huge parameter schemas with minimal valid schemas
            # This reduces token usage while keeping tools functional
            simplified_tools = []
            for tool in filtered_tools:
                try:
                    # Create a new ToolDefinition with minimal schema
                    if isinstance(tool, ToolDefinition):
                        # Extract required fields from original schema
                        original_schema = tool.parameters_json_schema or {}
                        required_fields = original_schema.get("required", [])

                        # Create minimal schema with just required properties
                        properties = {}
                        for field in required_fields:
                            properties[field] = {
                                "type": "string",
                                "description": f"Required parameter: {field}",
                            }

                        minimal_schema = {
                            "type": "object",
                            "properties": properties,
                            "additionalProperties": True,
                        }

                        if required_fields:
                            minimal_schema["required"] = required_fields

                        # Create new ToolDefinition with minimal schema
                        simplified_tool = ToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters_json_schema=minimal_schema,
                        )
                        simplified_tools.append(simplified_tool)
                    else:
                        # Fallback for non-ToolDefinition objects
                        simplified_tools.append(tool)
                except Exception as e:
                    logger.debug(f"Could not simplify tool {tool.name}: {e}")
                    simplified_tools.append(tool)

            logger.info(
                f"RAGToolFilter.list_tools(): Filtered tools from {len(all_tools)} to {len(simplified_tools)} "
                f"and replaced parameter schemas with minimal schemas"
            )

            return simplified_tools

        except Exception as e:
            logger.error(f"Error in list_tools: {e}")
            import traceback

            traceback.print_exc()
            return await self.wrapped_toolset.list_tools()

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        """Call a tool, filtering out tools not in the relevant set.

        Args:
            name: Name of the tool to call
            tool_args: Input parameters for the tool
            ctx: The run context
            tool: The tool definition

        Returns:
            The tool result

        Raises:
            ValueError: If the tool is not in the relevant set
        """
        try:
            relevant_names = await self._get_relevant_tool_names()

            if name not in relevant_names:
                error_msg = (
                    f"Tool '{name}' is not in the relevant tool set for this task. "
                    f"Use one of: {', '.join(sorted(relevant_names))}"
                )
                logger.warning(error_msg)
                raise ValueError(error_msg)

            # Delegate to wrapped toolset with correct signature
            result = await self.wrapped_toolset.call_tool(name, tool_args, ctx, tool)
            return result

        except ValueError:
            # Re-raise ValueError as-is (tool not in relevant set)
            raise

            return result

        except ValueError:
            # Re-raise ValueError as-is (tool not in relevant set)
            raise
        except Exception as e:
            # For other errors (API errors, validation errors), log as warning and return empty value
            logger.warning(
                f"Tool {name} returned error (treating as empty response): {e}"
            )
            # Return appropriate empty value based on tool return type
            return "" if tool.return_type == str else {}

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped toolset."""
        return getattr(self.wrapped_toolset, name)


async def index_mcp_tools(
    toolsets: List[AbstractToolset[Any]],
    vector_store: ToolVectorStore,
    embedder: ToolEmbedder,
) -> None:
    """Index all tools from MCP toolsets into the vector store.

    This extracts tools from MCP toolsets using list_tools() and embeds them in batches.

    Args:
        toolsets: List of MCP toolsets
        vector_store: The vector store to populate
        embedder: The embedder to use for creating embeddings
    """
    logger.info("Starting MCP tool indexing...")

    total_tools = 0
    for toolset in toolsets:
        try:
            # Use list_tools() to get all tools from the toolset (it's async)
            tools_list = await toolset.list_tools()
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

            # Embed tools in batches for efficiency
            embedded_tools = await embedder.embed_tools_batch(
                tool_metadata_list, batch_size=32
            )

            # Store all tools in vector store
            for tool in embedded_tools:
                vector_store.add_tool(tool)
                total_tools += 1

            # Save to disk after processing each toolset
            vector_store._save_to_disk()

        except Exception as e:
            logger.error(f"Failed to extract tools from toolset: {e}")
            import traceback

            traceback.print_exc()

    logger.info(f"Tool indexing complete: {total_tools} tools indexed")


async def create_filtered_toolsets(
    toolsets: List[AbstractToolset],
    vector_store: ToolVectorStore,
    embedder: ToolEmbedder,
    task_prompt: str,
    top_k: int = 300,
) -> List[RAGToolFilter]:
    """Create filtered versions of toolsets for a specific task.

    Args:
        toolsets: Original MCP toolsets
        vector_store: The populated vector store
        embedder: The embedder
        task_prompt: The task prompt to filter for
        top_k: Number of top relevant tools to include

    Returns:
        List of filtered toolsets
    """
    logger.info(f"Creating RAG-filtered toolsets for task: {task_prompt[:100]}...")

    filtered = []
    for toolset in toolsets:
        filtered_toolset = RAGToolFilter(
            wrapped_toolset=toolset,
            vector_store=vector_store,
            embedder=embedder,
            task_prompt=task_prompt,
            top_k=top_k,
        )
        filtered.append(filtered_toolset)

    logger.info(f"Created {len(filtered)} filtered toolsets")
    return filtered
