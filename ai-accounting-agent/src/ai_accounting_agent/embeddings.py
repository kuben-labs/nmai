"""Google embeddings provider using pydantic-ai.

This module provides direct integration with Google Gemini embedding models
without relying on machine-core or model-providers packages.
"""

import os
from typing import List, Optional

from loguru import logger


class GoogleEmbeddingProvider:
    """Wrapper for Google Gemini embedding model.

    This class provides a simple interface compatible with the existing
    ToolEmbedder class that expects an `embed` method.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """Initialize the Google embedding provider.

        Args:
            model_name: The embedding model name (defaults to gemini-embedding-001)
            api_key: Google API key (defaults to GCP_API_KEY env var)
            dimensions: Output embedding dimensions (defaults to EMBEDDING_DIMENSIONS env var or 3072)
        """
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "gemini-embedding-001"
        )
        self.api_key = api_key or os.getenv("GCP_API_KEY")
        self.dimensions = dimensions or int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

        if not self.api_key:
            raise ValueError(
                "Google API key is required for embeddings. "
                "Set GCP_API_KEY environment variable or pass api_key parameter."
            )

        # Import here to avoid issues if google-genai is not installed
        try:
            from google import genai  # type: ignore

            self._client = genai.Client(api_key=self.api_key)
            self._types = genai.types
            self._initialized = True
            logger.info(f"Initialized Google embedding provider: {self.model_name}")
        except ImportError:
            logger.error("google-genai package not installed")
            self._client = None
            self._types = None
            self._initialized = False

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not self._initialized or self._client is None:
            logger.warning("Google embedding provider not initialized")
            return [[] for _ in texts]

        try:
            # Use Google's new genai SDK
            result = self._client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=self._types.EmbedContentConfig(  # type: ignore
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.dimensions,
                ),
            )

            # Extract embeddings
            return [e.values for e in result.embeddings]

        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            return [[] for _ in texts]

    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed - delegates to sync for now.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        import asyncio

        return await asyncio.to_thread(self.embed, texts)


def get_embedding_provider(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> GoogleEmbeddingProvider:
    """Factory function to create a Google embedding provider.

    Args:
        model_name: The embedding model name
        api_key: Google API key
        dimensions: Output embedding dimensions

    Returns:
        GoogleEmbeddingProvider instance
    """
    return GoogleEmbeddingProvider(
        model_name=model_name,
        api_key=api_key,
        dimensions=dimensions,
    )
