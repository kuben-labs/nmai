"""LLM provider for pydantic-ai.

Supports:
- Google Gemini (via API key or Vertex AI)
- Anthropic Claude (via Vertex AI — no Anthropic API key needed)
- Configurable via LLM_PROVIDER and LLM_MODEL env vars
"""

import os
from typing import Optional, Union

from loguru import logger


def get_model(
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
) -> Union["GoogleModel", "AnthropicModel"]:
    """Get an LLM model instance for pydantic-ai.

    Supports multiple providers:
    - "google" (default): Google Gemini via API key
    - "vertex-gemini": Google Gemini via Vertex AI
    - "vertex-claude": Anthropic Claude via Google Vertex AI

    Configure via env vars:
        LLM_PROVIDER=google|vertex-gemini|vertex-claude
        LLM_MODEL=gemini-2.5-flash|claude-opus-4-6|etc.
        GCP_API_KEY=... (for google provider)
        GCP_PROJECT=... (for vertex providers)
        GCP_LOCATION=... (for vertex providers, default us-central1)

    Args:
        model_name: Override model name
        provider: Override provider name

    Returns:
        Model instance ready for pydantic-ai Agent
    """
    llm_provider = provider or os.getenv("LLM_PROVIDER", "google")
    model = (
        model_name
        or os.getenv("LLM_MODEL")
        or os.getenv("GCP_MODEL", "gemini-2.5-flash")
    )

    logger.info(f"Initializing model: provider={llm_provider}, model={model}")

    if llm_provider == "vertex-claude":
        return _get_vertex_claude(model)
    elif llm_provider == "vertex-gemini":
        return _get_vertex_gemini(model)
    else:
        return _get_google_gla(model)


def _get_google_gla(model: str):
    """Google Gemini via Generative Language API (API key)."""
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    api_key = os.getenv("GCP_API_KEY")
    if not api_key:
        raise ValueError("GCP_API_KEY env var required for google provider")

    provider = GoogleProvider(api_key=api_key)
    return GoogleModel(model, provider=provider)


def _get_vertex_gemini(model: str):
    """Google Gemini via Vertex AI (uses gcloud ADC)."""
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")

    provider = GoogleProvider(
        vertexai=True,
        project=project,
        location=location,
    )
    return GoogleModel(model, provider=provider)


def _get_vertex_claude(model: str):
    """Anthropic Claude via Google Vertex AI (uses gcloud ADC).

    No Anthropic API key needed — uses your Google Cloud credentials.
    Make sure Claude models are enabled in your Vertex AI Model Garden.
    """
    from anthropic import AsyncAnthropicVertex
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-east5")  # Claude often in us-east5

    if not project:
        raise ValueError("GCP_PROJECT env var required for vertex-claude provider")

    vertex_client = AsyncAnthropicVertex(
        region=location,
        project_id=project,
    )
    provider = AnthropicProvider(anthropic_client=vertex_client)
    return AnthropicModel(model, provider=provider)


# Backwards compatibility
def get_google_model(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    thinking_budget: Optional[int] = None,
):
    """Legacy function — use get_model() instead."""
    return get_model(model_name=model_name)


def get_default_model():
    """Get the default model from environment configuration."""
    return get_model()
