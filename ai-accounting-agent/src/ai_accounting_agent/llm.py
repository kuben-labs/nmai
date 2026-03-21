"""Google Gemini LLM provider using pydantic-ai.

This module provides direct integration with Google Gemini models
without relying on machine-core or model-providers packages.
"""

import os
from typing import Optional

from loguru import logger
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider


def get_google_model(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    thinking_budget: Optional[int] = None,
) -> GoogleModel:
    """Get a Google Gemini model instance for pydantic-ai.

    Args:
        model_name: The model name (defaults to LLM_MODEL or GCP_MODEL env var)
        api_key: Google API key (defaults to GCP_API_KEY env var)
        thinking_budget: Optional thinking budget for reasoning models

    Returns:
        GoogleModel instance ready for use with pydantic-ai Agent
    """
    # Get model name from params or environment
    model = (
        model_name
        or os.getenv("LLM_MODEL")
        or os.getenv("GCP_MODEL", "gemini-2.5-flash")
    )

    # Get API key from params or environment
    key = api_key or os.getenv("GCP_API_KEY")

    if not key:
        raise ValueError(
            "Google API key is required. Set GCP_API_KEY environment variable "
            "or pass api_key parameter."
        )

    logger.info(f"Initializing Google model: {model}")

    # Create provider with API key
    provider = GoogleProvider(api_key=key)

    # Create model with optional thinking config
    if thinking_budget is not None:
        settings = GoogleModelSettings(
            google_thinking_config={"thinking_budget": thinking_budget}
        )
        google_model = GoogleModel(model, provider=provider, model_settings=settings)
    else:
        google_model = GoogleModel(model, provider=provider)

    return google_model


def get_default_model() -> GoogleModel:
    """Get the default Google model from environment configuration.

    Returns:
        GoogleModel instance with default settings
    """
    return get_google_model()
