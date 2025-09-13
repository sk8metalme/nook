"""
Client factory for switching between Gemini and Claude clients.

This module provides a unified interface for creating AI clients,
allowing seamless switching between different providers.
"""

import os
from typing import Any, Union
from .gemini_client import GeminiClient, GeminiClientConfig, create_client as create_gemini_client
from .claude_client import ClaudeClient, ClaudeClientConfig, create_client as create_claude_client


def create_client(config: dict[str, Any] | None = None, **kwargs) -> Union[GeminiClient, ClaudeClient]:
    """
    Create an AI client based on environment configuration.

    The client type is determined by the AI_CLIENT_TYPE environment variable:
    - 'claude': Use Claude client
    - 'gemini' or unset: Use Gemini client (default)

    Parameters
    ----------
    config : dict[str, Any] | None
        Configuration dictionary for the client.
        If not provided, default values will be used.
    **kwargs
        Additional keyword arguments passed to the client constructor.

    Returns
    -------
    Union[GeminiClient, ClaudeClient]
        The configured AI client.

    Examples
    --------
    # Use default client (Gemini)
    client = create_client()

    # Force Claude client
    os.environ['AI_CLIENT_TYPE'] = 'claude'
    client = create_client()

    # Use Gemini with custom config
    os.environ['AI_CLIENT_TYPE'] = 'gemini'
    client = create_client({'temperature': 0.5})
    """
    client_type = os.environ.get("AI_CLIENT_TYPE", "gemini").lower()

    if client_type == "claude":
        return _create_claude_client(config, **kwargs)
    elif client_type == "gemini":
        return _create_gemini_client(config, **kwargs)
    else:
        raise ValueError(f"Unsupported AI_CLIENT_TYPE: {client_type}. Supported values: 'gemini', 'claude'")


def _create_gemini_client(config: dict[str, Any] | None = None, **kwargs) -> GeminiClient:
    """Create a Gemini client with the given configuration."""
    return create_gemini_client(config, **kwargs)


def _create_claude_client(config: dict[str, Any] | None = None, **kwargs) -> ClaudeClient:
    """Create a Claude client with the given configuration."""
    if config:
        # Convert config dict to ClaudeClientConfig
        claude_config = ClaudeClientConfig(
            model=config.get("model", "claude-3-5-sonnet-20241022"),
            temperature=config.get("temperature", 1.0),
            top_p=config.get("top_p", 0.95),
            max_output_tokens=config.get("max_output_tokens", 8192),
            timeout=config.get("timeout", 60000)
        )
    else:
        claude_config = None

    return create_claude_client(claude_config, **kwargs)