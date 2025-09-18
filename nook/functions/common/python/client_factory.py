"""
Client factory for switching between Gemini and Claude clients.

This module provides a unified interface for creating AI clients,
allowing seamless switching between different providers.
"""

import os
from typing import Any, Union
from .gemini_client import GeminiClient, GeminiClientConfig, create_client as create_gemini_client
from .claude_cli_client import ClaudeCLIClient, ClaudeCLIConfig, create_client as create_claude_cli_client


def create_client(config: dict[str, Any] | None = None, **kwargs) -> Union[GeminiClient, ClaudeCLIClient]:
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
    Union[GeminiClient, ClaudeCLIClient]
        The configured AI client.

    Examples
    --------
    # Use default client (Gemini)
    client = create_client()

    # Force Claude CLI client
    os.environ['AI_CLIENT_TYPE'] = 'claude'
    client = create_client()

    # Use Gemini with custom config
    os.environ['AI_CLIENT_TYPE'] = 'gemini'
    client = create_client({'temperature': 0.5})
    """
    client_type = os.environ.get("AI_CLIENT_TYPE", "gemini").lower()

    if client_type == "claude":
        return _create_claude_cli_client(config, **kwargs)
    elif client_type == "gemini":
        return _create_gemini_client(config, **kwargs)
    else:
        raise ValueError(f"Unsupported AI_CLIENT_TYPE: {client_type}. Supported values: 'gemini', 'claude'")


def _create_gemini_client(config: dict[str, Any] | None = None, **kwargs) -> GeminiClient:
    """Create a Gemini client with the given configuration."""
    return create_gemini_client(config, **kwargs)


def _create_claude_cli_client(config: dict[str, Any] | None = None, **kwargs) -> ClaudeCLIClient:
    """Create a Claude CLI client with the given configuration."""
    params: dict[str, Any] = {
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 1.0,
        "max_tokens": 8192,
        "timeout": 120,
        "max_prompt_chars": None,
        "min_request_interval_seconds": None,
    }

    if config:
        params.update({k: v for k, v in config.items() if k in params})

    env_overrides = {
        "model": os.environ.get("CLAUDE_MODEL"),
        "temperature": os.environ.get("CLAUDE_TEMPERATURE"),
        "max_tokens": os.environ.get("CLAUDE_MAX_OUTPUT_TOKENS"),
        "timeout": os.environ.get("CLAUDE_TIMEOUT_SECONDS"),
        "max_prompt_chars": os.environ.get("CLAUDE_MAX_PROMPT_CHARS"),
        "min_request_interval_seconds": os.environ.get("CLAUDE_MIN_REQUEST_INTERVAL_SECONDS"),
    }

    for key, value in env_overrides.items():
        if value is None:
            continue
        try:
            if key in {"temperature"}:
                params[key] = float(value)
            elif key in {"max_tokens", "timeout", "max_prompt_chars"}:
                params[key] = int(value)
            elif key in {"min_request_interval_seconds"}:
                params[key] = float(value)
            else:
                params[key] = value
        except ValueError:
            continue

    claude_config = ClaudeCLIConfig(**params)

    return create_claude_cli_client(claude_config, **kwargs)
