"""
Claude API Client for Lambda functions.

This module provides a common interface for interacting with the Claude API,
maintaining compatibility with the existing Gemini client interface.
"""

import os
from dataclasses import dataclass
from typing import Any, List, Dict
import logging

from anthropic import Anthropic
from anthropic.types import Message
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from anthropic import APIError, RateLimitError, APITimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClaudeClientConfig:
    """Configuration for the Claude client."""

    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 1.0
    top_p: float = 0.95
    max_output_tokens: int = 8192
    timeout: int = 60000

    def update(self, **kwargs) -> None:
        """Update the configuration with the given keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")


class ClaudeClient:
    """Client for interacting with the Claude API."""

    def __init__(self, config: ClaudeClientConfig | None = None, **kwargs):
        """
        Initialize the Claude client.

        Parameters
        ----------
        config : ClaudeClientConfig | None
            Configuration for the Claude client.
            If not provided, default values will be used.
        """
        self._api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        self._config = config or ClaudeClientConfig()
        self._config.update(**kwargs)

        self._client = Anthropic(
            api_key=self._api_key,
            timeout=self._config.timeout / 1000  # Convert to seconds
        )

        # Chat session state
        self._chat_context = None
        self._system_instruction = None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(lambda e: isinstance(e, (RateLimitError, APITimeoutError))),
        before_sleep=lambda retry_state: logger.info(f"Retrying due to {retry_state.outcome.exception()}...")
    )
    def generate_content(
        self,
        contents: str | list[str],
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """
        Generate content using the Claude API.

        Parameters
        ----------
        contents : str | list[str]
            The content to generate from.
        system_instruction : str | None
            The system instruction to use.
        model : str | None
            The model to use. If not provided, the model from the config will be used.
        temperature : float | None
            The temperature to use. If not provided, the temperature from the config will be used.
        top_p : float | None
            The top_p to use. If not provided, the top_p from the config will be used.
        max_output_tokens : int | None
            The max_output_tokens to use. If not provided, the max_output_tokens from the config will be used.

        Returns
        -------
        str
            The generated content.
        """
        if isinstance(contents, str):
            messages = [{"role": "user", "content": contents}]
        else:
            messages = [{"role": "user", "content": "\n".join(contents)}]

        # Build API parameters
        api_params = {
            "model": model or self._config.model,
            "messages": messages,
            "temperature": temperature or self._config.temperature,
            "top_p": top_p or self._config.top_p,
            "max_tokens": max_output_tokens or self._config.max_output_tokens,
        }

        if system_instruction:
            api_params["system"] = system_instruction

        try:
            response = self._client.messages.create(**api_params)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating content with Claude: {e}")
            raise

    def create_chat(
        self,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> None:
        """
        Create a chat session.

        Parameters
        ----------
        model : str | None
            The model to use for the chat session.
        temperature : float | None
            The temperature to use for the chat session.
        top_p : float | None
            The top_p to use for the chat session.
        max_output_tokens : int | None
            The max_output_tokens to use for the chat session.
        system_instruction : str | None
            The system instruction for the chat session.
        """
        self._chat_context = []
        self._system_instruction = system_instruction

        # Store chat-specific config overrides
        self._chat_config = {
            "model": model or self._config.model,
            "temperature": temperature or self._config.temperature,
            "top_p": top_p or self._config.top_p,
            "max_tokens": max_output_tokens or self._config.max_output_tokens,
        }

    def send_message(self, message: str) -> str:
        """
        Send a message in the chat session.

        Parameters
        ----------
        message : str
            The message to send.

        Returns
        -------
        str
            The response from Claude.
        """
        if self._chat_context is None:
            raise ValueError("No chat session created. Call create_chat() first.")

        # Add user message to context
        self._chat_context.append({"role": "user", "content": message})

        # Build API parameters with chat context
        api_params = {
            **self._chat_config,
            "messages": self._chat_context.copy(),
        }

        if self._system_instruction:
            api_params["system"] = self._system_instruction

        try:
            response = self._client.messages.create(**api_params)
            assistant_message = response.content[0].text

            # Add assistant response to context
            self._chat_context.append({"role": "assistant", "content": assistant_message})

            return assistant_message
        except Exception as e:
            logger.error(f"Error sending message in chat: {e}")
            raise

    @property
    def config(self) -> ClaudeClientConfig:
        """Get the current client configuration."""
        return self._config


def create_client(config: ClaudeClientConfig | None = None, **kwargs) -> ClaudeClient:
    """
    Factory function to create a Claude client.

    This maintains compatibility with the Gemini client factory pattern.
    """
    return ClaudeClient(config=config, **kwargs)