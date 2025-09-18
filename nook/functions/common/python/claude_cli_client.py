"""
Claude CLI client for integrating with Claude command-line interface.

This module provides a Python interface to the Claude CLI,
allowing seamless integration with the existing AI client architecture.
"""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from tenacity import retry, stop_after_attempt, wait_exponential
import threading


@dataclass
class ClaudeCLIConfig:
    """Configuration for Claude CLI client."""
    
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 1.0
    max_tokens: int = 8192
    timeout: int = 120  # Claude CLIは応答に時間がかかるため120秒に延長
    retry_attempts: int = 3
    skip_permissions: bool = True  # 権限チェックをスキップしてパフォーマンス向上
    max_prompt_chars: Optional[int] = None  # 送信前にプロンプトをトリミングする上限
    min_request_interval_seconds: Optional[float] = None  # 連続呼び出し間隔の下限
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_prompt_chars is not None and self.max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be positive when provided")
        if self.min_request_interval_seconds is not None and self.min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds must be non-negative when provided")


class ClaudeCLIError(Exception):
    """Base exception for Claude CLI client errors."""
    pass


class ClaudeCLITimeoutError(ClaudeCLIError):
    """Raised when Claude CLI command times out."""
    pass


class ClaudeCLIProcessError(ClaudeCLIError):
    """Raised when Claude CLI process fails."""
    pass


class ClaudeCLIClient:
    """
    Claude CLI client that provides a unified interface for AI content generation.
    
    This client integrates with the Claude CLI to provide the same interface
    as other AI clients in the system, enabling seamless switching between providers.
    """
    
    def __init__(self, config: Optional[ClaudeCLIConfig] = None, **kwargs) -> None:
        """
        Initialize Claude CLI client.
        
        Parameters
        ----------
        config : ClaudeCLIConfig, optional
            Configuration for the Claude CLI client.
        **kwargs
            Additional configuration parameters.
        """
        self.config = config or ClaudeCLIConfig()
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Check if Claude CLI is available
        self._check_claude_cli_availability()
        
        # Initialize session state
        self._session_history: List[Dict[str, str]] = []

        # rate limiting
        if not hasattr(self.__class__, "_request_lock"):
            self.__class__._request_lock = threading.Lock()
            self.__class__._last_request_ts = 0.0
    
    def _check_claude_cli_availability(self) -> None:
        """Check if Claude CLI is installed and available."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise ClaudeCLIError("Claude CLI is not properly installed or configured")
        except FileNotFoundError:
            raise ClaudeCLIError(
                "Claude CLI not found. Please install it with: npm install -g @anthropic-ai/claude-cli"
            )
        except subprocess.TimeoutExpired:
            raise ClaudeCLIError("Claude CLI check timed out")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _execute_claude_command(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Execute Claude CLI command with retry logic.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to Claude.
        system_instruction : str, optional
            System instruction to prepend to the prompt.
            
        Returns
        -------
        str
            The response from Claude CLI.
            
        Raises
        ------
        ClaudeCLITimeoutError
            If the command times out.
        ClaudeCLIProcessError
            If the command fails.
        """
        # Prepare the full prompt
        prompt, system_instruction = self._enforce_prompt_limit(prompt, system_instruction)
        self._throttle_if_needed()

        full_prompt = prompt
        if system_instruction:
            full_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
        
        try:
            # Prepare Claude CLI command with optimized options
            cmd = ["claude", "-p"]

            if self.config.model:
                cmd.extend(["--model", self.config.model])
            
            # Add performance optimization options
            if self.config.skip_permissions:
                cmd.append("--dangerously-skip-permissions")
            
            # Add output format for consistent results
            cmd.extend(["--output-format", "text"])
            
            # Add the prompt
            cmd.append(full_prompt)

            # Execute Claude CLI command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                check=True
            )
            
            response = result.stdout.strip()
            if not response:
                raise ClaudeCLIProcessError("Empty response from Claude CLI")
            
            return response
            
        except subprocess.TimeoutExpired as e:
            raise ClaudeCLITimeoutError(f"Claude CLI command timed out after {self.config.timeout}s") from e
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            raise ClaudeCLIProcessError(f"Claude CLI command failed: {error_msg}") from e
    
    def generate_content(
        self, 
        contents: str, 
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate content using Claude CLI.
        
        This method provides compatibility with the existing AI client interface.
        
        Parameters
        ----------
        contents : str
            The content/prompt to send to Claude.
        system_instruction : str, optional
            System instruction for the model.
        **kwargs
            Additional parameters (for compatibility).
            
        Returns
        -------
        str
            Generated content from Claude.
        """
        try:
            response = self._execute_claude_command(contents, system_instruction)
            
            # Add to session history
            self._session_history.append({
                "role": "user",
                "content": contents,
                "system": system_instruction or ""
            })
            self._session_history.append({
                "role": "assistant", 
                "content": response
            })
            
            return response
            
        except Exception as e:
            raise ClaudeCLIError(f"Failed to generate content: {str(e)}") from e

    def _enforce_prompt_limit(
        self,
        prompt: str,
        system_instruction: Optional[str],
    ) -> tuple[str, Optional[str]]:
        """Truncate the user prompt if it exceeds the configured character limit."""

        limit = self.config.max_prompt_chars
        if not limit:
            return prompt, system_instruction

        # When system instruction is present, keep it intact and only trim user prompt.
        prefix_length = 0
        if system_instruction:
            prefix_length = len("System: \n\nUser: ") + len(system_instruction)

        available_for_prompt = max(limit - prefix_length, 0)
        if len(prompt) <= available_for_prompt or available_for_prompt == 0:
            if available_for_prompt == 0:
                truncated_notice = "[入力テキストは長すぎるため全体を送信できませんでした]"
                return truncated_notice[:limit], system_instruction
            return prompt, system_instruction

        truncated_prompt = prompt[:available_for_prompt]
        truncated_prompt = truncated_prompt.rstrip()
        truncated_prompt += "\n\n[入力テキストは CLAUDE_MAX_PROMPT_CHARS の制限により途中までで送信されています]"
        return truncated_prompt, system_instruction

    def _throttle_if_needed(self) -> None:
        interval = self.config.min_request_interval_seconds
        if not interval or interval <= 0:
            return

        lock = getattr(self.__class__, "_request_lock")
        with lock:
            last_ts = getattr(self.__class__, "_last_request_ts", 0.0)
            now = time.monotonic()
            wait = interval - (now - last_ts)
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self.__class__._last_request_ts = now
    
    def chat_with_search(self, message: str) -> str:
        """
        Chat with Claude CLI (search functionality not directly supported).
        
        Parameters
        ----------
        message : str
            The message to send to Claude.
            
        Returns
        -------
        str
            Claude's response.
            
        Note
        ----
        Claude CLI doesn't have built-in search functionality like Gemini.
        This method provides basic chat functionality for compatibility.
        """
        return self.generate_content(
            message,
            system_instruction="You are a helpful assistant. Please provide detailed and accurate responses."
        )
    
    def start_chat(self, history: Optional[List[Dict[str, str]]] = None) -> 'ClaudeCLIClient':
        """
        Start a chat session (for compatibility).
        
        Parameters
        ----------
        history : List[Dict[str, str]], optional
            Previous chat history.
            
        Returns
        -------
        ClaudeCLIClient
            Returns self for method chaining.
        """
        if history:
            self._session_history = history.copy()
        return self
    
    def send_message(self, message: str) -> str:
        """
        Send a message in the current chat session.
        
        Parameters
        ----------
        message : str
            The message to send.
            
        Returns
        -------
        str
            Claude's response.
        """
        # Include recent history for context
        context = ""
        if self._session_history:
            # Include last few exchanges for context
            recent_history = self._session_history[-4:]  # Last 2 exchanges
            for entry in recent_history:
                role = entry.get("role", "")
                content = entry.get("content", "")
                if role and content:
                    context += f"{role.title()}: {content}\n"
        
        full_message = f"{context}\nUser: {message}" if context else message
        
        return self.generate_content(full_message)
    
    def get_session_history(self) -> List[Dict[str, str]]:
        """
        Get the current session history.
        
        Returns
        -------
        List[Dict[str, str]]
            The session history.
        """
        return self._session_history.copy()
    
    def clear_session(self) -> None:
        """Clear the current session history."""
        self._session_history.clear()


def create_client(config: Optional[ClaudeCLIConfig] = None, **kwargs) -> ClaudeCLIClient:
    """
    Create a Claude CLI client instance.
    
    Parameters
    ----------
    config : ClaudeCLIConfig, optional
        Configuration for the client.
    **kwargs
        Additional configuration parameters.
        
    Returns
    -------
    ClaudeCLIClient
        Configured Claude CLI client.
        
    Examples
    --------
    >>> client = create_client()
    >>> response = client.generate_content("Hello, Claude!")
    >>> print(response)
    """
    return ClaudeCLIClient(config=config, **kwargs)


# For backward compatibility
ClaudeClient = ClaudeCLIClient
ClaudeClientConfig = ClaudeCLIConfig
