"""
Test suite for client factory functionality.

Following TDD principles - these tests validate the factory function
that allows switching between Gemini and Claude clients.
"""

import pytest
import os
from unittest.mock import patch

# Import the modules we're testing
from nook.functions.common.python.client_factory import create_client
from nook.functions.common.python.gemini_client import GeminiClient
from nook.functions.common.python.claude_client import ClaudeClient


class TestClientFactory:
    """Test suite for client factory function."""

    def test_factory_creates_gemini_client_by_default(self):
        """Test that factory creates Gemini client when AI_CLIENT_TYPE is not set."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}, clear=True):
            with patch('nook.functions.common.python.gemini_client.genai.Client'):
                client = create_client()

                assert isinstance(client, GeminiClient)

    def test_factory_creates_gemini_client_explicitly(self):
        """Test that factory creates Gemini client when AI_CLIENT_TYPE=gemini."""
        with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'gemini', 'GEMINI_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.gemini_client.genai.Client'):
                client = create_client()

                assert isinstance(client, GeminiClient)

    def test_factory_creates_claude_client(self):
        """Test that factory creates Claude client when AI_CLIENT_TYPE=claude."""
        with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'claude', 'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                client = create_client()

                assert isinstance(client, ClaudeClient)

    def test_factory_raises_error_for_invalid_client_type(self):
        """Test that factory raises error for unsupported client type."""
        with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'invalid'}):
            with pytest.raises(ValueError, match="Unsupported AI_CLIENT_TYPE: invalid"):
                create_client()

    def test_factory_passes_config_to_gemini_client(self):
        """Test that factory correctly passes configuration to Gemini client."""
        config = {
            'temperature': 0.5,
            'max_output_tokens': 2048
        }

        with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'gemini', 'GEMINI_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.gemini_client.genai.Client'):
                client = create_client(config)

                assert isinstance(client, GeminiClient)
                assert client._config.temperature == 0.5
                assert client._config.max_output_tokens == 2048

    def test_factory_passes_config_to_claude_client(self):
        """Test that factory correctly passes configuration to Claude client."""
        config = {
            'temperature': 0.7,
            'max_output_tokens': 4096
        }

        with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'claude', 'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                client = create_client(config)

                assert isinstance(client, ClaudeClient)
                assert client.config.temperature == 0.7
                assert client.config.max_output_tokens == 4096

    def test_factory_passes_kwargs_to_gemini_client(self):
        """Test that factory correctly passes kwargs to Gemini client."""
        with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'gemini', 'GEMINI_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.gemini_client.genai.Client'):
                client = create_client(temperature=0.3, top_p=0.8)

                assert isinstance(client, GeminiClient)
                assert client._config.temperature == 0.3
                assert client._config.top_p == 0.8

    def test_factory_passes_kwargs_to_claude_client(self):
        """Test that factory correctly passes kwargs to Claude client."""
        with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'claude', 'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                client = create_client(temperature=0.2, max_output_tokens=1024)

                assert isinstance(client, ClaudeClient)
                assert client.config.temperature == 0.2
                assert client.config.max_output_tokens == 1024

    def test_factory_case_insensitive_client_type(self):
        """Test that factory handles case-insensitive client type values."""
        with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'CLAUDE', 'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                client = create_client()

                assert isinstance(client, ClaudeClient)

        with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'GEMINI', 'GEMINI_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.gemini_client.genai.Client'):
                client = create_client()

                assert isinstance(client, GeminiClient)


class TestFactoryIntegration:
    """Test suite for factory integration scenarios."""

    def test_factory_enables_environment_switching(self):
        """Test that factory enables switching between providers via environment."""
        # Start with Gemini
        with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'gemini', 'GEMINI_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.gemini_client.genai.Client'):
                gemini_client = create_client()
                assert isinstance(gemini_client, GeminiClient)

        # Switch to Claude
        with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'claude', 'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                claude_client = create_client()
                assert isinstance(claude_client, ClaudeClient)

    def test_factory_maintains_interface_compatibility(self):
        """Test that both client types maintain compatible interfaces."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key', 'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.gemini_client.genai.Client'):
                with patch('nook.functions.common.python.claude_client.Anthropic'):
                    # Test Gemini client
                    with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'gemini'}):
                        gemini_client = create_client()

                    # Test Claude client
                    with patch.dict(os.environ, {'AI_CLIENT_TYPE': 'claude'}):
                        claude_client = create_client()

                    # Both should have the same key methods
                    assert hasattr(gemini_client, 'generate_content')
                    assert hasattr(claude_client, 'generate_content')
                    assert callable(gemini_client.generate_content)
                    assert callable(claude_client.generate_content)