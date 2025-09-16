"""
Test suite for Claude client implementation.

Following TDD principles - these tests define the expected behavior
before implementation details are finalized.
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the modules we're testing
from nook.functions.common.python.claude_client import (
    ClaudeClient,
    ClaudeClientConfig,
    create_client
)


class TestClaudeClientConfig:
    """Test suite for Claude client configuration."""

    def test_default_configuration_values(self):
        """Test that default configuration values are set correctly."""
        config = ClaudeClientConfig()

        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.temperature == 1.0
        assert config.top_p == 0.95
        assert config.max_output_tokens == 8192
        assert config.timeout == 60000

    def test_configuration_update_with_valid_keys(self):
        """Test updating configuration with valid keys."""
        config = ClaudeClientConfig()

        config.update(temperature=0.5, max_output_tokens=4096)

        assert config.temperature == 0.5
        assert config.max_output_tokens == 4096
        # Other values should remain unchanged
        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.top_p == 0.95

    def test_configuration_update_with_invalid_key_raises_error(self):
        """Test that updating with invalid key raises ValueError."""
        config = ClaudeClientConfig()

        with pytest.raises(ValueError, match="Invalid configuration key: invalid_key"):
            config.update(invalid_key="invalid_value")


class TestClaudeClient:
    """Test suite for Claude client implementation."""

    @pytest.fixture
    def mock_responses(self):
        """Load mock API responses from fixtures."""
        fixtures_path = Path(__file__).parent / "fixtures" / "mock_responses.json"
        with open(fixtures_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client for testing."""
        with patch('nook.functions.common.python.claude_client.Anthropic') as mock:
            mock_client = Mock()
            mock.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def claude_config(self):
        """Standard Claude configuration for testing."""
        return ClaudeClientConfig(
            model="claude-3-5-sonnet-20241022",
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=8192,
            timeout=60000
        )

    @pytest.fixture
    def claude_client(self, mock_anthropic_client, claude_config):
        """Claude client instance for testing."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            return ClaudeClient(claude_config)

    def test_client_initialization_success(self, claude_config):
        """Test successful client initialization."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'valid-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic') as mock_anthropic:
                client = ClaudeClient(claude_config)

                assert client is not None
                assert client.config.model == "claude-3-5-sonnet-20241022"
                mock_anthropic.assert_called_once()

    def test_client_initialization_missing_api_key(self, claude_config):
        """Test client initialization fails with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is not set"):
                ClaudeClient(claude_config)

    def test_client_initialization_with_config_overrides(self):
        """Test client initialization with configuration overrides."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                client = ClaudeClient(temperature=0.5, max_output_tokens=4096)

                assert client.config.temperature == 0.5
                assert client.config.max_output_tokens == 4096

    def test_generate_content_success_string_input(self, claude_client, mock_anthropic_client, mock_responses):
        """Test successful content generation with string input."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Generated response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = claude_client.generate_content("Test content")

        assert result == "Generated response"
        mock_anthropic_client.messages.create.assert_called_once()

        # Verify the call parameters
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-3-5-sonnet-20241022"
        assert call_args["messages"] == [{"role": "user", "content": "Test content"}]
        assert call_args["temperature"] == 1.0

    def test_generate_content_success_list_input(self, claude_client, mock_anthropic_client):
        """Test successful content generation with list input."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Multi-content response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = claude_client.generate_content(["Content 1", "Content 2"])

        assert result == "Multi-content response"
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert call_args["messages"] == [{"role": "user", "content": "Content 1\nContent 2"}]

    def test_generate_content_with_system_instruction(self, claude_client, mock_anthropic_client):
        """Test content generation with system instruction."""
        mock_response = Mock()
        mock_response.content = [Mock(text="System guided response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = claude_client.generate_content(
            "Test content",
            system_instruction="You are a helpful assistant"
        )

        assert result == "System guided response"
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "system" in call_args
        assert call_args["system"] == "You are a helpful assistant"

    def test_generate_content_with_parameter_overrides(self, claude_client, mock_anthropic_client):
        """Test content generation with parameter overrides."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Override response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = claude_client.generate_content(
            "Test content",
            temperature=0.3,
            max_output_tokens=2048
        )

        assert result == "Override response"
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert call_args["temperature"] == 0.3
        assert call_args["max_tokens"] == 2048

    def test_create_chat_session(self, claude_client):
        """Test chat session creation."""
        claude_client.create_chat()

        assert claude_client._chat_context == []
        assert claude_client._system_instruction is None

    def test_create_chat_with_parameters(self, claude_client):
        """Test chat session creation with parameters."""
        claude_client.create_chat(
            temperature=0.7,
            system_instruction="You are a helpful assistant"
        )

        assert claude_client._chat_context == []
        assert claude_client._system_instruction == "You are a helpful assistant"

    def test_send_message_success(self, claude_client, mock_anthropic_client):
        """Test successful message sending in chat."""
        # Setup chat session
        claude_client.create_chat()

        # Setup mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Chat response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = claude_client.send_message("Hello")

        assert result == "Chat response"
        assert len(claude_client._chat_context) == 2  # User message + assistant response
        assert claude_client._chat_context[0] == {"role": "user", "content": "Hello"}
        assert claude_client._chat_context[1] == {"role": "assistant", "content": "Chat response"}

    def test_send_message_no_chat_session(self, claude_client):
        """Test sending message without chat session fails."""
        with pytest.raises(ValueError, match="No chat session created"):
            claude_client.send_message("Hello")

    def test_send_message_with_system_instruction(self, claude_client, mock_anthropic_client):
        """Test sending message in chat with system instruction."""
        claude_client.create_chat(system_instruction="You are a helpful assistant")

        mock_response = Mock()
        mock_response.content = [Mock(text="System guided chat response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = claude_client.send_message("Hello")

        assert result == "System guided chat response"
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert call_args["system"] == "You are a helpful assistant"

    @patch('nook.functions.common.python.claude_client.logger')
    def test_generate_content_api_error_logging(self, mock_logger, claude_client, mock_anthropic_client):
        """Test that API errors are properly logged."""
        # Setup to raise an exception
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            claude_client.generate_content("Test content")

        # Verify error logging
        mock_logger.error.assert_called_once()
        assert "Error generating content with Claude" in str(mock_logger.error.call_args)

    def test_config_property_access(self, claude_client):
        """Test accessing client configuration via property."""
        config = claude_client.config

        assert isinstance(config, ClaudeClientConfig)
        assert config.model == "claude-3-5-sonnet-20241022"


class TestClientFactory:
    """Test suite for client factory function."""

    def test_create_client_without_config(self):
        """Test creating client without explicit config."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                client = create_client()

                assert isinstance(client, ClaudeClient)
                assert client.config.model == "claude-3-5-sonnet-20241022"

    def test_create_client_with_config(self):
        """Test creating client with explicit config."""
        config = ClaudeClientConfig(temperature=0.5)

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                client = create_client(config=config)

                assert isinstance(client, ClaudeClient)
                assert client.config.temperature == 0.5

    def test_create_client_with_kwargs(self):
        """Test creating client with keyword arguments."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                client = create_client(temperature=0.3, max_output_tokens=2048)

                assert isinstance(client, ClaudeClient)
                assert client.config.temperature == 0.3
                assert client.config.max_output_tokens == 2048


# Test that would use actual retry mechanism (commented out for now as it takes time)
# class TestRetryMechanism:
#     """Test retry mechanism on API errors."""
#
#     def test_retry_on_rate_limit_error(self, claude_client, mock_anthropic_client):
#         """Test retry mechanism on rate limit errors."""
#         from anthropic import RateLimitError
#
#         # Setup to fail once, then succeed
#         mock_anthropic_client.messages.create.side_effect = [
#             RateLimitError("Rate limited"),
#             Mock(content=[Mock(text="Success after retry")])
#         ]
#
#         result = claude_client.generate_content("Test content")
#
#         assert result == "Success after retry"
#         assert mock_anthropic_client.messages.create.call_count == 2