"""
Tests for Claude CLI client.

This module contains comprehensive tests for the Claude CLI client,
including unit tests, integration tests, and error handling tests.
"""

import os
import subprocess
import tempfile
from unittest.mock import Mock, patch, MagicMock
import pytest

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from claude_cli_client import (
    ClaudeCLIClient,
    ClaudeCLIConfig,
    ClaudeCLIError,
    ClaudeCLITimeoutError,
    ClaudeCLIProcessError,
    create_client
)


class TestClaudeCLIConfig:
    """Test cases for ClaudeCLIConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ClaudeCLIConfig()
        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.temperature == 1.0
        assert config.max_tokens == 8192
        assert config.timeout == 60
        assert config.retry_attempts == 3
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ClaudeCLIConfig(
            model="claude-3-opus-20240229",
            temperature=0.5,
            max_tokens=4096,
            timeout=30,
            retry_attempts=5
        )
        assert config.model == "claude-3-opus-20240229"
        assert config.temperature == 0.5
        assert config.max_tokens == 4096
        assert config.timeout == 30
        assert config.retry_attempts == 5
    
    def test_invalid_temperature(self):
        """Test validation of temperature parameter."""
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
            ClaudeCLIConfig(temperature=-0.1)
        
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
            ClaudeCLIConfig(temperature=2.1)
    
    def test_invalid_max_tokens(self):
        """Test validation of max_tokens parameter."""
        with pytest.raises(ValueError, match="Max tokens must be positive"):
            ClaudeCLIConfig(max_tokens=0)
        
        with pytest.raises(ValueError, match="Max tokens must be positive"):
            ClaudeCLIConfig(max_tokens=-1)
    
    def test_invalid_timeout(self):
        """Test validation of timeout parameter."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            ClaudeCLIConfig(timeout=0)
        
        with pytest.raises(ValueError, match="Timeout must be positive"):
            ClaudeCLIConfig(timeout=-1)


class TestClaudeCLIClient:
    """Test cases for ClaudeCLIClient."""
    
    @patch('subprocess.run')
    def test_client_initialization_success(self, mock_run):
        """Test successful client initialization."""
        # Mock successful version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        
        client = ClaudeCLIClient()
        assert isinstance(client.config, ClaudeCLIConfig)
        assert client._session_history == []
    
    @patch('subprocess.run')
    def test_client_initialization_cli_not_found(self, mock_run):
        """Test client initialization when CLI is not found."""
        mock_run.side_effect = FileNotFoundError()
        
        with pytest.raises(ClaudeCLIError, match="Claude CLI not found"):
            ClaudeCLIClient()
    
    @patch('subprocess.run')
    def test_client_initialization_cli_error(self, mock_run):
        """Test client initialization when CLI returns error."""
        mock_run.return_value = Mock(returncode=1, stderr="Error")
        
        with pytest.raises(ClaudeCLIError, match="Claude CLI is not properly installed"):
            ClaudeCLIClient()
    
    @patch('subprocess.run')
    def test_client_initialization_timeout(self, mock_run):
        """Test client initialization timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("claude", 10)
        
        with pytest.raises(ClaudeCLIError, match="Claude CLI check timed out"):
            ClaudeCLIClient()
    
    @patch('subprocess.run')
    def test_generate_content_success(self, mock_run):
        """Test successful content generation."""
        # Mock version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        client = ClaudeCLIClient()
        
        # Mock content generation
        mock_run.return_value = Mock(
            returncode=0,
            stdout="This is Claude's response.",
            stderr=""
        )
        
        response = client.generate_content("Hello, Claude!")
        
        assert response == "This is Claude's response."
        assert len(client._session_history) == 2
        assert client._session_history[0]["role"] == "user"
        assert client._session_history[1]["role"] == "assistant"
    
    @patch('subprocess.run')
    def test_generate_content_with_system_instruction(self, mock_run):
        """Test content generation with system instruction."""
        # Mock version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        client = ClaudeCLIClient()
        
        # Mock content generation
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Response with system instruction.",
            stderr=""
        )
        
        response = client.generate_content(
            "Hello, Claude!",
            system_instruction="You are a helpful assistant."
        )
        
        assert response == "Response with system instruction."
        # Check that system instruction was included in the call
        mock_run.assert_called_with(
            ["claude", "-p", "System: You are a helpful assistant.\n\nUser: Hello, Claude!"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )
    
    @patch('subprocess.run')
    def test_generate_content_timeout(self, mock_run):
        """Test content generation timeout."""
        # Mock version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        client = ClaudeCLIClient()
        
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired("claude", 60)
        
        with pytest.raises(ClaudeCLITimeoutError, match="Claude CLI command timed out"):
            client.generate_content("Hello, Claude!")
    
    @patch('subprocess.run')
    def test_generate_content_process_error(self, mock_run):
        """Test content generation process error."""
        # Mock version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        client = ClaudeCLIClient()
        
        # Mock process error
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "claude", stderr="Command failed"
        )
        
        with pytest.raises(ClaudeCLIProcessError, match="Claude CLI command failed"):
            client.generate_content("Hello, Claude!")
    
    @patch('subprocess.run')
    def test_generate_content_empty_response(self, mock_run):
        """Test handling of empty response."""
        # Mock version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        client = ClaudeCLIClient()
        
        # Mock empty response
        mock_run.return_value = Mock(
            returncode=0,
            stdout="",
            stderr=""
        )
        
        with pytest.raises(ClaudeCLIProcessError, match="Empty response from Claude CLI"):
            client.generate_content("Hello, Claude!")
    
    @patch('subprocess.run')
    def test_chat_with_search(self, mock_run):
        """Test chat with search functionality."""
        # Mock version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        client = ClaudeCLIClient()
        
        # Mock chat response
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Chat response from Claude.",
            stderr=""
        )
        
        response = client.chat_with_search("What is AI?")
        
        assert response == "Chat response from Claude."
        # Verify system instruction was added
        expected_prompt = "System: You are a helpful assistant. Please provide detailed and accurate responses.\n\nUser: What is AI?"
        mock_run.assert_called_with(
            ["claude", "-p", expected_prompt],
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )
    
    @patch('subprocess.run')
    def test_start_chat(self, mock_run):
        """Test starting a chat session."""
        # Mock version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        client = ClaudeCLIClient()
        
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        chat_client = client.start_chat(history)
        
        assert chat_client is client
        assert client._session_history == history
    
    @patch('subprocess.run')
    def test_send_message_with_context(self, mock_run):
        """Test sending message with chat context."""
        # Mock version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        client = ClaudeCLIClient()
        
        # Set up chat history
        client._session_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        # Mock response
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Response with context.",
            stderr=""
        )
        
        response = client.send_message("How are you?")
        
        assert response == "Response with context."
        # Verify context was included
        expected_context = "User: Hello\nAssistant: Hi there!\nUser: How are you?"
        mock_run.assert_called_with(
            ["claude", "-p", expected_context],
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )
    
    @patch('subprocess.run')
    def test_session_management(self, mock_run):
        """Test session history management."""
        # Mock version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        client = ClaudeCLIClient()
        
        # Test empty history
        assert client.get_session_history() == []
        
        # Add some history
        client._session_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]
        
        history = client.get_session_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        
        # Test clear session
        client.clear_session()
        assert client.get_session_history() == []
    
    @patch('subprocess.run')
    def test_config_override_with_kwargs(self, mock_run):
        """Test configuration override with kwargs."""
        # Mock version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        
        client = ClaudeCLIClient(temperature=0.5, max_tokens=4096)
        
        assert client.config.temperature == 0.5
        assert client.config.max_tokens == 4096


class TestCreateClient:
    """Test cases for create_client function."""
    
    @patch('subprocess.run')
    def test_create_client_default(self, mock_run):
        """Test creating client with default configuration."""
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        
        client = create_client()
        
        assert isinstance(client, ClaudeCLIClient)
        assert isinstance(client.config, ClaudeCLIConfig)
    
    @patch('subprocess.run')
    def test_create_client_with_config(self, mock_run):
        """Test creating client with custom configuration."""
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        
        config = ClaudeCLIConfig(temperature=0.7, max_tokens=2048)
        client = create_client(config=config)
        
        assert isinstance(client, ClaudeCLIClient)
        assert client.config.temperature == 0.7
        assert client.config.max_tokens == 2048
    
    @patch('subprocess.run')
    def test_create_client_with_kwargs(self, mock_run):
        """Test creating client with kwargs."""
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        
        client = create_client(temperature=0.3, timeout=120)
        
        assert isinstance(client, ClaudeCLIClient)
        assert client.config.temperature == 0.3
        assert client.config.timeout == 120


class TestErrorHandling:
    """Test cases for error handling."""
    
    @patch('subprocess.run')
    def test_retry_mechanism(self, mock_run):
        """Test retry mechanism for transient failures."""
        # Mock version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        client = ClaudeCLIClient()
        
        # Mock transient failure then success
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "claude", stderr="Temporary error"),
            subprocess.CalledProcessError(1, "claude", stderr="Temporary error"),
            Mock(returncode=0, stdout="Success after retry", stderr="")
        ]
        
        response = client.generate_content("Test retry")
        
        assert response == "Success after retry"
        assert mock_run.call_count == 4  # 1 for version check + 3 for retries
    
    @patch('subprocess.run')
    def test_retry_exhaustion(self, mock_run):
        """Test behavior when retries are exhausted."""
        # Mock version check
        mock_run.return_value = Mock(returncode=0, stdout="claude 1.0.0")
        client = ClaudeCLIClient()
        
        # Mock persistent failure
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "claude", stderr="Persistent error"
        )
        
        with pytest.raises(ClaudeCLIError):
            client.generate_content("Test persistent failure")


@pytest.mark.integration
class TestClaudeCLIIntegration:
    """Integration tests for Claude CLI client."""
    
    def test_claude_cli_availability(self):
        """Test if Claude CLI is available for integration tests."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                pytest.skip("Claude CLI not available for integration tests")
        except FileNotFoundError:
            pytest.skip("Claude CLI not installed")
        except subprocess.TimeoutExpired:
            pytest.skip("Claude CLI check timed out")
    
    @pytest.mark.skipif(
        not os.environ.get("CLAUDE_INTEGRATION_TEST"),
        reason="Set CLAUDE_INTEGRATION_TEST=1 to run integration tests"
    )
    def test_real_claude_cli_interaction(self):
        """Test real interaction with Claude CLI (requires CLI to be installed)."""
        try:
            client = ClaudeCLIClient()
            response = client.generate_content("Say 'Hello, World!' in exactly those words.")
            
            assert isinstance(response, str)
            assert len(response) > 0
            # Note: We can't assert exact content as Claude's responses may vary
            
        except ClaudeCLIError as e:
            pytest.skip(f"Claude CLI not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
