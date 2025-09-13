"""
Basic integration tests for Claude client functionality.

These tests validate core integration scenarios without depending
on complex external dependencies.
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from nook.functions.common.python.claude_client import ClaudeClient, ClaudeClientConfig


@pytest.mark.integration
class TestBasicClaudeIntegration:
    """Basic integration tests for Claude client."""

    @pytest.fixture
    def mock_claude_responses(self):
        """Mock Claude API responses for testing."""
        return {
            "simple_response": "This is a simple response from Claude.",
            "paper_summary": """# Machine Learning Research Summary

## Key Findings
- Novel neural network architecture
- 15% improvement in accuracy
- Reduced computational requirements

## Technical Details
The researchers implemented a transformer-based model with attention mechanisms.

## Implications
This work could significantly impact natural language processing applications.""",
            "tech_analysis": """## Tech News Analysis

**Main Points:**
1. New AI breakthrough announced
2. 50% performance improvement claimed
3. Expected commercial release in 2024

**Assessment:**
The technology shows promise but requires further validation."""
        }

    def test_claude_content_generation_integration(self, mock_claude_responses):
        """Test basic content generation with Claude client."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic') as mock_anthropic:
                # Setup mock client
                mock_client = Mock()
                mock_anthropic.return_value = mock_client

                # Setup response
                mock_response = Mock()
                mock_response.content = [Mock(text=mock_claude_responses["simple_response"])]
                mock_client.messages.create.return_value = mock_response

                # Test client
                client = ClaudeClient()
                result = client.generate_content("Test prompt")

                # Verify result
                assert result == mock_claude_responses["simple_response"]
                assert mock_client.messages.create.called

    def test_claude_paper_summarization_format(self, mock_claude_responses):
        """Test that Claude can generate paper summaries in expected format."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic') as mock_anthropic:
                mock_client = Mock()
                mock_anthropic.return_value = mock_client

                mock_response = Mock()
                mock_response.content = [Mock(text=mock_claude_responses["paper_summary"])]
                mock_client.messages.create.return_value = mock_response

                client = ClaudeClient()
                result = client.generate_content(
                    "Summarize this research paper",
                    system_instruction="You are a research assistant"
                )

                # Verify format
                assert "# " in result  # Has title
                assert "## Key Findings" in result  # Has sections
                assert "## Technical Details" in result
                assert "## Implications" in result

                # Verify API call
                call_args = mock_client.messages.create.call_args[1]
                assert call_args["system"] == "You are a research assistant"

    def test_claude_tech_news_analysis_format(self, mock_claude_responses):
        """Test that Claude can analyze tech news in expected format."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic') as mock_anthropic:
                mock_client = Mock()
                mock_anthropic.return_value = mock_client

                mock_response = Mock()
                mock_response.content = [Mock(text=mock_claude_responses["tech_analysis"])]
                mock_client.messages.create.return_value = mock_response

                client = ClaudeClient()
                result = client.generate_content("Analyze this tech news article")

                # Verify format
                assert "## Tech News Analysis" in result
                assert "**Main Points:**" in result
                assert "**Assessment:**" in result

    def test_claude_chat_session_integration(self):
        """Test Claude chat session functionality."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic') as mock_anthropic:
                mock_client = Mock()
                mock_anthropic.return_value = mock_client

                # Setup multiple responses for conversation
                responses = [
                    Mock(content=[Mock(text="Hello! How can I help you?")]),
                    Mock(content=[Mock(text="I can help summarize research papers.")]),
                    Mock(content=[Mock(text="Yes, I can analyze multiple papers at once.")])
                ]
                mock_client.messages.create.side_effect = responses

                client = ClaudeClient()
                client.create_chat(system_instruction="You are a helpful research assistant")

                # Send multiple messages
                response1 = client.send_message("Hello")
                response2 = client.send_message("Can you help with research papers?")
                response3 = client.send_message("Can you handle multiple papers?")

                # Verify responses
                assert response1 == "Hello! How can I help you?"
                assert response2 == "I can help summarize research papers."
                assert response3 == "Yes, I can analyze multiple papers at once."

                # Verify conversation context was maintained
                assert len(client._chat_context) == 6  # 3 user + 3 assistant messages
                assert client._chat_context[0]["role"] == "user"
                assert client._chat_context[1]["role"] == "assistant"

    def test_claude_error_handling_integration(self):
        """Test Claude client error handling in integration scenarios."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic') as mock_anthropic:
                mock_client = Mock()
                mock_anthropic.return_value = mock_client

                # Simulate generic exception (more realistic for testing)
                mock_client.messages.create.side_effect = Exception("API Connection Error")

                client = ClaudeClient()

                # Should raise the error
                with pytest.raises(Exception, match="API Connection Error"):
                    client.generate_content("Test prompt")

    def test_claude_configuration_integration(self):
        """Test Claude client with different configurations."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic') as mock_anthropic:
                mock_client = Mock()
                mock_anthropic.return_value = mock_client

                mock_response = Mock()
                mock_response.content = [Mock(text="Configured response")]
                mock_client.messages.create.return_value = mock_response

                # Test with custom configuration
                config = ClaudeClientConfig(
                    model="claude-3-5-sonnet-20241022",
                    temperature=0.3,
                    max_output_tokens=2048
                )

                client = ClaudeClient(config)
                result = client.generate_content("Test with custom config")

                # Verify configuration was used
                call_args = mock_client.messages.create.call_args[1]
                assert call_args["model"] == "claude-3-5-sonnet-20241022"
                assert call_args["temperature"] == 0.3
                assert call_args["max_tokens"] == 2048

    def test_environment_switching_integration(self):
        """Test switching between different environments."""
        # Test Claude environment
        with patch.dict(os.environ, {
            'AI_PROVIDER': 'claude',
            'ANTHROPIC_API_KEY': 'claude-key'
        }):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                from nook.functions.common.python.claude_client import create_client
                client = create_client()

                assert isinstance(client, ClaudeClient)
                assert client.config.model == "claude-3-5-sonnet-20241022"

        # Test fallback behavior with missing provider
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                from nook.functions.common.python.claude_client import create_client
                client = create_client()

                assert isinstance(client, ClaudeClient)


@pytest.mark.integration
class TestClaudeFunctionalParity:
    """Test functional parity between Claude and expected behavior."""

    def test_response_quality_structure(self):
        """Test that Claude responses have expected structure."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic') as mock_anthropic:
                mock_client = Mock()
                mock_anthropic.return_value = mock_client

                # Mock structured response
                structured_response = """# Research Analysis

## Summary
This paper introduces a novel approach to machine learning.

## Methodology
The authors used transformer architecture with attention mechanisms.

## Results
- 15% improvement in accuracy
- 30% reduction in training time
- Better generalization on unseen data

## Limitations
- Limited to text-based tasks
- Requires large computational resources

## Conclusion
The approach shows significant promise for practical applications."""

                mock_response = Mock()
                mock_response.content = [Mock(text=structured_response)]
                mock_client.messages.create.return_value = mock_response

                client = ClaudeClient()
                result = client.generate_content("Analyze this research paper")

                # Verify structure
                assert result.count('#') >= 5  # Multiple sections
                assert '## Summary' in result
                assert '## Methodology' in result
                assert '## Results' in result
                assert '## Limitations' in result
                assert '## Conclusion' in result

    def test_response_length_consistency(self):
        """Test that Claude responses have appropriate length."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic') as mock_anthropic:
                mock_client = Mock()
                mock_anthropic.return_value = mock_client

                # Mock responses of different lengths
                responses = [
                    "Short response.",
                    "This is a medium length response that provides some detail about the topic at hand.",
                    """This is a comprehensive response that covers multiple aspects of the topic.

It includes detailed analysis, examples, and structured information that would be
typical for academic or professional content generation. The response maintains
coherence throughout while providing valuable insights and actionable information."""
                ]

                for response_text in responses:
                    mock_response = Mock()
                    mock_response.content = [Mock(text=response_text)]
                    mock_client.messages.create.return_value = mock_response

                    client = ClaudeClient()
                    result = client.generate_content("Generate content")

                    # Basic quality checks
                    assert len(result) > 0
                    assert isinstance(result, str)
                    # Should have reasonable word count
                    word_count = len(result.split())
                    assert word_count > 0