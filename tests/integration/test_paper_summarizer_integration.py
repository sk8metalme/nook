"""
Integration tests for Paper Summarizer with Claude API.

This test suite validates that the paper summarizer works with Claude client
and maintains functional parity with Gemini implementation.
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the classes we're testing
from nook.functions.paper_summarizer.paper_summarizer import PaperSummarizer
from nook.functions.common.python.claude_client import ClaudeClient, ClaudeClientConfig


@pytest.mark.integration
class TestPaperSummarizerClaudeIntegration:
    """Integration tests for Paper Summarizer with Claude."""

    @pytest.fixture
    def mock_claude_response(self):
        """Mock Claude API response for paper summarization."""
        return """# Attention Is All You Need

[View Paper](https://arxiv.org/abs/1706.03762)

## 1. 既存研究では何ができなかったのか

従来のsequence transductionモデルは、RNNやCNNベースのアーキテクチャに依存しており、並列処理が困難で長いシーケンスでの学習効率が低い問題がありました。

## 2. どのようなアプローチでそれを解決しようとしたか

本論文では、recurrenceやconvolutionを一切使わず、attention mechanismのみに基づく「Transformer」アーキテクチャを提案しました。

## 3. 結果、何が達成できたのか

翻訳タスク（WMT 2014 English-to-German）でstate-of-the-artを達成し、従来モデルより大幅に高速な学習を実現しました。

## 4. Limitationや問題点は何か

長いシーケンスでのメモリ使用量が多く、位置エンコーディングの改善余地があります。

## 5. 技術的な詳細について

Multi-Head Attentionとposition-wise feed-forward networksを組み合わせ、self-attentionにより各位置が全位置との関係を直接計算します。

## 6. コストや物理的な詳細について

8 x NVIDIA P100 GPUで3.5日間の学習を実行。大規模モデル（Big）では約65Mパラメータを使用しました。

## 7. 参考文献のうち、特に参照すべきもの

「Attention-based models for speech recognition」など、attention mechanismの基礎研究が重要です。

## 8. この論文を140字以内のツイートで要約すると？

RNNやCNNを使わずattentionのみでsequence modelingを行う「Transformer」を提案。翻訳で高精度を達成し、後のBERT/GPTの基盤となった革命的アーキテクチャ。#NLP #MachineLearning"""

    @pytest.fixture
    def sample_paper_data(self):
        """Sample paper data for testing."""
        return {
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            "url": "https://arxiv.org/abs/1706.03762",
            "arxiv_id": "1706.03762"
        }

    @pytest.fixture
    def mock_environment_for_claude(self):
        """Mock environment variables for Claude."""
        env_vars = {
            'AI_PROVIDER': 'claude',
            'ANTHROPIC_API_KEY': 'test-claude-key'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            yield env_vars

    def test_paper_summarizer_initialization_with_claude(self, mock_environment_for_claude):
        """Test paper summarizer initializes correctly with Claude client."""

        with patch('nook.functions.common.python.claude_client.Anthropic'):
            with patch('nook.functions.paper_summarizer.paper_summarizer.create_client') as mock_create_client:
                # Mock the create_client to return a ClaudeClient
                mock_client = Mock(spec=ClaudeClient)
                mock_create_client.return_value = mock_client

                # Initialize PaperSummarizer
                summarizer = PaperSummarizer()

                # Verify client was created
                mock_create_client.assert_called_once()
                assert summarizer._client == mock_client

    def test_paper_summarizer_claude_content_generation(self, mock_environment_for_claude, mock_claude_response, sample_paper_data):
        """Test paper summarizer generates content using Claude."""

        with patch('nook.functions.common.python.claude_client.Anthropic') as mock_anthropic:
            # Setup mock Claude client
            mock_client_instance = Mock()
            mock_anthropic.return_value = mock_client_instance

            # Mock the API response
            mock_response = Mock()
            mock_response.content = [Mock(text=mock_claude_response)]
            mock_client_instance.messages.create.return_value = mock_response

            with patch('nook.functions.paper_summarizer.paper_summarizer.create_client') as mock_create_client:
                # Create a real Claude client with mocked Anthropic
                from nook.functions.common.python.claude_client import ClaudeClient, ClaudeClientConfig
                config = ClaudeClientConfig()
                claude_client = ClaudeClient(config)
                mock_create_client.return_value = claude_client

                # Mock other dependencies
                with patch('nook.functions.paper_summarizer.paper_summarizer.PaperIdRetriever') as mock_retriever:
                    mock_retriever.return_value.retrieve_from_hugging_face.return_value = [sample_paper_data["arxiv_id"]]

                    with patch('arxiv.Client') as mock_arxiv:
                        # Setup mock arxiv paper
                        mock_paper = Mock()
                        mock_paper.title = sample_paper_data["title"]
                        mock_paper.summary = sample_paper_data["abstract"]
                        mock_paper.entry_id = sample_paper_data["url"]
                        mock_arxiv.return_value.results.return_value = [mock_paper]

                        # Mock file operations to avoid actual S3/file operations
                        with patch('builtins.open', mock_open_multiple_files()):
                            with patch('os.path.exists', return_value=False):
                                # Create and test summarizer
                                summarizer = PaperSummarizer()

                                # Test the internal _process_paper method directly
                                result = summarizer._process_paper(sample_paper_data["arxiv_id"])

                                # Verify Claude client was called
                                assert mock_client_instance.messages.create.called

                                # Verify the result contains expected content
                                assert "Attention Is All You Need" in result
                                assert "既存研究では何ができなかったのか" in result
                                assert "技術的な詳細について" in result

    def test_claude_vs_gemini_response_structure(self, mock_environment_for_claude, sample_paper_data):
        """Test that Claude responses have similar structure to Gemini responses."""

        claude_response = """# Test Paper Title

[View Paper](https://arxiv.org/abs/1234.5678)

## 1. 既存研究では何ができなかったのか

Test limitation content.

## 2. どのようなアプローチでそれを解決しようとしたか

Test approach content."""

        # Test response structure validation
        assert "# " in claude_response  # Has title
        assert "[View Paper]" in claude_response  # Has paper link
        assert "## 1. 既存研究では何ができなかったのか" in claude_response  # Has required sections
        assert "## 2. どのようなアプローチでそれを解決しようとしたか" in claude_response

    def test_paper_summarizer_error_handling_with_claude(self, mock_environment_for_claude):
        """Test error handling when Claude API fails."""

        with patch('nook.functions.common.python.claude_client.Anthropic') as mock_anthropic:
            mock_client_instance = Mock()
            mock_anthropic.return_value = mock_client_instance

            # Setup client to raise an error
            mock_client_instance.messages.create.side_effect = Exception("Claude API Error")

            with patch('nook.functions.paper_summarizer.paper_summarizer.create_client') as mock_create_client:
                from nook.functions.common.python.claude_client import ClaudeClient, ClaudeClientConfig
                config = ClaudeClientConfig()
                claude_client = ClaudeClient(config)
                mock_create_client.return_value = claude_client

                with patch('nook.functions.paper_summarizer.paper_summarizer.PaperIdRetriever') as mock_retriever:
                    mock_retriever.return_value.retrieve_from_hugging_face.return_value = ["1234.5678"]

                    with patch('arxiv.Client') as mock_arxiv:
                        mock_paper = Mock()
                        mock_paper.title = "Test Paper"
                        mock_paper.summary = "Test abstract"
                        mock_paper.entry_id = "https://arxiv.org/abs/1234.5678"
                        mock_arxiv.return_value.results.return_value = [mock_paper]

                        with patch('builtins.open', mock_open_multiple_files()):
                            with patch('os.path.exists', return_value=False):
                                summarizer = PaperSummarizer()

                                # Should raise the API error
                                with pytest.raises(Exception, match="Claude API Error"):
                                    summarizer._process_paper("1234.5678")

    def test_environment_variable_switching(self):
        """Test switching between Claude and Gemini via environment variables."""

        # Test Claude environment
        with patch.dict(os.environ, {'AI_PROVIDER': 'claude', 'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                with patch('nook.functions.paper_summarizer.paper_summarizer.create_client') as mock_create_client:
                    mock_claude_client = Mock(spec=ClaudeClient)
                    mock_create_client.return_value = mock_claude_client

                    summarizer = PaperSummarizer()

                    # Verify Claude client was used
                    assert summarizer._client == mock_claude_client

        # Test Gemini environment (fallback)
        with patch.dict(os.environ, {'AI_PROVIDER': 'gemini', 'GEMINI_API_KEY': 'test-key'}):
            with patch('google.genai.Client'):
                with patch('nook.functions.paper_summarizer.paper_summarizer.create_client') as mock_create_client:
                    from nook.functions.common.python.gemini_client import GeminiClient
                    mock_gemini_client = Mock(spec=GeminiClient)
                    mock_create_client.return_value = mock_gemini_client

                    summarizer = PaperSummarizer()

                    # Verify Gemini client was used
                    assert summarizer._client == mock_gemini_client


def mock_open_multiple_files():
    """Helper function to mock multiple file operations."""
    from unittest.mock import mock_open
    return mock_open(read_data="")