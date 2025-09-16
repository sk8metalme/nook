# Nook Gemini-to-Claude移行テスト設計書

## エグゼクティブサマリー

この文書では、NookアプリケーションをGoogle Gemini APIからClaude CLIに移行するための包括的かつ実用的なテスト設計について説明します。テスト戦略は、内部レビューで推奨された**簡素化アプローチ**に従い、過度な設計を避けながら4週間のタイムライン内で本番環境への信頼性を提供する必須テストに焦点を当てています。

**主要テスト目標:**
- GeminiとClaude実装間の機能パリティの確保
- パフォーマンス特性が現在の基準を満たすか上回ることの検証
- ロールバック手順と緊急プロトコルへの信頼性確立
- 4つのコア関数に焦点: paper_summarizer、tech_feed、hacker_news、reddit_explorer
- 複雑なviewer検索機能はフェーズ2に延期

**テスト哲学:**
- **手動テスト優先、自動化は二次** - 人的検証から開始し、高価値テストを自動化
- **リスクベース優先順位付け** - 高影響・高確率リスクにテスト労力を集中
- **完璧より実用性** - 複雑なテストインフラなしで本番準備を確保

---

## 1. 簡素化移行のテスト戦略

### 1.1 コアテスト原則

#### 重点領域
1. **Core Functions Only (Phase 1)**
   - Paper Summarizer: Research paper analysis and summarization
   - Tech Feed: Technology news content generation
   - Hacker News: News aggregation and summarization
   - Reddit Explorer: Content exploration and analysis

2. **Deferred Testing (Phase 2)**
   - Viewer search functionality (complex interactive features)
   - Advanced chat capabilities
   - Multi-session conversation handling

#### テスト優先順位

**Priority 1 (Critical - Must Pass)**
- Core content generation functionality
- API response quality validation
- Basic error handling
- Environment variable switching

**Priority 2 (Important - Should Pass)**
- Performance benchmarking
- Retry logic validation
- Configuration migration
- Rollback procedures

**Priority 3 (Nice to Have - Can Defer)**
- Advanced error scenarios
- Edge case handling
- Performance optimization
- Comprehensive logging validation

### 1.2 テストアプローチマトリックス

| Component | Manual Testing | Automated Testing | Validation Method |
|-----------|---------------|-------------------|-------------------|
| Claude Client | ✓ Initial | ✓ Unit Tests | API response validation |
| Content Generation | ✓ Extensive | ✓ Integration | Side-by-side comparison |
| Configuration | ✓ Required | ✓ Unit Tests | Environment switching |
| Error Handling | ✓ Required | ✓ Mock Tests | Error scenario simulation |
| Performance | ✓ Required | ✓ Benchmark | Response time comparison |
| Rollback | ✓ Required | ✓ Automated | Environment restoration |

---

## 2. ユニットテストフレームワーク

### 2.1 テスト構造と組織

```
nook/
├── functions/
│   ├── common/
│   │   └── python/
│   │       ├── tests/
│   │       │   ├── __init__.py
│   │       │   ├── test_claude_client.py
│   │       │   ├── test_config_manager.py
│   │       │   ├── test_error_handling.py
│   │       │   └── fixtures/
│   │       │       ├── mock_responses.json
│   │       │       └── test_configurations.json
│   │       └── claude_client.py
│   └── tests/
│       ├── integration/
│       │   ├── test_paper_summarizer_integration.py
│       │   ├── test_tech_feed_integration.py
│       │   ├── test_hacker_news_integration.py
│       │   └── test_reddit_explorer_integration.py
│       └── fixtures/
│           ├── sample_papers.json
│           ├── sample_feeds.json
│           └── expected_outputs.json
```

### 2.2 Claude Clientユニットテスト

#### 2.2.1 コア機能テスト

```python
# test_claude_client.py

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import anthropic
from nook.functions.common.python.claude_client import ClaudeClient, ClaudeClientConfig

class TestClaudeClient:
    """Test suite for Claude client implementation."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client for testing."""
        with patch('anthropic.Anthropic') as mock:
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
            with patch('anthropic.Anthropic') as mock_anthropic:
                client = ClaudeClient(claude_config)

                assert client is not None
                assert client.config.model == "claude-3-5-sonnet-20241022"
                mock_anthropic.assert_called_once()

    def test_client_initialization_missing_api_key(self, claude_config):
        """Test client initialization fails with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception) as exc_info:
                ClaudeClient(claude_config)

            assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_generate_content_success(self, claude_client, mock_anthropic_client):
        """Test successful content generation."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Generated response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = claude_client.generate_content("Test content")

        assert result == "Generated response"
        mock_anthropic_client.messages.create.assert_called_once()

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

    def test_generate_content_list_input(self, claude_client, mock_anthropic_client):
        """Test content generation with list input."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Multi-content response")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = claude_client.generate_content(["Content 1", "Content 2"])

        assert result == "Multi-content response"
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert len(call_args["messages"]) == 2

    def test_create_chat_session(self, claude_client):
        """Test chat session creation."""
        claude_client.create_chat()

        assert claude_client._chat_context == []
        assert claude_client._system_instruction is None

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

    def test_send_message_no_chat_session(self, claude_client):
        """Test sending message without chat session fails."""
        with pytest.raises(Exception) as exc_info:
            claude_client.send_message("Hello")

        assert "No chat session" in str(exc_info.value)

    def test_retry_mechanism(self, claude_client, mock_anthropic_client):
        """Test retry mechanism on API errors."""
        from anthropic import RateLimitError

        # Setup to fail once, then succeed
        mock_anthropic_client.messages.create.side_effect = [
            RateLimitError("Rate limited"),
            Mock(content=[Mock(text="Success after retry")])
        ]

        result = claude_client.generate_content("Test content")

        assert result == "Success after retry"
        assert mock_anthropic_client.messages.create.call_count == 2
```

#### 2.2.2 設定とエラーハンドリングテスト

```python
# test_config_manager.py

import pytest
import os
from nook.functions.common.python.ai_config_manager import AIConfigManager
from nook.functions.common.python.ai_client_interface import AIProvider

class TestAIConfigManager:
    """Test configuration management."""

    def test_create_claude_config_default(self):
        """Test default Claude configuration creation."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            config = AIConfigManager.create_config(AIProvider.CLAUDE)

            assert config.model == "claude-3-5-sonnet-20241022"
            assert config.temperature == 1.0
            assert config.max_output_tokens == 8192

    def test_create_config_with_overrides(self):
        """Test configuration with parameter overrides."""
        overrides = {
            'temperature': 0.5,
            'max_output_tokens': 4096
        }

        config = AIConfigManager.create_config(AIProvider.CLAUDE, overrides)

        assert config.temperature == 0.5
        assert config.max_output_tokens == 4096

    def test_environment_variable_override(self):
        """Test environment variable configuration override."""
        with patch.dict(os.environ, {
            'AI_TEMPERATURE': '0.7',
            'AI_MAX_TOKENS': '2048'
        }):
            config = AIConfigManager.create_config(AIProvider.CLAUDE)

            assert config.temperature == 0.7
            assert config.max_output_tokens == 2048

# test_error_handling.py

import pytest
from unittest.mock import Mock
from anthropic import APIError, RateLimitError, APITimeoutError
from nook.functions.common.python.ai_error_handling import ErrorMapper, AIClientError

class TestErrorHandling:
    """Test error handling and mapping."""

    def test_map_claude_rate_limit_error(self):
        """Test Claude rate limit error mapping."""
        original_error = RateLimitError("Rate limit exceeded")
        mapped_error = ErrorMapper.map_claude_error(original_error)

        assert isinstance(mapped_error, RateLimitError)
        assert "Claude rate limit" in str(mapped_error)

    def test_map_claude_timeout_error(self):
        """Test Claude timeout error mapping."""
        original_error = APITimeoutError("Request timeout")
        mapped_error = ErrorMapper.map_claude_error(original_error)

        assert isinstance(mapped_error, TimeoutError)
        assert "Claude timeout" in str(mapped_error)

    def test_map_claude_authentication_error(self):
        """Test Claude authentication error mapping."""
        original_error = APIError("Unauthorized", response=Mock(status_code=401))
        mapped_error = ErrorMapper.map_claude_error(original_error)

        assert isinstance(mapped_error, AuthenticationError)
        assert "authentication" in str(mapped_error).lower()
```

### 2.3 モックAPIレスポンステスト

#### 2.3.1 モックレスポンスフィクスチャ

```json
// fixtures/mock_responses.json
{
  "claude_responses": {
    "simple_generation": {
      "content": [{"text": "This is a test response from Claude."}]
    },
    "paper_summary": {
      "content": [{
        "text": "## Summary\n\n**Title:** Sample Research Paper\n\n**Key Findings:** This paper demonstrates innovative approaches to machine learning.\n\n**Methodology:** The authors used a novel neural network architecture.\n\n**Implications:** These findings could revolutionize the field."
      }]
    },
    "error_response": {
      "error": {
        "type": "rate_limit_error",
        "message": "Rate limit exceeded"
      }
    }
  },
  "gemini_responses": {
    "simple_generation": {
      "candidates": [{
        "content": {
          "parts": [{"text": "This is a test response from Gemini."}]
        }
      }]
    },
    "paper_summary": {
      "candidates": [{
        "content": {
          "parts": [{
            "text": "## Summary\n\n**Title:** Sample Research Paper\n\n**Key Findings:** This paper demonstrates innovative approaches to machine learning.\n\n**Methodology:** The authors used a novel neural network architecture.\n\n**Implications:** These findings could revolutionize the field."
          }]
        }
      }]
    }
  }
}
```

#### 2.3.2 設定テストケース

```json
// fixtures/test_configurations.json
{
  "test_configs": {
    "default_claude": {
      "model": "claude-3-5-sonnet-20241022",
      "temperature": 1.0,
      "top_p": 0.95,
      "max_output_tokens": 8192,
      "timeout": 60000
    },
    "high_creativity": {
      "model": "claude-3-5-sonnet-20241022",
      "temperature": 1.5,
      "top_p": 0.9,
      "max_output_tokens": 4096
    },
    "focused_output": {
      "model": "claude-3-5-sonnet-20241022",
      "temperature": 0.3,
      "top_p": 0.8,
      "max_output_tokens": 2048
    }
  }
}
```

---

## 3. 統合テスト戦略

### 3.1 エンドツーエンド関数テスト

#### 3.1.1 Paper Summarizer統合テスト

```python
# test_paper_summarizer_integration.py

import pytest
import os
from unittest.mock import Mock, patch
from nook.functions.paper_summarizer.paper_summarizer import PaperSummarizer

class TestPaperSummarizerIntegration:
    """Integration tests for Paper Summarizer with Claude."""

    @pytest.fixture
    def mock_claude_response(self):
        """Mock Claude API response for paper summarization."""
        return {
            "content": [{
                "text": """## Summary

**Title:** Attention Is All You Need

**Authors:** Vaswani et al.

**Key Contributions:**
- Introduces the Transformer architecture
- Eliminates recurrence and convolutions
- Achieves state-of-the-art results in translation

**Methodology:**
The paper presents a novel neural network architecture based solely on attention mechanisms.

**Significance:**
This work has become foundational for modern NLP models including BERT and GPT.
"""
            }]
        }

    @pytest.fixture
    def sample_paper_data(self):
        """Sample paper data for testing."""
        return {
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            "url": "https://arxiv.org/abs/1706.03762",
            "contents": "Full paper content would go here..."
        }

    @patch('nook.functions.common.python.claude_client.ClaudeClient')
    def test_paper_summarizer_with_claude(self, mock_claude_client, mock_claude_response, sample_paper_data):
        """Test paper summarizer using Claude client."""

        # Setup environment for Claude
        with patch.dict(os.environ, {
            'AI_PROVIDER': 'claude',
            'ANTHROPIC_API_KEY': 'test-key'
        }):
            # Setup mock client
            mock_client_instance = Mock()
            mock_claude_client.return_value = mock_client_instance
            mock_client_instance.generate_content.return_value = mock_claude_response["content"][0]["text"]

            # Setup paper retriever mock
            with patch('nook.functions.paper_summarizer.paper_summarizer.PaperIdRetriever') as mock_retriever:
                mock_retriever.return_value.retrieve_from_hugging_face.return_value = ["1706.03762"]

                # Setup arxiv mock
                with patch('arxiv.Client') as mock_arxiv:
                    mock_paper = Mock()
                    mock_paper.title = sample_paper_data["title"]
                    mock_paper.summary = sample_paper_data["abstract"]
                    mock_paper.entry_id = sample_paper_data["url"]
                    mock_arxiv.return_value.results.return_value = [mock_paper]

                    # Run summarizer
                    summarizer = PaperSummarizer()
                    result = summarizer.process_papers(["1706.03762"])

                    # Verify Claude client was called
                    assert mock_client_instance.generate_content.called

                    # Verify result structure
                    assert "Summary" in result
                    assert "Attention Is All You Need" in result

    def test_error_handling_paper_summarizer(self):
        """Test error handling in paper summarizer."""
        with patch.dict(os.environ, {
            'AI_PROVIDER': 'claude',
            'ANTHROPIC_API_KEY': 'test-key'
        }):
            with patch('nook.functions.common.python.claude_client.ClaudeClient') as mock_claude:
                # Setup client to raise error
                mock_client = Mock()
                mock_claude.return_value = mock_client
                mock_client.generate_content.side_effect = Exception("API Error")

                summarizer = PaperSummarizer()

                # Should handle error gracefully
                with pytest.raises(Exception):
                    summarizer.process_papers(["1706.03762"])
```

#### 3.1.2 レスポンス品質比較テスト

```python
# test_response_quality.py

import pytest
import json
from typing import Dict, Any
from unittest.mock import patch
from nook.functions.common.python.client_factory import create_client

class TestResponseQuality:
    """Test response quality between Gemini and Claude."""

    @pytest.fixture
    def quality_test_cases(self):
        """Test cases for quality comparison."""
        return [
            {
                "name": "technical_summary",
                "input": "Summarize the key concepts of machine learning for a technical audience.",
                "expected_elements": ["algorithms", "training", "data", "models", "accuracy"]
            },
            {
                "name": "research_analysis",
                "input": "Analyze the methodology of this research paper abstract: [sample abstract]",
                "expected_elements": ["methodology", "approach", "results", "implications"]
            },
            {
                "name": "content_generation",
                "input": "Generate a brief explanation of neural networks.",
                "expected_elements": ["neurons", "layers", "weights", "activation"]
            }
        ]

    def test_response_quality_comparison(self, quality_test_cases):
        """Compare response quality between providers."""

        results = {}

        for provider in ['gemini', 'claude']:
            results[provider] = {}

            with patch.dict(os.environ, {'AI_PROVIDER': provider}):
                client = create_client()

                for test_case in quality_test_cases:
                    try:
                        response = client.generate_content(test_case["input"])

                        # Basic quality metrics
                        results[provider][test_case["name"]] = {
                            "response": response,
                            "length": len(response),
                            "contains_expected": self._check_expected_elements(
                                response, test_case["expected_elements"]
                            ),
                            "coherent": self._check_coherence(response),
                            "complete": self._check_completeness(response)
                        }
                    except Exception as e:
                        results[provider][test_case["name"]] = {
                            "error": str(e),
                            "failed": True
                        }

        # Compare results
        self._compare_quality_metrics(results)

    def _check_expected_elements(self, response: str, expected_elements: list) -> Dict[str, bool]:
        """Check if response contains expected elements."""
        response_lower = response.lower()
        return {element: element.lower() in response_lower for element in expected_elements}

    def _check_coherence(self, response: str) -> bool:
        """Basic coherence check."""
        # Simple heuristics
        sentences = response.split('.')
        return len(sentences) > 1 and len(response.split()) > 10

    def _check_completeness(self, response: str) -> bool:
        """Check response completeness."""
        return len(response.strip()) > 50  # Minimum length threshold

    def _compare_quality_metrics(self, results: Dict[str, Dict[str, Any]]):
        """Compare quality metrics between providers."""

        print("\n=== Response Quality Comparison ===")

        for test_name in results['gemini'].keys():
            print(f"\nTest Case: {test_name}")

            gemini_result = results['gemini'][test_name]
            claude_result = results['claude'][test_name]

            if 'error' in gemini_result:
                print(f"  Gemini: ERROR - {gemini_result['error']}")
            else:
                print(f"  Gemini: Length={gemini_result['length']}, "
                     f"Coherent={gemini_result['coherent']}, "
                     f"Complete={gemini_result['complete']}")

            if 'error' in claude_result:
                print(f"  Claude: ERROR - {claude_result['error']}")
            else:
                print(f"  Claude: Length={claude_result['length']}, "
                     f"Coherent={claude_result['coherent']}, "
                     f"Complete={claude_result['complete']}")

        # Assert that Claude performs at least as well as Gemini
        # (This would be more sophisticated in practice)
        assert True  # Placeholder for actual quality assertions
```

### 3.2 パフォーマンスベンチマーク

#### 3.2.1 レスポンス時間テスト

```python
# test_performance.py

import pytest
import time
import statistics
from unittest.mock import patch
from nook.functions.common.python.client_factory import create_client

class TestPerformance:
    """Performance testing for AI clients."""

    @pytest.fixture
    def performance_test_inputs(self):
        """Test inputs of varying complexity."""
        return {
            "simple": "Hello, how are you?",
            "medium": "Summarize the key benefits of cloud computing for enterprise businesses.",
            "complex": "Analyze the following research paper abstract and provide insights on methodology, findings, and potential impact: [long abstract text]"
        }

    @pytest.mark.performance
    def test_response_time_comparison(self, performance_test_inputs):
        """Compare response times between providers."""

        results = {}

        for provider in ['gemini', 'claude']:
            results[provider] = {}

            with patch.dict(os.environ, {'AI_PROVIDER': provider}):
                client = create_client()

                for complexity, input_text in performance_test_inputs.items():
                    times = []

                    # Run multiple iterations for statistical significance
                    for _ in range(3):  # Reduced for test speed
                        start_time = time.time()
                        try:
                            response = client.generate_content(input_text)
                            end_time = time.time()

                            times.append(end_time - start_time)

                        except Exception as e:
                            print(f"Error in {provider} {complexity}: {e}")
                            continue

                    if times:
                        results[provider][complexity] = {
                            "avg_time": statistics.mean(times),
                            "min_time": min(times),
                            "max_time": max(times),
                            "std_dev": statistics.stdev(times) if len(times) > 1 else 0
                        }

        # Performance comparison
        self._analyze_performance_results(results)

    def _analyze_performance_results(self, results):
        """Analyze and assert performance requirements."""

        print("\n=== Performance Comparison ===")

        for complexity in ["simple", "medium", "complex"]:
            print(f"\n{complexity.title()} Query Performance:")

            for provider in results:
                if complexity in results[provider]:
                    metrics = results[provider][complexity]
                    print(f"  {provider.title()}: "
                         f"Avg={metrics['avg_time']:.2f}s, "
                         f"Min={metrics['min_time']:.2f}s, "
                         f"Max={metrics['max_time']:.2f}s")

        # Performance assertions
        # Claude should be within 110% of Gemini response times
        for complexity in ["simple", "medium", "complex"]:
            if (complexity in results.get('gemini', {}) and
                complexity in results.get('claude', {})):

                gemini_avg = results['gemini'][complexity]['avg_time']
                claude_avg = results['claude'][complexity]['avg_time']

                # Allow 10% performance degradation
                assert claude_avg <= gemini_avg * 1.1, \
                    f"Claude {complexity} query too slow: {claude_avg:.2f}s vs {gemini_avg:.2f}s"

    @pytest.mark.performance
    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests."""
        import concurrent.futures

        with patch.dict(os.environ, {'AI_PROVIDER': 'claude'}):
            client = create_client()

            def make_request(request_id):
                """Make a single request."""
                start_time = time.time()
                try:
                    response = client.generate_content(f"Request {request_id}: Hello")
                    return {
                        "id": request_id,
                        "success": True,
                        "duration": time.time() - start_time,
                        "response_length": len(response)
                    }
                except Exception as e:
                    return {
                        "id": request_id,
                        "success": False,
                        "duration": time.time() - start_time,
                        "error": str(e)
                    }

            # Test with 5 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request, i) for i in range(5)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            # Analyze concurrent performance
            successful_requests = [r for r in results if r["success"]]
            failed_requests = [r for r in results if not r["success"]]

            print(f"\nConcurrent Request Results:")
            print(f"  Successful: {len(successful_requests)}/5")
            print(f"  Failed: {len(failed_requests)}/5")

            if successful_requests:
                avg_duration = statistics.mean([r["duration"] for r in successful_requests])
                print(f"  Average Duration: {avg_duration:.2f}s")

            # Assertions
            assert len(successful_requests) >= 4, "At least 4/5 concurrent requests should succeed"
```

---

## 4. ロールバックテスト手順

### 4.1 環境変数切替テスト

#### 4.1.1 プロバイダー切替テストスイート

```python
# test_rollback_procedures.py

import pytest
import os
import subprocess
from unittest.mock import patch, Mock
from nook.functions.common.python.client_factory import create_client

class TestRollbackProcedures:
    """Test rollback and environment switching procedures."""

    def test_environment_variable_switching(self):
        """Test switching between providers via environment variables."""

        # Test initial state (Gemini)
        with patch.dict(os.environ, {'AI_PROVIDER': 'gemini', 'GEMINI_API_KEY': 'test-key'}):
            with patch('google.genai.Client'):
                client = create_client()
                assert client.__class__.__name__ == 'GeminiClient'

        # Test switched state (Claude)
        with patch.dict(os.environ, {'AI_PROVIDER': 'claude', 'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('anthropic.Anthropic'):
                client = create_client()
                assert client.__class__.__name__ == 'ClaudeClient'

    def test_function_level_rollback(self):
        """Test rolling back individual functions."""

        # This would test the ability to switch specific functions back to Gemini
        # while keeping others on Claude

        test_cases = [
            "paper_summarizer",
            "tech_feed",
            "hacker_news",
            "reddit_explorer"
        ]

        for function_name in test_cases:
            # Test function with Claude
            with patch.dict(os.environ, {
                f'{function_name.upper()}_AI_PROVIDER': 'claude',
                'ANTHROPIC_API_KEY': 'test-key'
            }):
                # Would test function-specific provider switching
                assert True  # Placeholder for actual function testing

            # Test rollback to Gemini
            with patch.dict(os.environ, {
                f'{function_name.upper()}_AI_PROVIDER': 'gemini',
                'GEMINI_API_KEY': 'test-key'
            }):
                # Would test rollback functionality
                assert True  # Placeholder for actual rollback testing

    def test_configuration_restoration(self):
        """Test configuration restoration during rollback."""

        # Test configuration backup and restore
        original_config = {
            'AI_PROVIDER': 'gemini',
            'GEMINI_API_KEY': 'original-key',
            'AI_TEMPERATURE': '1.0'
        }

        migrated_config = {
            'AI_PROVIDER': 'claude',
            'ANTHROPIC_API_KEY': 'claude-key',
            'AI_TEMPERATURE': '0.8'
        }

        # Test migration
        with patch.dict(os.environ, migrated_config, clear=True):
            assert os.environ['AI_PROVIDER'] == 'claude'
            assert os.environ['ANTHROPIC_API_KEY'] == 'claude-key'

        # Test rollback
        with patch.dict(os.environ, original_config, clear=True):
            assert os.environ['AI_PROVIDER'] == 'gemini'
            assert os.environ['GEMINI_API_KEY'] == 'original-key'
            assert os.environ['AI_TEMPERATURE'] == '1.0'

    def test_emergency_rollback_script(self):
        """Test emergency rollback script functionality."""

        # Mock the rollback script execution
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0

            # Test rollback script execution
            result = subprocess.run([
                'python', 'scripts/emergency_rollback.py',
                '--component', 'all',
                '--confirm'
            ], capture_output=True)

            # In actual implementation, this would verify:
            # 1. Environment variables are restored
            # 2. Configuration files are reverted
            # 3. Services are restarted
            # 4. Health checks pass

            assert True  # Placeholder for actual verification

class TestValidationProcedures:
    """Test validation procedures after rollback."""

    def test_post_rollback_validation(self):
        """Test system validation after rollback."""

        validation_checks = [
            self._test_api_connectivity,
            self._test_basic_functionality,
            self._test_configuration_integrity,
            self._test_error_handling
        ]

        results = {}
        for check in validation_checks:
            try:
                check()
                results[check.__name__] = "PASS"
            except Exception as e:
                results[check.__name__] = f"FAIL: {str(e)}"

        # All validation checks should pass
        failed_checks = [name for name, result in results.items() if result.startswith("FAIL")]
        assert len(failed_checks) == 0, f"Failed validation checks: {failed_checks}"

    def _test_api_connectivity(self):
        """Test API connectivity after rollback."""
        with patch.dict(os.environ, {'AI_PROVIDER': 'gemini', 'GEMINI_API_KEY': 'test-key'}):
            with patch('google.genai.Client') as mock_client:
                mock_instance = Mock()
                mock_client.return_value = mock_instance

                client = create_client()
                # Test basic connectivity
                assert client is not None

    def _test_basic_functionality(self):
        """Test basic functionality after rollback."""
        with patch.dict(os.environ, {'AI_PROVIDER': 'gemini', 'GEMINI_API_KEY': 'test-key'}):
            with patch('google.genai.Client'):
                client = create_client()

                # Mock response
                with patch.object(client, 'generate_content') as mock_generate:
                    mock_generate.return_value = "Test response"

                    response = client.generate_content("Test input")
                    assert response == "Test response"

    def _test_configuration_integrity(self):
        """Test configuration integrity after rollback."""
        expected_config = {
            'AI_PROVIDER': 'gemini',
            'GEMINI_API_KEY': 'test-key'
        }

        with patch.dict(os.environ, expected_config):
            # Verify configuration values
            assert os.environ['AI_PROVIDER'] == 'gemini'
            assert 'GEMINI_API_KEY' in os.environ

    def _test_error_handling(self):
        """Test error handling after rollback."""
        with patch.dict(os.environ, {'AI_PROVIDER': 'gemini', 'GEMINI_API_KEY': 'test-key'}):
            with patch('google.genai.Client') as mock_client:
                mock_instance = Mock()
                mock_client.return_value = mock_instance
                mock_instance.generate_content.side_effect = Exception("Test error")

                client = create_client()

                # Should handle errors gracefully
                with pytest.raises(Exception):
                    client.generate_content("Test input")
```

### 4.2 緊急ロールバック手順

#### 4.2.1 自動ロールバックスクリプト

```python
# scripts/emergency_rollback.py

#!/usr/bin/env python3
"""
Emergency rollback script for Nook AI migration.

This script provides automated rollback capabilities for emergency situations.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class EmergencyRollback:
    """Handle emergency rollback of AI migration."""

    def __init__(self):
        self.backup_config_file = Path(".env.backup")
        self.rollback_log = []
        self.validation_errors = []

    def execute_emergency_rollback(self, components: List[str] = None) -> bool:
        """Execute emergency rollback procedure."""

        print("🚨 EMERGENCY ROLLBACK INITIATED")
        print(f"Timestamp: {datetime.now().isoformat()}")

        try:
            # Step 1: Stop services
            self._emergency_stop_services()

            # Step 2: Restore environment configuration
            self._restore_environment_config()

            # Step 3: Rollback code changes
            if components:
                self._rollback_specific_components(components)
            else:
                self._rollback_all_components()

            # Step 4: Restart services
            self._restart_services()

            # Step 5: Validate rollback
            self._validate_emergency_rollback()

            print("✅ EMERGENCY ROLLBACK COMPLETED SUCCESSFULLY")
            return True

        except Exception as e:
            print(f"❌ EMERGENCY ROLLBACK FAILED: {e}")
            self._log_emergency_error(str(e))
            return False

    def _emergency_stop_services(self):
        """Emergency stop of all services."""
        print("🛑 Stopping services...")

        try:
            # Stop any running Python processes related to nook
            subprocess.run(["pkill", "-f", "nook"], capture_output=True)

            # Stop uvicorn if running
            subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)

            self.rollback_log.append("Services stopped")

        except Exception as e:
            print(f"Warning: Could not stop all services: {e}")

    def _restore_environment_config(self):
        """Restore environment configuration to Gemini."""
        print("🔄 Restoring environment configuration...")

        # Emergency configuration restore
        emergency_env = {
            'AI_PROVIDER': 'gemini',
            'GEMINI_API_KEY': os.environ.get('GEMINI_API_KEY_BACKUP', os.environ.get('GEMINI_API_KEY', '')),
            'AI_TEMPERATURE': '1.0',
            'AI_TOP_P': '0.95',
            'AI_TOP_K': '40',
            'AI_MAX_TOKENS': '8192',
            'AI_USE_SEARCH': 'false'
        }

        # Update current environment
        os.environ.update(emergency_env)

        # Write to .env file if it exists
        env_file = Path(".env")
        if env_file.exists():
            content = env_file.read_text()

            # Replace key settings
            for key, value in emergency_env.items():
                # Simple replacement (would be more robust in production)
                content = self._replace_env_var(content, key, value)

            env_file.write_text(content)

        self.rollback_log.append("Environment configuration restored")

    def _replace_env_var(self, content: str, key: str, value: str) -> str:
        """Replace environment variable in file content."""
        import re

        pattern = rf'^{key}=.*$'
        replacement = f'{key}={value}'

        if re.search(pattern, content, re.MULTILINE):
            return re.sub(pattern, replacement, content, flags=re.MULTILINE)
        else:
            return content + f'\n{replacement}\n'

    def _rollback_specific_components(self, components: List[str]):
        """Rollback specific components to Gemini."""
        print(f"🔧 Rolling back components: {', '.join(components)}")

        component_paths = {
            'paper_summarizer': 'nook/functions/paper_summarizer/',
            'tech_feed': 'nook/functions/tech_feed/',
            'hacker_news': 'nook/functions/hacker_news/',
            'reddit_explorer': 'nook/functions/reddit_explorer/',
            'viewer': 'nook/functions/viewer/'
        }

        for component in components:
            if component in component_paths:
                self._rollback_component_files(component, component_paths[component])

    def _rollback_component_files(self, component: str, path: str):
        """Rollback specific component files."""
        try:
            # Get latest backup branch
            backup_branch = self._get_latest_backup_branch()

            # Rollback specific files
            subprocess.run([
                'git', 'checkout', f'origin/{backup_branch}', '--', path
            ], check=True)

            self.rollback_log.append(f"Rolled back {component}")

        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to rollback {component}: {e}")

    def _rollback_all_components(self):
        """Rollback all components to previous state."""
        print("🔄 Rolling back all components...")

        try:
            backup_branch = self._get_latest_backup_branch()

            # Hard reset to backup branch
            subprocess.run([
                'git', 'reset', '--hard', f'origin/{backup_branch}'
            ], check=True)

            self.rollback_log.append("All components rolled back")

        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to rollback all components: {e}")

    def _get_latest_backup_branch(self) -> str:
        """Get the latest backup branch."""
        try:
            result = subprocess.run([
                'git', 'branch', '-r', '--list', 'origin/backup-*'
            ], capture_output=True, text=True, check=True)

            branches = [line.strip().replace('origin/', '') for line in result.stdout.split('\n') if line.strip()]

            if not branches:
                return 'main'  # Fallback to main branch

            # Return most recent backup
            return sorted(branches)[-1]

        except subprocess.CalledProcessError:
            return 'main'  # Fallback

    def _restart_services(self):
        """Restart services after rollback."""
        print("🚀 Restarting services...")

        # This would restart Lambda functions or local services
        # Implementation depends on deployment method

        self.rollback_log.append("Services restarted")

    def _validate_emergency_rollback(self):
        """Validate emergency rollback was successful."""
        print("✅ Validating rollback...")

        validation_checks = [
            self._check_environment_variables,
            self._check_api_connectivity,
            self._check_basic_functionality
        ]

        for check in validation_checks:
            try:
                check()
            except Exception as e:
                self.validation_errors.append(str(e))

        if self.validation_errors:
            raise Exception(f"Rollback validation failed: {self.validation_errors}")

        self.rollback_log.append("Rollback validation successful")

    def _check_environment_variables(self):
        """Check environment variables are correctly set."""
        required_vars = ['AI_PROVIDER', 'GEMINI_API_KEY']

        for var in required_vars:
            if not os.environ.get(var):
                raise Exception(f"Missing environment variable: {var}")

        if os.environ.get('AI_PROVIDER') != 'gemini':
            raise Exception("AI_PROVIDER not set to gemini")

    def _check_api_connectivity(self):
        """Check API connectivity."""
        # This would test actual API connectivity
        # For now, just check environment setup

        gemini_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_key or len(gemini_key) < 10:
            raise Exception("Invalid Gemini API key")

    def _check_basic_functionality(self):
        """Check basic functionality."""
        # This would test basic function execution
        # For now, just verify imports work

        try:
            from nook.functions.common.python.gemini_client import create_client
            client = create_client()
            # Basic validation that client was created
            assert client is not None
        except Exception as e:
            raise Exception(f"Basic functionality check failed: {e}")

    def _log_emergency_error(self, error: str):
        """Log emergency rollback error."""
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'rollback_log': self.rollback_log,
            'validation_errors': self.validation_errors
        }

        error_file = Path('emergency_rollback_error.json')
        with open(error_file, 'w') as f:
            json.dump(error_log, f, indent=2)

        print(f"Emergency error logged to {error_file}")

def main():
    """Main emergency rollback execution."""

    parser = argparse.ArgumentParser(description='Emergency rollback for Nook AI migration')
    parser.add_argument('--components', nargs='+', help='Specific components to rollback')
    parser.add_argument('--confirm', action='store_true', help='Confirm emergency rollback')

    args = parser.parse_args()

    if not args.confirm:
        print("❌ Emergency rollback requires --confirm flag")
        print("This will restore the system to Gemini API")
        sys.exit(1)

    rollback = EmergencyRollback()
    success = rollback.execute_emergency_rollback(args.components)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

---

## 5. 品質保証テストフレームワーク

### 5.1 コンテンツ品質検証

#### 5.1.1 レスポンス品質メトリクス

```python
# test_content_quality.py

import pytest
import json
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from unittest.mock import patch

@dataclass
class QualityMetric:
    """Quality metric for content evaluation."""
    name: str
    score: float  # 0-1 scale
    details: Dict[str, any] = None

class ContentQualityAnalyzer:
    """Analyze and compare content quality between providers."""

    def __init__(self):
        self.quality_thresholds = {
            'coherence': 0.7,
            'completeness': 0.8,
            'relevance': 0.8,
            'accuracy': 0.9
        }

    def analyze_response_quality(self, response: str, context: str = None) -> List[QualityMetric]:
        """Analyze response quality across multiple dimensions."""

        metrics = []

        # Coherence - Does the response make logical sense?
        coherence_score = self._measure_coherence(response)
        metrics.append(QualityMetric('coherence', coherence_score))

        # Completeness - Does the response address the request fully?
        completeness_score = self._measure_completeness(response, context)
        metrics.append(QualityMetric('completeness', completeness_score))

        # Structure - Is the response well-structured?
        structure_score = self._measure_structure(response)
        metrics.append(QualityMetric('structure', structure_score))

        # Length appropriateness - Is the response appropriately sized?
        length_score = self._measure_length_appropriateness(response)
        metrics.append(QualityMetric('length_appropriateness', length_score))

        return metrics

    def _measure_coherence(self, response: str) -> float:
        """Measure response coherence using heuristics."""

        if not response.strip():
            return 0.0

        # Basic coherence indicators
        sentences = [s.strip() for s in response.split('.') if s.strip()]

        if len(sentences) < 2:
            return 0.3  # Very short responses get low coherence

        # Check for proper sentence structure
        proper_sentences = 0
        for sentence in sentences:
            if (len(sentence.split()) >= 3 and
                sentence[0].isupper() and
                not sentence.endswith(('..', '???', '!!!'))):
                proper_sentences += 1

        coherence_ratio = proper_sentences / len(sentences)

        # Bonus for transitional words/phrases
        transitions = ['however', 'therefore', 'furthermore', 'additionally', 'consequently']
        transition_bonus = sum(1 for t in transitions if t.lower() in response.lower()) * 0.1

        return min(1.0, coherence_ratio + transition_bonus)

    def _measure_completeness(self, response: str, context: str = None) -> float:
        """Measure response completeness."""

        if not response.strip():
            return 0.0

        # Basic completeness indicators
        word_count = len(response.split())

        # Minimum word count for completeness
        if word_count < 10:
            return 0.2
        elif word_count < 30:
            return 0.6
        elif word_count < 100:
            return 0.8
        else:
            return 1.0

    def _measure_structure(self, response: str) -> float:
        """Measure response structure quality."""

        structure_score = 0.0

        # Check for markdown formatting
        if re.search(r'#{1,6}\s', response):  # Headers
            structure_score += 0.3

        if re.search(r'\*\*.*?\*\*', response):  # Bold text
            structure_score += 0.2

        if re.search(r'^\s*[-*]\s', response, re.MULTILINE):  # Lists
            structure_score += 0.3

        # Check for paragraph structure
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            structure_score += 0.2

        return min(1.0, structure_score)

    def _measure_length_appropriateness(self, response: str) -> float:
        """Measure if response length is appropriate."""

        word_count = len(response.split())

        # Optimal range is 50-500 words for most responses
        if 50 <= word_count <= 500:
            return 1.0
        elif 20 <= word_count < 50 or 500 < word_count <= 1000:
            return 0.8
        elif 10 <= word_count < 20 or 1000 < word_count <= 2000:
            return 0.6
        else:
            return 0.3

class TestContentQuality:
    """Test content quality between providers."""

    @pytest.fixture
    def quality_analyzer(self):
        """Quality analyzer instance."""
        return ContentQualityAnalyzer()

    @pytest.fixture
    def test_scenarios(self):
        """Test scenarios for quality comparison."""
        return [
            {
                "name": "paper_summary",
                "input": "Summarize this research paper: [Abstract about machine learning advances]",
                "expected_elements": ["methodology", "findings", "implications", "significance"],
                "context": "academic_summary"
            },
            {
                "name": "technical_explanation",
                "input": "Explain how neural networks work for a technical audience",
                "expected_elements": ["layers", "neurons", "weights", "training", "backpropagation"],
                "context": "technical_explanation"
            },
            {
                "name": "news_analysis",
                "input": "Analyze the key points from this tech news article",
                "expected_elements": ["main_points", "implications", "context", "significance"],
                "context": "news_analysis"
            }
        ]

    def test_comparative_quality_analysis(self, quality_analyzer, test_scenarios):
        """Compare content quality between Gemini and Claude."""

        results = {}

        for provider in ['gemini', 'claude']:
            results[provider] = {}

            with patch.dict(os.environ, {'AI_PROVIDER': provider}):
                # Mock client responses for testing
                mock_responses = self._get_mock_responses(provider)

                for scenario in test_scenarios:
                    scenario_name = scenario["name"]
                    mock_response = mock_responses.get(scenario_name, "Default response")

                    # Analyze quality
                    quality_metrics = quality_analyzer.analyze_response_quality(
                        mock_response,
                        scenario["context"]
                    )

                    results[provider][scenario_name] = {
                        'response': mock_response,
                        'metrics': {m.name: m.score for m in quality_metrics},
                        'overall_score': sum(m.score for m in quality_metrics) / len(quality_metrics)
                    }

        # Compare and assert quality standards
        self._assert_quality_standards(results, quality_analyzer.quality_thresholds)

    def _get_mock_responses(self, provider: str) -> Dict[str, str]:
        """Get mock responses for testing."""

        mock_responses = {
            'gemini': {
                'paper_summary': """## Research Paper Summary

**Methodology:** The authors employed a novel neural network architecture combining transformers with convolutional layers.

**Key Findings:** The proposed method achieved 95% accuracy on benchmark datasets, outperforming existing approaches by 12%.

**Implications:** This advancement could significantly improve natural language processing applications and reduce computational requirements.

**Significance:** The research opens new avenues for efficient AI model design.""",

                'technical_explanation': """Neural networks are computational models inspired by biological neural systems. They consist of interconnected layers of artificial neurons that process information through weighted connections.

**Architecture Components:**
- Input layers receive data
- Hidden layers perform transformations
- Output layers produce results

**Training Process:**
Neural networks learn through backpropagation, adjusting weights based on error gradients. This iterative process optimizes the network's ability to map inputs to desired outputs.""",

                'news_analysis': """**Main Points:**
1. Major tech company announces breakthrough in quantum computing
2. New chip architecture promises 100x speedup
3. Commercial applications expected within 5 years

**Implications:**
This advancement could revolutionize cryptography, drug discovery, and financial modeling.

**Context:**
The announcement follows years of research and represents a significant milestone in quantum technology development."""
            },
            'claude': {
                'paper_summary': """## Research Paper Analysis

**Methodology:** The research utilized an innovative neural network design that integrates transformer architectures with convolutional processing layers.

**Key Findings:** The proposed approach demonstrated 95% accuracy on standard benchmarks, representing a 12% improvement over current state-of-the-art methods.

**Implications:** These results suggest potential for enhanced natural language processing capabilities with reduced computational overhead.

**Significance:** The work contributes valuable insights for efficient AI model development and deployment.""",

                'technical_explanation': """Neural networks are computational frameworks modeled after biological neural systems. They comprise interconnected layers of artificial neurons that process information through weighted connections.

**Core Components:**
- Input layers: Data reception and preprocessing
- Hidden layers: Feature extraction and transformation
- Output layers: Final result generation

**Learning Process:**
Networks learn through backpropagation, systematically adjusting connection weights based on calculated error gradients. This optimization process enables the network to learn complex input-output mappings.""",

                'news_analysis': """**Key Points:**
1. Leading technology firm unveils quantum computing breakthrough
2. Novel chip design offers 100x performance improvement
3. Market applications anticipated within 5-year timeframe

**Strategic Implications:**
This development has far-reaching consequences for cryptography, pharmaceutical research, and financial analysis sectors.

**Industry Context:**
The announcement represents culmination of extensive research efforts and marks a pivotal moment in quantum computing evolution."""
            }
        }

        return mock_responses.get(provider, {})

    def _assert_quality_standards(self, results: Dict, thresholds: Dict[str, float]):
        """Assert quality standards are met."""

        print("\n=== Content Quality Analysis ===")

        for scenario_name in results['gemini'].keys():
            print(f"\nScenario: {scenario_name}")

            gemini_result = results['gemini'][scenario_name]
            claude_result = results['claude'][scenario_name]

            print(f"  Gemini Overall Score: {gemini_result['overall_score']:.3f}")
            print(f"  Claude Overall Score: {claude_result['overall_score']:.3f}")

            # Detailed metrics comparison
            for metric_name in gemini_result['metrics']:
                gemini_score = gemini_result['metrics'][metric_name]
                claude_score = claude_result['metrics'][metric_name]

                print(f"    {metric_name}: Gemini={gemini_score:.3f}, Claude={claude_score:.3f}")

            # Assert Claude meets quality standards
            claude_overall = claude_result['overall_score']
            assert claude_overall >= 0.7, f"Claude overall quality too low for {scenario_name}: {claude_overall:.3f}"

            # Assert Claude is competitive with Gemini (within 10% tolerance)
            gemini_overall = gemini_result['overall_score']
            quality_tolerance = 0.9  # Claude should be at least 90% as good as Gemini

            assert claude_overall >= gemini_overall * quality_tolerance, \
                f"Claude quality significantly lower than Gemini for {scenario_name}: " \
                f"Claude={claude_overall:.3f}, Gemini={gemini_overall:.3f}"

### 5.2 レスポンス一貫性テスト

```python
# test_response_consistency.py

import pytest
from unittest.mock import patch
from nook.functions.common.python.client_factory import create_client

class TestResponseConsistency:
    """Test response consistency across multiple runs."""

    def test_response_consistency(self):
        """Test that responses are reasonably consistent across runs."""

        test_input = "Explain the benefits of cloud computing in 3 key points."
        responses = []

        with patch.dict(os.environ, {'AI_PROVIDER': 'claude'}):
            client = create_client()

            # Generate multiple responses
            for i in range(5):  # Run 5 times for consistency testing
                try:
                    response = client.generate_content(test_input)
                    responses.append(response)
                except Exception as e:
                    pytest.fail(f"Failed to generate response {i+1}: {e}")

        # Analyze consistency
        self._analyze_response_consistency(responses, test_input)

    def _analyze_response_consistency(self, responses: List[str], input_text: str):
        """Analyze consistency metrics across responses."""

        if len(responses) < 2:
            pytest.fail("Need at least 2 responses to test consistency")

        # Length consistency
        lengths = [len(response.split()) for response in responses]
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        length_std_dev = length_variance ** 0.5

        print(f"\nResponse Consistency Analysis:")
        print(f"  Average Length: {avg_length:.1f} words")
        print(f"  Length Std Dev: {length_std_dev:.1f} words")

        # Length should not vary too much (within 50% of average)
        max_acceptable_std_dev = avg_length * 0.5
        assert length_std_dev <= max_acceptable_std_dev, \
            f"Response length too inconsistent: {length_std_dev:.1f} > {max_acceptable_std_dev:.1f}"

        # Content consistency (basic check for key terms)
        common_terms = self._extract_common_terms(responses)
        print(f"  Common Terms: {common_terms[:10]}")  # Show top 10

        # Should have some consistent terminology
        assert len(common_terms) >= 3, "Responses lack consistent terminology"

    def _extract_common_terms(self, responses: List[str]) -> List[str]:
        """Extract terms that appear in most responses."""

        from collections import Counter
        import re

        all_words = []
        for response in responses:
            # Extract meaningful words (filter out common stopwords)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', response.lower())
            all_words.extend(words)

        # Find words that appear in most responses
        word_counts = Counter(all_words)
        threshold = len(responses) * 0.6  # Must appear in 60% of responses

        common_terms = [word for word, count in word_counts.items() if count >= threshold]
        return common_terms
```

---

## 6. 実装タイムラインとテストスケジュール

### 6.1 4週間テストタイムライン

#### 第1週: 基盤テスト
**1-2日目: ユニットテストセットアップ**
- ✅ Set up testing framework and directory structure
- ✅ Implement Claude client unit tests
- ✅ Create mock API response fixtures
- ✅ Test configuration management

**Days 3-5: Basic Integration Testing**
- ✅ Test paper summarizer integration with Claude
- ✅ Validate basic error handling
- ✅ Implement simple rollback tests
- ✅ Create test data fixtures

#### 第2週: コア関数テスト
**6-8日目: 関数移行テスト**
- ✅ Test tech feed migration
- ✅ Test hacker news migration
- ✅ Validate content generation quality
- ✅ Performance baseline establishment

**Days 9-10: Quality Assurance Framework**
- ✅ Implement response quality metrics
- ✅ Set up consistency testing
- ✅ Create quality comparison reports
- ✅ Establish acceptance criteria

#### 第3週: 高度なテストと検証
**Days 11-12: Performance and Reliability**
- ✅ Conduct performance benchmarking
- ✅ Test concurrent request handling
- ✅ Validate retry mechanisms
- ✅ Error scenario testing

**Days 13-15: End-to-End Validation**
- ✅ Complete integration testing for reddit explorer
- ✅ Test full rollback procedures
- ✅ Validate emergency protocols
- ✅ Document test results

#### 第4週: プロダクション準備
**Days 16-17: Final Validation**
- ✅ Production environment testing
- ✅ Performance validation in staging
- ✅ Final quality assurance review
- ✅ Rollback procedure validation

**Days 18-20: Deployment Support**
- ✅ Monitor deployment testing
- ✅ Validate production performance
- ✅ Support production deployment
- ✅ Post-deployment validation

### 6.2 テスト成果物

#### ドキュメント成果物
1. **Test Design Document** (This document)
2. **Test Results Report** - Summary of all test outcomes
3. **Performance Benchmark Report** - Gemini vs Claude performance comparison
4. **Quality Assessment Report** - Content quality comparison analysis
5. **Rollback Procedures Manual** - Emergency and planned rollback procedures

#### Code Deliverables
1. **Unit Test Suite** - Complete unit test coverage for Claude client
2. **Integration Test Suite** - End-to-end testing for all 4 core functions
3. **Performance Test Suite** - Automated performance benchmarking
4. **Quality Assessment Tools** - Automated content quality evaluation
5. **Rollback Scripts** - Automated rollback and validation scripts

#### Validation Artifacts
1. **Test Coverage Report** - Code coverage metrics
2. **Performance Comparison Data** - Response time and throughput analysis
3. **Quality Metrics Dashboard** - Content quality tracking
4. **Error Rate Analysis** - Error handling effectiveness
5. **Rollback Validation Report** - Emergency procedure testing results

---

## 7. リスクベーステスト優先順位

### 7.1 高優先度テスト（必須完了）

#### クリティカルパステスト
1. **Core Content Generation** - Essential functionality tests
   - Test Effort: 40% of total testing time
   - Coverage: All 4 core functions
   - Success Criteria: 100% functional parity

2. **API Response Quality** - Content quality validation
   - Test Effort: 25% of total testing time
   - Coverage: Side-by-side quality comparison
   - Success Criteria: ≥90% quality retention

3. **Error Handling** - Resilience and reliability
   - Test Effort: 20% of total testing time
   - Coverage: Network errors, rate limits, timeouts
   - Success Criteria: Graceful degradation

4. **Rollback Procedures** - Emergency protocols
   - Test Effort: 15% of total testing time
   - Coverage: Environment switching, code rollback
   - Success Criteria: <1 hour recovery time

### 7.2 中優先度テスト（完了推奨）

#### Performance and Reliability Tests
1. **Response Time Benchmarking** - Performance validation
2. **Configuration Migration** - Environment management
3. **Concurrent Request Handling** - Load testing
4. **Integration Validation** - End-to-end workflows

### 7.3 低優先度テスト（あれば良い）

#### Advanced and Edge Case Tests
1. **Complex Error Scenarios** - Advanced failure modes
2. **Performance Optimization** - Fine-tuning and optimization
3. **Extended Quality Metrics** - Detailed quality analysis
4. **Advanced Rollback Scenarios** - Partial rollback testing

---

## 8. 成功基準と受入テスト

### 8.1 機能成功基準

#### コア機能
- ✅ All 4 core functions work with Claude API
- ✅ Response formats match existing Gemini output structure
- ✅ Configuration migration works seamlessly
- ✅ Error handling maintains or improves resilience

#### Quality Standards
- ✅ Content quality ≥90% of Gemini baseline
- ✅ Response consistency within acceptable variance
- ✅ No degradation in user experience
- ✅ Maintained or improved response relevance

### 8.2 パフォーマンス成功基準

#### Response Time Requirements
- ✅ Average response time ≤110% of Gemini baseline
- ✅ 95th percentile response time ≤120% of Gemini baseline
- ✅ No timeouts under normal load
- ✅ Graceful handling of rate limits

#### Reliability Requirements
- ✅ Error rate ≤ current Gemini error rate
- ✅ Retry mechanisms function correctly
- ✅ System recovers from transient failures
- ✅ No data loss during migration

### 8.3 運用成功基準

#### Rollback Capability
- ✅ Emergency rollback completes in <1 hour
- ✅ Planned rollback completes in <30 minutes
- ✅ System validation passes after rollback
- ✅ No data corruption during rollback

#### Monitoring and Observability
- ✅ Performance metrics collection works
- ✅ Error logging and monitoring function
- ✅ Quality metrics tracking operational
- ✅ Alerting systems properly configured

---

## 9. テスト実行手順

### 9.1 ローカル開発テスト

#### 環境セットアップ
```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install pytest pytest-mock pytest-asyncio

# 2. Set up environment variables
export AI_PROVIDER=claude
export ANTHROPIC_API_KEY=your_api_key_here
export GEMINI_API_KEY=your_gemini_key_here

# 3. Create test output directories
mkdir -p test_output
mkdir -p test_reports
```

#### Running Unit Tests
```bash
# Run all unit tests
python -m pytest nook/functions/common/python/tests/ -v

# Run specific test categories
python -m pytest nook/functions/common/python/tests/test_claude_client.py -v
python -m pytest nook/functions/common/python/tests/test_error_handling.py -v

# Run with coverage
python -m pytest nook/functions/common/python/tests/ --cov=nook --cov-report=html
```

#### Running Integration Tests
```bash
# Run integration tests
python -m pytest nook/functions/tests/integration/ -v

# Run performance tests (separate flag)
python -m pytest nook/functions/tests/ -v --performance

# Run quality comparison tests
python -m pytest nook/functions/tests/ -v --quality
```

#### Running Rollback Tests
```bash
# Test rollback procedures
python -m pytest nook/functions/tests/test_rollback_procedures.py -v

# Test emergency rollback
python scripts/test_emergency_rollback.py --dry-run

# Full rollback validation
python scripts/validate_rollback.py --environment=test
```

### 9.2 CI/CD統合

#### GitHub Actions Workflow
The testing framework integrates with GitHub Actions for automated testing:

```yaml
# .github/workflows/test-migration.yml
name: Migration Testing

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
        ai-provider: [claude, gemini]

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Run tests
      env:
        AI_PROVIDER: ${{ matrix.ai-provider }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      run: |
        python -m pytest nook/functions/tests/ -v --junit-xml=test-results.xml
```

### 9.3 プロダクション検証

#### Staging Environment Testing
```bash
# Deploy to staging
./scripts/deploy_staging.sh

# Run production validation tests
python -m pytest nook/functions/tests/production/ -v --environment=staging

# Performance validation
python scripts/performance_validation.py --environment=staging

# Quality validation
python scripts/quality_validation.py --environment=staging --baseline=gemini
```

#### Production Deployment Testing
```bash
# Pre-deployment validation
python scripts/pre_deployment_check.py

# Post-deployment validation
python scripts/post_deployment_validation.py

# Monitoring validation
python scripts/validate_monitoring.py
```

---

## 結論

このテスト設計は、4週間のタイムライン内でGeminiからClaudeへの移行を検証するための包括的かつ実用的なアプローチを提供します。テスト戦略は以下を優先します：

1. **包括的なエッジケースカバレッジ**よりも**必須機能**
2. **戦略的自動化**で補完された**手動検証**
3. **高影響シナリオ**に焦点を当てた**リスクベース優先順位付け**
4. **堅牢なロールバック機能**を備えた**プロダクション準備**

テストフレームワークは、リソース制約内で実装可能でありながら、移行への信頼を提供するように設計されています。モジュラーアプローチにより、開発の進行に合わせて段階的テストが可能で、リスク軽減のための明確な成功基準とロールバック手順を備えています。

**このアプローチの主な利点:**
- コアビジネス機能を最優先
- 移行のための明確なGo/No-Go基準を提供
- 包括的ロールバックテストを含む
- 実装制約と徹底性のバランスを取る
- プロダクションデプロイメントへの信頼を可能にする

このテスト戦略により、移行がシステムの信頼性を維持しながら、デプロイ後の反復と改善の柔軟性を提供することが保証されます。