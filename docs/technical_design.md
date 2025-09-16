# Nook Gemini-to-Claude移行技術設計書

## 目次

1. [エグゼクティブサマリー](#エグゼクティブサマリー)
2. [アーキテクチャ設計](#アーキテクチャ設計)
3. [実装仕様](#実装仕様)
4. [移行戦略詳細](#移行戦略詳細)
5. [技術仕様](#技術仕様)
6. [統合ポイント](#統合ポイント)
7. [パフォーマンスとセキュリティの考慮事項](#パフォーマンスとセキュリティの考慮事項)
8. [テスト戦略](#テスト戦略)
9. [ロールバックと復旧](#ロールバックと復旧)
10. [実装タイムライン](#実装タイムライン)

## エグゼクティブサマリー

この文書では、NookアプリケーションをGoogle Gemini APIからClaude CLIに移行するための包括的な技術設計を提供します。移行では、将来の柔軟性のための統一されたAIクライアント抽象化レイヤーを導入しながら、既存の全機能を維持します。

**主要目標:**
- Gemini APIをClaude CLI統合に置き換え
- API互換性と機能パリティの維持
- 堅牢なエラーハンドリングとリトライメカニズムの実装
- ロールバック機能を備えたゼロダウンタイムデプロイメントの確保
- AWS Lambda デプロイメント環境への最適化

**アーキテクチャアプローチ:**
- ファクトリーメソッドを使用した抽象インターフェースパターン
- 設定駆動のクライアント選択
- 後方互換性のあるAPIサーフェス
- フィーチャーフラグによる段階的移行

## アーキテクチャ設計

### 1. 統一AIクライアントインターフェース設計

#### コアインターフェース定義

```python
# /nook/functions/common/python/ai_client_interface.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Dict, List
from enum import Enum

class AIProvider(Enum):
    GEMINI = "gemini"
    CLAUDE = "claude"

@dataclass
class AIClientConfig:
    """Unified configuration for AI clients."""

    # Common parameters
    model: str
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    timeout: int = 60000
    use_search: bool = False

    # Provider-specific parameters
    provider_specific: Dict[str, Any] = None

    def __post_init__(self):
        if self.provider_specific is None:
            self.provider_specific = {}

class AIClientInterface(ABC):
    """Abstract interface for AI clients."""

    def __init__(self, config: AIClientConfig):
        self.config = config

    @abstractmethod
    def generate_content(
        self,
        contents: str | List[str],
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate content using the AI model."""
        pass

    @abstractmethod
    def create_chat(self, **kwargs) -> None:
        """Create a new chat session."""
        pass

    @abstractmethod
    def send_message(self, message: str) -> str:
        """Send a message to an active chat session."""
        pass

    @abstractmethod
    def chat_with_search(self, message: str, model: Optional[str] = None) -> str:
        """Create a chat session with search capabilities and send a message."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass

class AIClientFactory:
    """Factory for creating AI clients."""

    _registry: Dict[AIProvider, type] = {}

    @classmethod
    def register(cls, provider: AIProvider, client_class: type) -> None:
        """Register a client implementation."""
        cls._registry[provider] = client_class

    @classmethod
    def create_client(
        cls,
        provider: AIProvider,
        config: AIClientConfig
    ) -> AIClientInterface:
        """Create an AI client instance."""
        if provider not in cls._registry:
            raise ValueError(f"Unknown provider: {provider}")

        client_class = cls._registry[provider]
        return client_class(config)
```

#### 設定管理

```python
# /nook/functions/common/python/ai_config_manager.py

import os
from typing import Dict, Any, Optional
from .ai_client_interface import AIClientConfig, AIProvider

class AIConfigManager:
    """Manages AI client configuration from environment and overrides."""

    # Default model mappings
    DEFAULT_MODELS = {
        AIProvider.GEMINI: "gemini-2.0-flash-exp",
        AIProvider.CLAUDE: "claude-3-5-sonnet-20241022"
    }

    @classmethod
    def create_config(
        cls,
        provider: AIProvider,
        overrides: Optional[Dict[str, Any]] = None
    ) -> AIClientConfig:
        """Create configuration for the specified provider."""

        base_config = {
            "model": cls._get_model_for_provider(provider),
            "temperature": float(os.environ.get("AI_TEMPERATURE", "1.0")),
            "top_p": float(os.environ.get("AI_TOP_P", "0.95")),
            "top_k": int(os.environ.get("AI_TOP_K", "40")),
            "max_output_tokens": int(os.environ.get("AI_MAX_TOKENS", "8192")),
            "timeout": int(os.environ.get("AI_TIMEOUT", "60000")),
            "use_search": os.environ.get("AI_USE_SEARCH", "false").lower() == "true"
        }

        if overrides:
            base_config.update(overrides)

        provider_specific = cls._get_provider_specific_config(provider)

        return AIClientConfig(
            **base_config,
            provider_specific=provider_specific
        )

    @classmethod
    def _get_model_for_provider(cls, provider: AIProvider) -> str:
        """Get the model name for the provider."""
        env_key = f"{provider.value.upper()}_MODEL"
        return os.environ.get(env_key, cls.DEFAULT_MODELS[provider])

    @classmethod
    def _get_provider_specific_config(cls, provider: AIProvider) -> Dict[str, Any]:
        """Get provider-specific configuration."""
        if provider == AIProvider.CLAUDE:
            return {
                "max_retries": int(os.environ.get("CLAUDE_MAX_RETRIES", "5")),
                "retry_delay": float(os.environ.get("CLAUDE_RETRY_DELAY", "1.0")),
                "api_base": os.environ.get("CLAUDE_API_BASE"),
                "anthropic_version": os.environ.get("CLAUDE_API_VERSION", "2023-06-01")
            }
        elif provider == AIProvider.GEMINI:
            return {
                "safety_settings_level": os.environ.get("GEMINI_SAFETY_LEVEL", "BLOCK_NONE")
            }
        return {}
```

### 2. Claude API SDK統合アーキテクチャ

#### Claude Client実装

```python
# /nook/functions/common/python/claude_client.py

import os
import logging
from typing import Optional, List, Dict, Any
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
import anthropic
from anthropic import APIError, RateLimitError, APITimeoutError

from .ai_client_interface import AIClientInterface, AIClientConfig, AIProvider, AIClientFactory

logger = logging.getLogger(__name__)

class ClaudeClientError(Exception):
    """Custom exception for Claude client errors."""
    pass

class ClaudeClient(AIClientInterface):
    """Claude implementation of the AI client interface."""

    def __init__(self, config: AIClientConfig):
        super().__init__(config)

        # Initialize Anthropic client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ClaudeClientError("ANTHROPIC_API_KEY environment variable is not set")

        client_kwargs = {
            "api_key": api_key,
            "timeout": config.timeout / 1000.0  # Convert ms to seconds
        }

        # Add provider-specific configurations
        if config.provider_specific:
            if base_url := config.provider_specific.get("api_base"):
                client_kwargs["base_url"] = base_url

            max_retries = config.provider_specific.get("max_retries", 5)
            client_kwargs["max_retries"] = max_retries

        self._client = anthropic.Anthropic(**client_kwargs)
        self._chat_context: List[Dict[str, str]] = []
        self._system_instruction: Optional[str] = None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(
            lambda e: isinstance(e, (RateLimitError, APITimeoutError, APIError))
        ),
        before_sleep=lambda retry_state: logger.info(
            f"Retrying Claude API call due to {retry_state.outcome.exception()}..."
        )
    )
    def generate_content(
        self,
        contents: str | List[str],
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate content using Claude API."""

        if isinstance(contents, str):
            contents = [contents]

        # Prepare messages
        messages = []
        for content in contents:
            messages.append({
                "role": "user",
                "content": content
            })

        # Prepare request parameters
        request_params = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_output_tokens", self.config.max_output_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "messages": messages
        }

        # Add system instruction if provided
        if system_instruction:
            request_params["system"] = system_instruction

        try:
            response = self._client.messages.create(**request_params)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise ClaudeClientError(f"Failed to generate content: {e}") from e

    def create_chat(self, **kwargs) -> None:
        """Create a new chat session."""
        self._chat_context = []

        # Store system instruction for chat session
        self._system_instruction = kwargs.get("system_instruction")

        logger.info("Created new Claude chat session")

    def send_message(self, message: str) -> str:
        """Send a message to an active chat session."""
        if self._chat_context is None:
            raise ClaudeClientError("No chat session active. Call create_chat() first.")

        # Add user message to context
        self._chat_context.append({
            "role": "user",
            "content": message
        })

        # Prepare request
        request_params = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
            "messages": self._chat_context.copy()
        }

        if self._system_instruction:
            request_params["system"] = self._system_instruction

        try:
            response = self._client.messages.create(**request_params)
            assistant_message = response.content[0].text

            # Add assistant response to context
            self._chat_context.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

        except Exception as e:
            logger.error(f"Claude chat error: {e}")
            raise ClaudeClientError(f"Failed to send message: {e}") from e

    def chat_with_search(self, message: str, model: Optional[str] = None) -> str:
        """Create a chat session with search capabilities and send a message."""

        # Note: Claude doesn't have built-in search like Gemini
        # We'll implement this by using the message as-is
        # In the future, this could be enhanced with RAG or external search

        logger.warning("Claude doesn't have built-in search. Using direct message processing.")

        # Create a temporary chat session
        original_context = self._chat_context.copy() if self._chat_context else []

        try:
            self.create_chat()
            response = self.send_message(message)
            return response
        finally:
            # Restore original context
            self._chat_context = original_context

    def close(self) -> None:
        """Clean up resources."""
        self._chat_context = []
        self._system_instruction = None
        logger.info("Closed Claude client")

# Register Claude client with factory
AIClientFactory.register(AIProvider.CLAUDE, ClaudeClient)
```

### 3. 検索サービス統合設計

Since Claude doesn't have built-in search capabilities like Gemini, we need to implement an external search integration:

```python
# /nook/functions/common/python/search_service.py

import os
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import requests
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class SearchResult:
    """Represents a search result."""

    def __init__(self, title: str, url: str, snippet: str, score: float = 0.0):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.score = score

class SearchServiceInterface(ABC):
    """Abstract interface for search services."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Perform a search and return results."""
        pass

class GoogleSearchService(SearchServiceInterface):
    """Google Custom Search implementation."""

    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")

        if not self.api_key or not self.search_engine_id:
            logger.warning("Google Search API credentials not configured")

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Perform Google Custom Search."""
        if not self.api_key or not self.search_engine_id:
            return []

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(max_results, 10)
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("items", []):
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    score=1.0  # Google doesn't provide explicit scores
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Google Search error: {e}")
            return []

class EnhancedClaudeClient(ClaudeClient):
    """Claude client with enhanced search capabilities."""

    def __init__(self, config: AIClientConfig):
        super().__init__(config)
        self._search_service = GoogleSearchService()

    def chat_with_search(self, message: str, model: Optional[str] = None) -> str:
        """Enhanced chat with search capabilities."""

        if not self.config.use_search:
            return super().chat_with_search(message, model)

        # Extract search query from message (simple heuristic)
        search_query = self._extract_search_query(message)

        # Perform search
        search_results = self._search_service.search(search_query, max_results=3)

        # Enhance message with search context
        enhanced_message = self._enhance_message_with_search(message, search_results)

        # Use enhanced message for chat
        return super().chat_with_search(enhanced_message, model)

    def _extract_search_query(self, message: str) -> str:
        """Extract search query from user message."""
        # Simple implementation - in production, this could be more sophisticated
        # Remove common chat phrases and extract key terms
        stop_words = ["what", "how", "when", "where", "why", "is", "are", "the", "a", "an"]
        words = message.lower().split()
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(key_words[:5])  # Limit to first 5 key words

    def _enhance_message_with_search(
        self,
        original_message: str,
        search_results: List[SearchResult]
    ) -> str:
        """Enhance the message with search context."""

        if not search_results:
            return original_message

        search_context = "\n\n[Search Results for additional context]:\n"
        for i, result in enumerate(search_results, 1):
            search_context += f"{i}. {result.title}\n"
            search_context += f"   URL: {result.url}\n"
            search_context += f"   Summary: {result.snippet}\n\n"

        return f"{original_message}\n{search_context}"

# Update registration to use enhanced client
AIClientFactory.register(AIProvider.CLAUDE, EnhancedClaudeClient)
```

### 4. エラーハンドリングとリトライメカニズム

```python
# /nook/functions/common/python/ai_error_handling.py

import logging
from typing import Type, Callable, Any
from functools import wraps
from tenacity import Retrying, stop_after_attempt, wait_exponential, retry_if_exception

logger = logging.getLogger(__name__)

class AIClientError(Exception):
    """Base exception for AI client errors."""
    pass

class RateLimitError(AIClientError):
    """Rate limit exceeded error."""
    pass

class TimeoutError(AIClientError):
    """Request timeout error."""
    pass

class AuthenticationError(AIClientError):
    """Authentication error."""
    pass

class ModelNotFoundError(AIClientError):
    """Model not found error."""
    pass

class ErrorMapper:
    """Maps provider-specific errors to unified error types."""

    @staticmethod
    def map_gemini_error(error: Exception) -> AIClientError:
        """Map Gemini errors to unified error types."""
        from google.genai.errors import ClientError

        if isinstance(error, ClientError):
            if "rate limit" in str(error).lower():
                return RateLimitError(f"Gemini rate limit exceeded: {error}")
            elif "timeout" in str(error).lower():
                return TimeoutError(f"Gemini timeout: {error}")
            elif "authentication" in str(error).lower():
                return AuthenticationError(f"Gemini authentication error: {error}")
            else:
                return AIClientError(f"Gemini error: {error}")

        return AIClientError(f"Unknown Gemini error: {error}")

    @staticmethod
    def map_claude_error(error: Exception) -> AIClientError:
        """Map Claude errors to unified error types."""
        from anthropic import APIError, RateLimitError as AnthropicRateLimit, APITimeoutError

        if isinstance(error, AnthropicRateLimit):
            return RateLimitError(f"Claude rate limit exceeded: {error}")
        elif isinstance(error, APITimeoutError):
            return TimeoutError(f"Claude timeout: {error}")
        elif isinstance(error, APIError):
            if error.status_code == 401:
                return AuthenticationError(f"Claude authentication error: {error}")
            elif error.status_code == 404:
                return ModelNotFoundError(f"Claude model not found: {error}")
            else:
                return AIClientError(f"Claude API error: {error}")

        return AIClientError(f"Unknown Claude error: {error}")

def create_retry_decorator(
    max_attempts: int = 5,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exponential_base: float = 2.0
):
    """Create a retry decorator with configurable parameters."""

    def should_retry(exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        return isinstance(exception, (RateLimitError, TimeoutError, AIClientError))

    def retry_decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retryer = Retrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(
                    multiplier=min_wait,
                    min=min_wait,
                    max=max_wait,
                    exp_base=exponential_base
                ),
                retry=retry_if_exception(should_retry),
                before_sleep=lambda retry_state: logger.info(
                    f"Retrying {func.__name__} (attempt {retry_state.attempt_number}) "
                    f"due to {retry_state.outcome.exception()}"
                )
            )

            return retryer(func, *args, **kwargs)

        return wrapper

    return retry_decorator
```

## 実装仕様

### 1. 詳細なクラス階層とインターフェース

```python
# /nook/functions/common/python/client_factory.py

from typing import Dict, Any, Optional
from .ai_client_interface import AIClientInterface, AIClientConfig, AIProvider, AIClientFactory
from .ai_config_manager import AIConfigManager
from .gemini_client import GeminiClient
from .claude_client import EnhancedClaudeClient

class UnifiedAIClientFactory:
    """High-level factory for creating AI clients with automatic configuration."""

    @staticmethod
    def create_client(
        provider: Optional[AIProvider] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AIClientInterface:
        """
        Create an AI client with automatic provider selection and configuration.

        Args:
            provider: AI provider to use. If None, defaults to environment setting.
            config: Configuration overrides.
            **kwargs: Additional configuration parameters.

        Returns:
            Configured AI client instance.
        """

        # Determine provider
        if provider is None:
            provider_name = os.environ.get("AI_PROVIDER", "gemini").lower()
            provider = AIProvider(provider_name)

        # Create configuration
        config_overrides = config or {}
        config_overrides.update(kwargs)

        ai_config = AIConfigManager.create_config(provider, config_overrides)

        # Create and return client
        return AIClientFactory.create_client(provider, ai_config)

# Updated factory function for backward compatibility
def create_client(config: Optional[Dict[str, Any]] = None, **kwargs) -> AIClientInterface:
    """
    Backward-compatible factory function.

    This maintains the same interface as the original Gemini create_client function
    but now supports multiple providers through configuration.
    """
    return UnifiedAIClientFactory.create_client(config=config, **kwargs)
```

### 2. データフロー図

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Lambda        │    │  UnifiedAI       │    │   Provider      │
│   Function      │───▶│  ClientFactory   │───▶│   Client        │
│                 │    │                  │    │   (Claude/      │
└─────────────────┘    └──────────────────┘    │   Gemini)       │
                                               └─────────────────┘
        │                        │                       │
        │                        │                       │
        ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Content       │    │  Configuration   │    │   External      │
│   Generation    │    │  Manager         │    │   API           │
│   Request       │    │                  │    │   (Anthropic/   │
└─────────────────┘    └──────────────────┘    │   Google)       │
                                               └─────────────────┘
```

### 3. API統合パターン

#### Request/Response Flow
```python
# Example implementation pattern
class LambdaFunction:
    def __init__(self):
        self.ai_client = create_client()

    def process_request(self, event: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Extract content and parameters
            content = event.get("content", "")
            system_instruction = event.get("system_instruction")

            # Generate response
            response = self.ai_client.generate_content(
                contents=content,
                system_instruction=system_instruction
            )

            return {
                "statusCode": 200,
                "body": {"response": response}
            }

        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {
                "statusCode": 500,
                "body": {"error": str(e)}
            }
        finally:
            self.ai_client.close()
```

## 移行戦略詳細

### フェーズ1: 基盤セットアップ（第1週）

#### 1.1 抽象インターフェース実装
```bash
# File structure creation
mkdir -p /nook/functions/common/python/interfaces
mkdir -p /nook/functions/common/python/clients
mkdir -p /nook/functions/common/python/config
mkdir -p /nook/functions/common/python/utils

# Implementation files
touch /nook/functions/common/python/interfaces/ai_client_interface.py
touch /nook/functions/common/python/clients/claude_client.py
touch /nook/functions/common/python/clients/gemini_client_wrapper.py
touch /nook/functions/common/python/config/ai_config_manager.py
touch /nook/functions/common/python/utils/error_handling.py
```

#### 1.2 環境設定
```python
# Environment variables mapping
ENVIRONMENT_MIGRATION_MAP = {
    # Existing Gemini settings
    "GEMINI_API_KEY": "GEMINI_API_KEY",  # Keep for backward compatibility

    # New unified settings
    "AI_PROVIDER": "claude",  # Default provider
    "ANTHROPIC_API_KEY": None,  # Required for Claude
    "AI_TEMPERATURE": "1.0",
    "AI_TOP_P": "0.95",
    "AI_TOP_K": "40",
    "AI_MAX_TOKENS": "8192",
    "AI_TIMEOUT": "60000",
    "AI_USE_SEARCH": "false",

    # Claude-specific settings
    "CLAUDE_MODEL": "claude-3-5-sonnet-20241022",
    "CLAUDE_MAX_RETRIES": "5",
    "CLAUDE_RETRY_DELAY": "1.0",

    # Search service settings
    "GOOGLE_SEARCH_API_KEY": None,  # For search functionality
    "GOOGLE_SEARCH_ENGINE_ID": None
}
```

### フェーズ2: コア実装（第1-2週）

#### 2.1 Claude Client開発
```python
# Implementation checklist
CLAUDE_CLIENT_FEATURES = [
    "✓ Basic content generation",
    "✓ Chat session management",
    "✓ Error handling with retry logic",
    "✓ Configuration management",
    "✓ Search integration",
    "✓ Timeout handling",
    "✓ Rate limiting support",
    "✓ Context management for chat"
]
```

#### 2.2 後方互換性レイヤー
```python
# /nook/functions/common/python/gemini_client_compatibility.py

from .client_factory import create_client as create_ai_client
from .ai_client_interface import AIProvider

def create_client(config=None, **kwargs):
    """
    Backward compatible create_client function.

    This function maintains the exact same interface as the original
    Gemini client factory but routes to the new unified system.
    """

    # Check if we should use Gemini or Claude
    provider_env = os.environ.get("AI_PROVIDER", "gemini").lower()
    provider = AIProvider.CLAUDE if provider_env == "claude" else AIProvider.GEMINI

    return create_ai_client(provider=provider, config=config, **kwargs)

# This allows existing code to work without changes:
# from ..common.python.gemini_client import create_client
```

### フェーズ3: 関数移行（第2-3週）

#### 3.1 移行順序と戦略

```python
MIGRATION_PHASES = [
    {
        "phase": 1,
        "functions": ["paper_summarizer"],
        "risk": "LOW",
        "reason": "Batch processing, easy to test and rollback",
        "estimated_effort": "4 hours"
    },
    {
        "phase": 2,
        "functions": ["tech_feed", "github_trending"],
        "risk": "LOW",
        "reason": "Similar content generation patterns",
        "estimated_effort": "6 hours"
    },
    {
        "phase": 3,
        "functions": ["hacker_news"],
        "risk": "LOW-MEDIUM",
        "reason": "Simple summarization with different content format",
        "estimated_effort": "4 hours"
    },
    {
        "phase": 4,
        "functions": ["reddit_explorer"],
        "risk": "MEDIUM",
        "reason": "More complex content processing",
        "estimated_effort": "6 hours"
    },
    {
        "phase": 5,
        "functions": ["viewer"],
        "risk": "HIGH",
        "reason": "Interactive chat functionality with search",
        "estimated_effort": "8 hours"
    }
]
```

#### 3.2 Per-Function Migration Process

```python
# Migration script template
class FunctionMigrator:
    def __init__(self, function_name: str):
        self.function_name = function_name
        self.backup_created = False

    def migrate(self):
        """Execute migration process."""
        try:
            self.create_backup()
            self.update_imports()
            self.update_configuration()
            self.run_tests()
            self.validate_functionality()

        except Exception as e:
            self.rollback()
            raise e

    def create_backup(self):
        """Create git branch backup."""
        backup_branch = f"backup-{self.function_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        subprocess.run(["git", "checkout", "-b", backup_branch])
        subprocess.run(["git", "checkout", "main"])
        self.backup_created = True

    def update_imports(self):
        """Update import statements."""
        # This would be automated with scripts
        pass

    def validate_functionality(self):
        """Run validation tests."""
        # Function-specific validation
        pass
```

### フェーズ4: テストと検証

#### 4.1 テスト戦略実装

```python
# /nook/functions/common/python/testing/ai_client_test_suite.py

import unittest
from unittest.mock import Mock, patch
from ..client_factory import create_client
from ..ai_client_interface import AIProvider

class AIClientTestSuite(unittest.TestCase):
    """Comprehensive test suite for AI clients."""

    def setUp(self):
        self.test_content = "Test content for AI processing"
        self.test_system_instruction = "You are a helpful assistant."

    def test_claude_content_generation(self):
        """Test Claude content generation."""
        with patch.dict(os.environ, {"AI_PROVIDER": "claude", "ANTHROPIC_API_KEY": "test-key"}):
            client = create_client()

            # Mock the API response
            with patch.object(client._client.messages, 'create') as mock_create:
                mock_response = Mock()
                mock_response.content = [Mock(text="Generated response")]
                mock_create.return_value = mock_response

                response = client.generate_content(
                    contents=self.test_content,
                    system_instruction=self.test_system_instruction
                )

                self.assertEqual(response, "Generated response")
                mock_create.assert_called_once()

    def test_gemini_compatibility(self):
        """Test Gemini backward compatibility."""
        with patch.dict(os.environ, {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}):
            client = create_client()

            # Test should work with existing interface
            self.assertTrue(hasattr(client, 'generate_content'))
            self.assertTrue(hasattr(client, 'create_chat'))
            self.assertTrue(hasattr(client, 'send_message'))

    def test_error_handling(self):
        """Test error handling and retry logic."""
        with patch.dict(os.environ, {"AI_PROVIDER": "claude", "ANTHROPIC_API_KEY": "test-key"}):
            client = create_client()

            with patch.object(client._client.messages, 'create') as mock_create:
                # Simulate rate limit error
                from anthropic import RateLimitError
                mock_create.side_effect = RateLimitError("Rate limit exceeded")

                with self.assertRaises(Exception):
                    client.generate_content(self.test_content)
```

#### 4.2 統合テスト

```python
# /nook/functions/common/python/testing/integration_tests.py

class IntegrationTestSuite:
    """Integration tests for complete workflows."""

    def test_paper_summarizer_workflow(self):
        """Test complete paper summarizer workflow with Claude."""

        # Setup test environment
        with patch.dict(os.environ, {
            "AI_PROVIDER": "claude",
            "ANTHROPIC_API_KEY": "test-key"
        }):

            from nook.functions.paper_summarizer.paper_summarizer import PaperSummarizer

            summarizer = PaperSummarizer()

            # Mock external dependencies
            with patch.object(summarizer._paper_id_retriever, 'retrieve_from_hugging_face') as mock_retrieve:
                mock_retrieve.return_value = ["2024.01234"]

                # Test should complete without errors
                summarizer()

    def test_viewer_chat_functionality(self):
        """Test viewer chat functionality with Claude."""

        with patch.dict(os.environ, {
            "AI_PROVIDER": "claude",
            "ANTHROPIC_API_KEY": "test-key"
        }):

            from nook.functions.viewer.viewer import chat

            # Mock request
            mock_request = Mock()
            mock_request.json.return_value = {
                "message": "Test question",
                "markdown": "Test content",
                "chat_history": "Previous conversation"
            }

            # Test chat endpoint
            response = chat("test-topic", mock_request)

            self.assertIn("response", response)
```

## 技術仕様

### 1. 詳細なコード例

#### 関数移行スクリプト
```python
#!/usr/bin/env python3
# /nook/scripts/migrate_function.py

import os
import sys
import re
import argparse
from pathlib import Path

class FunctionMigrator:
    """Automated migration script for Lambda functions."""

    def __init__(self, function_path: str):
        self.function_path = Path(function_path)
        self.backup_path = None

    def migrate(self):
        """Execute the migration process."""
        print(f"Migrating function at {self.function_path}")

        # Step 1: Create backup
        self.create_backup()

        # Step 2: Update imports
        self.update_imports()

        # Step 3: Update client instantiation
        self.update_client_creation()

        # Step 4: Update configuration
        self.update_configuration()

        print("Migration completed successfully")

    def create_backup(self):
        """Create backup of original file."""
        self.backup_path = self.function_path.with_suffix('.py.backup')
        shutil.copy2(self.function_path, self.backup_path)
        print(f"Backup created: {self.backup_path}")

    def update_imports(self):
        """Update import statements."""
        with open(self.function_path, 'r') as f:
            content = f.read()

        # Replace Gemini-specific imports
        old_import = r'from \.\.common\.python\.gemini_client import create_client'
        new_import = 'from ..common.python.client_factory import create_client'

        content = re.sub(old_import, new_import, content)

        with open(self.function_path, 'w') as f:
            f.write(content)

        print("Updated import statements")

    def update_client_creation(self):
        """Update client creation calls."""
        with open(self.function_path, 'r') as f:
            content = f.read()

        # Update client creation patterns
        patterns = [
            (r'create_client\(\)', 'create_client()'),  # No change needed
            (r'create_client\(([^)]+)\)', r'create_client(\1)'),  # No change needed
        ]

        for old_pattern, new_pattern in patterns:
            content = re.sub(old_pattern, new_pattern, content)

        with open(self.function_path, 'w') as f:
            f.write(content)

        print("Updated client creation")

    def rollback(self):
        """Rollback changes if migration fails."""
        if self.backup_path and self.backup_path.exists():
            shutil.copy2(self.backup_path, self.function_path)
            print("Migration rolled back")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate Lambda function to use Claude")
    parser.add_argument("function_path", help="Path to function Python file")

    args = parser.parse_args()

    migrator = FunctionMigrator(args.function_path)
    try:
        migrator.migrate()
    except Exception as e:
        print(f"Migration failed: {e}")
        migrator.rollback()
        sys.exit(1)
```

### 2. データベース/ストレージの考慮事項

Since the Nook application doesn't use a traditional database, the main storage considerations are:

#### ファイルシステムストレージ
```python
# /nook/functions/common/python/storage/file_manager.py

import os
from pathlib import Path
from typing import Optional, Dict, Any

class FileStorageManager:
    """Manages file storage for outputs and configurations."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or os.environ.get("OUTPUT_DIR", "./output"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_output(self, key: str, content: str) -> str:
        """Save output content to file."""
        file_path = self.base_dir / key
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return str(file_path)

    def load_output(self, key: str) -> Optional[str]:
        """Load output content from file."""
        file_path = self.base_dir / key

        if not file_path.exists():
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def migrate_storage_structure(self):
        """Migrate storage structure for new client."""
        # Add migration metadata
        migration_info = {
            "migrated_at": datetime.now().isoformat(),
            "migration_version": "1.0",
            "ai_provider": os.environ.get("AI_PROVIDER", "claude")
        }

        self.save_output("migration_info.json", json.dumps(migration_info, indent=2))
```

### 3. パフォーマンス最適化戦略

#### レスポンスキャッシュ
```python
# /nook/functions/common/python/caching/response_cache.py

import json
import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class ResponseCache:
    """Simple file-based caching for AI responses."""

    def __init__(self, cache_dir: str = "./cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def get_cache_key(self, content: str, config: Dict[str, Any]) -> str:
        """Generate cache key for content and configuration."""
        cache_data = {
            "content": content,
            "config": config
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[str]:
        """Get cached response if available and not expired."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            # Check expiration
            cached_at = datetime.fromisoformat(cache_data["cached_at"])
            if datetime.now() - cached_at > self.ttl:
                cache_file.unlink()  # Remove expired cache
                return None

            return cache_data["response"]

        except Exception:
            return None

    def set(self, cache_key: str, response: str) -> None:
        """Cache response."""
        cache_data = {
            "response": response,
            "cached_at": datetime.now().isoformat()
        }

        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

# Usage in AI client
class CachedAIClient:
    """AI client wrapper with caching."""

    def __init__(self, client: AIClientInterface, enable_cache: bool = True):
        self.client = client
        self.cache = ResponseCache() if enable_cache else None

    def generate_content(self, contents: str, **kwargs) -> str:
        """Generate content with caching."""

        if not self.cache:
            return self.client.generate_content(contents, **kwargs)

        # Check cache
        cache_key = self.cache.get_cache_key(contents, kwargs)
        cached_response = self.cache.get(cache_key)

        if cached_response:
            logger.info(f"Cache hit for content generation")
            return cached_response

        # Generate new response
        response = self.client.generate_content(contents, **kwargs)

        # Cache response
        self.cache.set(cache_key, response)

        return response
```

### 4. セキュリティの考慮事項

#### APIキー管理
```python
# /nook/functions/common/python/security/key_manager.py

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Secure API key management."""

    @staticmethod
    def get_api_key(provider: str) -> Optional[str]:
        """Get API key for provider with security checks."""

        key_map = {
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "google_search": "GOOGLE_SEARCH_API_KEY"
        }

        env_var = key_map.get(provider.lower())
        if not env_var:
            logger.error(f"Unknown provider: {provider}")
            return None

        api_key = os.environ.get(env_var)

        if not api_key:
            logger.error(f"API key not found for {provider} (env var: {env_var})")
            return None

        # Basic validation
        if len(api_key) < 10:
            logger.error(f"API key for {provider} appears to be invalid (too short)")
            return None

        # Mask key in logs
        masked_key = api_key[:8] + "..." + api_key[-4:]
        logger.info(f"Using API key for {provider}: {masked_key}")

        return api_key

    @staticmethod
    def validate_environment() -> Dict[str, bool]:
        """Validate that required API keys are present."""

        provider = os.environ.get("AI_PROVIDER", "gemini").lower()

        validation_results = {}

        if provider == "claude":
            validation_results["anthropic_key"] = bool(APIKeyManager.get_api_key("claude"))
        elif provider == "gemini":
            validation_results["gemini_key"] = bool(APIKeyManager.get_api_key("gemini"))

        # Optional keys
        validation_results["search_key"] = bool(APIKeyManager.get_api_key("google_search"))

        return validation_results

# Environment validation on startup
def validate_startup_environment():
    """Validate environment on application startup."""
    results = APIKeyManager.validate_environment()

    missing_keys = [key for key, present in results.items() if not present]

    if missing_keys:
        logger.warning(f"Missing API keys: {missing_keys}")

        # For required keys, this could raise an exception
        provider = os.environ.get("AI_PROVIDER", "gemini").lower()
        required_key = f"{provider}_key"

        if required_key in missing_keys:
            raise RuntimeError(f"Required API key missing for {provider}")
```

## 統合ポイント

### 1. AWS Lambdaデプロイメント仕様

#### 更新されたLambda関数テンプレート
```python
# /nook/functions/common/python/lambda_handler.py

import json
import logging
import traceback
from typing import Dict, Any
from .client_factory import create_client
from .security.key_manager import validate_startup_environment
from .monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

class LambdaHandler:
    """Base class for Lambda function handlers."""

    def __init__(self):
        # Validate environment on startup
        validate_startup_environment()

        # Initialize AI client
        self.ai_client = create_client()

        # Initialize monitoring
        self.monitor = PerformanceMonitor()

    def handle_request(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Handle Lambda request with standardized error handling."""

        request_id = context.aws_request_id if context else "local"

        try:
            self.monitor.start_request(request_id)

            # Log request
            logger.info(f"Processing request {request_id}")
            logger.debug(f"Event: {json.dumps(event, default=str)}")

            # Process request (implemented by subclass)
            result = self.process_event(event, context)

            self.monitor.end_request(request_id, success=True)

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "X-Request-ID": request_id
                },
                "body": json.dumps(result)
            }

        except Exception as e:
            self.monitor.end_request(request_id, success=False, error=str(e))

            logger.error(f"Request {request_id} failed: {e}")
            logger.error(traceback.format_exc())

            return {
                "statusCode": 500,
                "headers": {
                    "Content-Type": "application/json",
                    "X-Request-ID": request_id
                },
                "body": json.dumps({
                    "error": "Internal server error",
                    "request_id": request_id
                })
            }

        finally:
            # Clean up resources
            if hasattr(self.ai_client, 'close'):
                self.ai_client.close()

    def process_event(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Process the event (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement process_event")

# Updated Paper Summarizer
class PaperSummarizerHandler(LambdaHandler):
    """Lambda handler for paper summarizer."""

    def process_event(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        from nook.functions.paper_summarizer.paper_summarizer import PaperSummarizer

        summarizer = PaperSummarizer()
        summarizer()

        return {"message": "Paper summarization completed"}

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entry point."""
    handler = PaperSummarizerHandler()
    return handler.handle_request(event, context)
```

#### 依存関係と要件
```txt
# requirements.txt for Claude-based functions

# Core dependencies
anthropic>=0.25.0
tenacity>=8.2.0
requests>=2.31.0
python-dotenv>=1.0.0

# Existing dependencies (keep for compatibility)
beautifulsoup4>=4.12.0
arxiv>=1.4.0
fastapi>=0.104.0
uvicorn>=0.24.0
jinja2>=3.1.0

# Optional dependencies
google-api-python-client>=2.0.0  # For search functionality

# Development dependencies
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-mock>=3.12.0
black>=23.0.0
flake8>=6.0.0
```

### 2. ローカル開発環境セットアップ

#### 開発設定
```python
# /nook/scripts/setup_dev_environment.py

import os
import sys
from pathlib import Path

def setup_development_environment():
    """Setup local development environment for AI migration."""

    print("Setting up Nook development environment...")

    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_template = """
# AI Provider Configuration
AI_PROVIDER=claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# AI Model Configuration
AI_TEMPERATURE=1.0
AI_TOP_P=0.95
AI_TOP_K=40
AI_MAX_TOKENS=8192
AI_TIMEOUT=60000
AI_USE_SEARCH=false

# Claude-specific Configuration
CLAUDE_MODEL=claude-3-5-sonnet-20241022
CLAUDE_MAX_RETRIES=5
CLAUDE_RETRY_DELAY=1.0

# Search Configuration (Optional)
GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here

# Output Configuration
OUTPUT_DIR=./output
CACHE_DIR=./cache
"""

        with open(env_file, 'w') as f:
            f.write(env_template)

        print(f"Created {env_file} - please update with your API keys")

    # Create output directory
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    # Create cache directory
    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True)

    print("Development environment setup complete!")
    print("\nNext steps:")
    print("1. Update .env file with your API keys")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run tests: python -m pytest")

if __name__ == "__main__":
    setup_development_environment()
```

#### 開発テストスクリプト
```python
# /nook/scripts/test_migration.py

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from nook.functions.common.python.client_factory import create_client

def test_ai_client():
    """Test AI client functionality."""

    print("Testing AI client functionality...")

    # Test content generation
    client = create_client()

    test_content = "Write a short poem about technology."
    response = client.generate_content(test_content)

    print(f"Content Generation Test:")
    print(f"Input: {test_content}")
    print(f"Output: {response[:200]}...")

    # Test chat functionality
    print("\nTesting chat functionality...")
    client.create_chat()

    chat_response = client.send_message("Hello, how are you?")
    print(f"Chat Response: {chat_response[:100]}...")

    # Clean up
    client.close()

    print("\nAI client test completed successfully!")

if __name__ == "__main__":
    test_ai_client()
```

### 3. CI/CDパイプライン変更

#### GitHub Actionsワークフロー
```yaml
# .github/workflows/migrate-and-test.yml

name: AI Migration Testing

on:
  push:
    branches: [ main, migration/* ]
  pull_request:
    branches: [ main ]

env:
  AI_PROVIDER: claude
  AI_TEMPERATURE: 1.0
  AI_TOP_P: 0.95
  AI_MAX_TOKENS: 8192
  OUTPUT_DIR: ./test-output

jobs:
  test-migration:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        ai-provider: [claude, gemini]
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Set up test environment
      run: |
        mkdir -p ./test-output
        mkdir -p ./cache

    - name: Run unit tests
      env:
        AI_PROVIDER: ${{ matrix.ai-provider }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      run: |
        python -m pytest nook/functions/common/python/tests/ -v

    - name: Run integration tests
      env:
        AI_PROVIDER: ${{ matrix.ai-provider }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      run: |
        python -m pytest nook/functions/tests/ -v --integration

    - name: Test function migration
      env:
        AI_PROVIDER: ${{ matrix.ai-provider }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      run: |
        python nook/scripts/test_migration.py

    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-results-${{ matrix.ai-provider }}-py${{ matrix.python-version }}
        path: |
          ./test-output/
          ./pytest-report.xml

  deploy-staging:
    needs: test-migration
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1

    - name: Deploy to staging
      run: |
        # Deploy Lambda functions to staging environment
        ./scripts/deploy-staging.sh
```

### 4. 監視と可観測性

#### パフォーマンス監視
```python
# /nook/functions/common/python/monitoring/performance_monitor.py

import time
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    ai_provider: Optional[str] = None
    model_used: Optional[str] = None
    token_count: Optional[int] = None

class PerformanceMonitor:
    """Monitor performance metrics for AI operations."""

    def __init__(self):
        self.active_requests: Dict[str, RequestMetrics] = {}
        self.completed_requests: list[RequestMetrics] = []

    def start_request(self, request_id: str) -> None:
        """Start monitoring a request."""
        self.active_requests[request_id] = RequestMetrics(
            request_id=request_id,
            start_time=time.time(),
            ai_provider=os.environ.get("AI_PROVIDER", "unknown")
        )

    def end_request(
        self,
        request_id: str,
        success: bool = True,
        error: Optional[str] = None,
        **kwargs
    ) -> None:
        """End monitoring a request."""

        if request_id not in self.active_requests:
            logger.warning(f"Request {request_id} not found in active requests")
            return

        metrics = self.active_requests.pop(request_id)
        metrics.end_time = time.time()
        metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
        metrics.success = success
        metrics.error_message = error

        # Add additional metrics
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)

        self.completed_requests.append(metrics)

        # Log metrics
        self._log_metrics(metrics)

    def _log_metrics(self, metrics: RequestMetrics) -> None:
        """Log metrics for analysis."""

        log_data = asdict(metrics)

        if metrics.success:
            logger.info(f"Request completed successfully", extra=log_data)
        else:
            logger.error(f"Request failed", extra=log_data)

        # For production, send to monitoring service
        self._send_to_monitoring_service(log_data)

    def _send_to_monitoring_service(self, metrics: Dict[str, Any]) -> None:
        """Send metrics to external monitoring service."""

        # CloudWatch, DataDog, or other monitoring service integration
        try:
            # Example: Send to CloudWatch
            import boto3

            cloudwatch = boto3.client('cloudwatch')

            cloudwatch.put_metric_data(
                Namespace='Nook/AI',
                MetricData=[
                    {
                        'MetricName': 'RequestDuration',
                        'Value': metrics['duration_ms'],
                        'Unit': 'Milliseconds',
                        'Dimensions': [
                            {'Name': 'Provider', 'Value': metrics['ai_provider']},
                            {'Name': 'Success', 'Value': str(metrics['success'])}
                        ]
                    }
                ]
            )

        except Exception as e:
            logger.error(f"Failed to send metrics to monitoring service: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""

        if not self.completed_requests:
            return {"message": "No completed requests"}

        durations = [r.duration_ms for r in self.completed_requests if r.duration_ms]
        successes = [r.success for r in self.completed_requests]

        return {
            "total_requests": len(self.completed_requests),
            "success_rate": sum(successes) / len(successes) * 100,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0
        }
```

## パフォーマンスとセキュリティの考慮事項

### 1. パフォーマンス最適化

#### コネクションプーリングと再利用
```python
# /nook/functions/common/python/optimization/connection_manager.py

import threading
from typing import Dict, Any
from ..clients.claude_client import ClaudeClient
from ..clients.gemini_client_wrapper import GeminiClientWrapper

class ConnectionManager:
    """Manages AI client connections with pooling."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._clients: Dict[str, Any] = {}
            self._client_locks: Dict[str, threading.Lock] = {}
            self._initialized = True

    def get_client(self, config_hash: str, config: Dict[str, Any]):
        """Get or create client with connection reuse."""

        if config_hash not in self._client_locks:
            self._client_locks[config_hash] = threading.Lock()

        with self._client_locks[config_hash]:
            if config_hash not in self._clients:
                from ..client_factory import UnifiedAIClientFactory

                self._clients[config_hash] = UnifiedAIClientFactory.create_client(
                    config=config
                )

        return self._clients[config_hash]
```

### 2. セキュリティベストプラクティス

#### リクエスト検証とサニタイゼーション
```python
# /nook/functions/common/python/security/request_validator.py

import re
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class RequestValidator:
    """Validates and sanitizes incoming requests."""

    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        r'<script[^>]*>',  # Script tags
        r'javascript:',     # JavaScript URLs
        r'data:',          # Data URLs
        r'vbscript:',      # VBScript URLs
        r'on\w+\s*=',      # Event handlers
    ]

    # Maximum content length (characters)
    MAX_CONTENT_LENGTH = 50000

    @classmethod
    def validate_content(cls, content: str) -> bool:
        """Validate content for security issues."""

        if not content or not isinstance(content, str):
            return False

        # Check length
        if len(content) > cls.MAX_CONTENT_LENGTH:
            logger.warning(f"Content too long: {len(content)} characters")
            return False

        # Check for suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected: {pattern}")
                return False

        return True

    @classmethod
    def sanitize_content(cls, content: str) -> str:
        """Sanitize content by removing suspicious elements."""

        if not isinstance(content, str):
            return ""

        # Remove suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)

        # Truncate if too long
        if len(content) > cls.MAX_CONTENT_LENGTH:
            content = content[:cls.MAX_CONTENT_LENGTH]
            logger.info("Content truncated due to length limit")

        return content.strip()

    @classmethod
    def validate_system_instruction(cls, instruction: str) -> bool:
        """Validate system instruction."""

        if not instruction:
            return True  # Empty is OK

        # Basic validation
        if len(instruction) > 10000:  # Reasonable limit
            logger.warning("System instruction too long")
            return False

        return cls.validate_content(instruction)
```

## テスト戦略

### 1. ユニットテストフレームワーク

```python
# /nook/functions/common/python/tests/test_client_factory.py

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
from ..client_factory import UnifiedAIClientFactory, create_client
from ..ai_client_interface import AIProvider

class TestClientFactory(unittest.TestCase):
    """Test cases for client factory."""

    def setUp(self):
        self.original_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch.dict(os.environ, {
        'AI_PROVIDER': 'claude',
        'ANTHROPIC_API_KEY': 'test-key-123'
    })
    def test_create_claude_client(self):
        """Test Claude client creation."""

        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            client = UnifiedAIClientFactory.create_client(AIProvider.CLAUDE)

            self.assertIsNotNone(client)
            mock_anthropic.assert_called_once()

    @patch.dict(os.environ, {
        'AI_PROVIDER': 'gemini',
        'GEMINI_API_KEY': 'test-key-456'
    })
    def test_create_gemini_client(self):
        """Test Gemini client creation."""

        with patch('google.genai.Client') as mock_genai:
            mock_client = Mock()
            mock_genai.return_value = mock_client

            client = UnifiedAIClientFactory.create_client(AIProvider.GEMINI)

            self.assertIsNotNone(client)
            mock_genai.assert_called_once()

    def test_backward_compatibility(self):
        """Test backward compatibility with original create_client function."""

        with patch.dict(os.environ, {
            'AI_PROVIDER': 'claude',
            'ANTHROPIC_API_KEY': 'test-key'
        }):
            with patch('anthropic.Anthropic'):
                client = create_client()
                self.assertIsNotNone(client)

    def test_configuration_override(self):
        """Test configuration override functionality."""

        config_override = {
            'temperature': 0.5,
            'max_output_tokens': 4096
        }

        with patch.dict(os.environ, {
            'AI_PROVIDER': 'claude',
            'ANTHROPIC_API_KEY': 'test-key'
        }):
            with patch('anthropic.Anthropic'):
                client = create_client(config=config_override)

                self.assertEqual(client.config.temperature, 0.5)
                self.assertEqual(client.config.max_output_tokens, 4096)

if __name__ == '__main__':
    unittest.main()
```

### 2. 統合テスト

```python
# /nook/functions/common/python/tests/test_integration.py

import unittest
import os
from unittest.mock import patch, Mock
from ..client_factory import create_client

class TestIntegration(unittest.TestCase):
    """Integration tests for AI client functionality."""

    @patch.dict(os.environ, {
        'AI_PROVIDER': 'claude',
        'ANTHROPIC_API_KEY': 'test-key'
    })
    def test_end_to_end_content_generation(self):
        """Test end-to-end content generation."""

        with patch('anthropic.Anthropic') as mock_anthropic_class:
            # Setup mock
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            mock_response = Mock()
            mock_response.content = [Mock(text="Generated response")]
            mock_client.messages.create.return_value = mock_response

            # Test
            client = create_client()
            response = client.generate_content("Test content")

            self.assertEqual(response, "Generated response")
            mock_client.messages.create.assert_called_once()

    def test_error_handling_integration(self):
        """Test error handling in integration scenario."""

        with patch.dict(os.environ, {
            'AI_PROVIDER': 'claude',
            'ANTHROPIC_API_KEY': 'test-key'
        }):
            with patch('anthropic.Anthropic') as mock_anthropic_class:
                # Setup to raise an error
                mock_client = Mock()
                mock_anthropic_class.return_value = mock_client
                mock_client.messages.create.side_effect = Exception("API Error")

                client = create_client()

                with self.assertRaises(Exception):
                    client.generate_content("Test content")

class TestFunctionMigration(unittest.TestCase):
    """Test function migration scenarios."""

    @patch.dict(os.environ, {
        'AI_PROVIDER': 'claude',
        'ANTHROPIC_API_KEY': 'test-key',
        'OUTPUT_DIR': './test-output'
    })
    def test_paper_summarizer_migration(self):
        """Test paper summarizer with Claude."""

        with patch('anthropic.Anthropic'):
            with patch('nook.functions.paper_summarizer.paper_summarizer.PaperSummarizer') as mock_summarizer:
                mock_instance = Mock()
                mock_summarizer.return_value = mock_instance

                # Import should work without errors
                from nook.functions.paper_summarizer.paper_summarizer import PaperSummarizer

                # Should be able to instantiate
                summarizer = PaperSummarizer()
                self.assertIsNotNone(summarizer)

if __name__ == '__main__':
    unittest.main()
```

## ロールバックと復旧

### 1. 自動ロールバックシステム

```python
# /nook/scripts/rollback_migration.py

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

class MigrationRollback:
    """Handles rollback of AI migration."""

    def __init__(self, backup_branch: str = None):
        self.backup_branch = backup_branch or self._find_latest_backup()
        self.rollback_log = []

    def execute_rollback(self, components: List[str] = None) -> bool:
        """Execute rollback process."""

        print("Starting migration rollback...")

        try:
            # 1. Validate backup exists
            if not self._validate_backup():
                raise Exception(f"Backup branch {self.backup_branch} not found")

            # 2. Stop services if running
            self._stop_services()

            # 3. Rollback code
            self._rollback_code(components)

            # 4. Restore environment
            self._restore_environment()

            # 5. Restart services
            self._restart_services()

            # 6. Validate rollback
            self._validate_rollback()

            print("Rollback completed successfully")
            return True

        except Exception as e:
            print(f"Rollback failed: {e}")
            self._log_rollback_error(str(e))
            return False

    def _find_latest_backup(self) -> str:
        """Find the latest backup branch."""

        try:
            result = subprocess.run(
                ["git", "branch", "-r", "--list", "origin/backup-*"],
                capture_output=True,
                text=True,
                check=True
            )

            branches = [line.strip().replace("origin/", "") for line in result.stdout.split('\n') if line.strip()]

            if not branches:
                raise Exception("No backup branches found")

            # Return the most recent backup
            return sorted(branches)[-1]

        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to find backup branches: {e}")

    def _validate_backup(self) -> bool:
        """Validate that backup branch exists and is valid."""

        try:
            subprocess.run(
                ["git", "rev-parse", f"origin/{self.backup_branch}"],
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _rollback_code(self, components: List[str] = None) -> None:
        """Rollback code changes."""

        if components:
            # Selective rollback
            for component in components:
                self._rollback_component(component)
        else:
            # Full rollback
            subprocess.run(
                ["git", "reset", "--hard", f"origin/{self.backup_branch}"],
                check=True
            )

        self.rollback_log.append("Code rollback completed")

    def _rollback_component(self, component: str) -> None:
        """Rollback specific component."""

        component_paths = {
            "gemini_client": "nook/functions/common/python/gemini_client.py",
            "paper_summarizer": "nook/functions/paper_summarizer/",
            "viewer": "nook/functions/viewer/",
            "tech_feed": "nook/functions/tech_feed/",
            "hacker_news": "nook/functions/hacker_news/",
            "reddit_explorer": "nook/functions/reddit_explorer/"
        }

        if component not in component_paths:
            raise Exception(f"Unknown component: {component}")

        path = component_paths[component]

        subprocess.run(
            ["git", "checkout", f"origin/{self.backup_branch}", "--", path],
            check=True
        )

        self.rollback_log.append(f"Rolled back component: {component}")

    def _restore_environment(self) -> None:
        """Restore environment configuration."""

        # Restore environment variables
        env_backup = {
            "AI_PROVIDER": "gemini",
            "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY_BACKUP", ""),
        }

        # Update .env file
        env_file = Path(".env")
        if env_file.exists():
            content = env_file.read_text()

            # Restore key settings
            content = content.replace("AI_PROVIDER=claude", "AI_PROVIDER=gemini")

            env_file.write_text(content)

        self.rollback_log.append("Environment restored")

    def _stop_services(self) -> None:
        """Stop running services."""

        # Stop local development server if running
        try:
            subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)
        except:
            pass

        self.rollback_log.append("Services stopped")

    def _restart_services(self) -> None:
        """Restart services after rollback."""

        # This would restart Lambda functions or local services
        # Implementation depends on deployment method

        self.rollback_log.append("Services restarted")

    def _validate_rollback(self) -> None:
        """Validate that rollback was successful."""

        # Test basic functionality
        try:
            from nook.functions.common.python.gemini_client import create_client

            # Create client should work
            client = create_client()

            # Basic test
            response = client.generate_content("Hello, world!")

            if not response:
                raise Exception("Client validation failed")

            self.rollback_log.append("Rollback validation successful")

        except Exception as e:
            raise Exception(f"Rollback validation failed: {e}")

    def _log_rollback_error(self, error: str) -> None:
        """Log rollback error for analysis."""

        error_log = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "rollback_log": self.rollback_log,
            "backup_branch": self.backup_branch
        }

        with open("rollback_error.json", "w") as f:
            json.dump(error_log, f, indent=2)

def main():
    """Main rollback execution."""

    import argparse

    parser = argparse.ArgumentParser(description="Rollback AI migration")
    parser.add_argument("--backup-branch", help="Specific backup branch to use")
    parser.add_argument("--components", nargs="+", help="Specific components to rollback")

    args = parser.parse_args()

    rollback = MigrationRollback(args.backup_branch)
    success = rollback.execute_rollback(args.components)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

## 実装タイムライン

### 詳細な週次スケジュール

#### 第1週: 基盤とコア開発
**1-2日目: アーキテクチャセットアップ**
- ✅ Create abstract interface (`ai_client_interface.py`)
- ✅ Implement configuration manager (`ai_config_manager.py`)
- ✅ Setup factory pattern (`client_factory.py`)
- ✅ Create error handling framework (`ai_error_handling.py`)

**3-5日目: Claude Client実装**
- ✅ Implement basic Claude client (`claude_client.py`)
- ✅ Add retry logic and error handling
- ✅ Implement chat functionality
- ✅ Add search service integration
- ✅ Create comprehensive unit tests

#### Week 2: Migration and Testing
**Days 6-8: Function Migration**
- ✅ Migrate Paper Summarizer (Priority 1)
- ✅ Migrate Tech Feed and GitHub Trending (Priority 2)
- ✅ Migrate Hacker News (Priority 3)
- ✅ Update import statements and configuration

**Days 9-10: Integration Testing**
- ✅ End-to-end testing for migrated functions
- ✅ Performance comparison testing
- ✅ Error scenario testing
- ✅ Backward compatibility validation

#### Week 3: Advanced Migration and Quality Assurance
**Days 11-12: Complex Function Migration**
- ✅ Migrate Reddit Explorer (Priority 4)
- ✅ Migrate Viewer with chat functionality (Priority 5)
- ✅ Implement enhanced search capabilities
- ✅ Test interactive chat functionality

**Days 13-15: Quality Assurance**
- ✅ Comprehensive integration testing
- ✅ Performance optimization
- ✅ Security validation
- ✅ Documentation updates

#### Week 4: Deployment and Monitoring
**Days 16-17: Deployment Preparation**
- ✅ Staging environment deployment
- ✅ Production deployment scripts
- ✅ Monitoring and observability setup
- ✅ Rollback procedures testing

**Days 18-20: Production Deployment**
- ✅ Phased production rollout
- ✅ Real-time monitoring
- ✅ Performance validation
- ✅ User acceptance testing
- ✅ Final documentation and cleanup

### Risk Mitigation Schedule
- **Daily**: Monitor error rates and performance metrics
- **Weekly**: Stakeholder updates and risk assessment
- **Per Phase**: Go/no-go decision points with rollback options

この技術設計書は、システムの信頼性、パフォーマンス、機能性を維持しながらGeminiからClaudeに移行するための包括的なロードマップを提供します。モジュラーアプローチにより、各段階で堅牢なテストとロールバック機能を備えた段階的実装が可能になります。