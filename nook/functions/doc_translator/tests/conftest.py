"""
Shared test configuration for documentation translation tests.

This module provides common fixtures and configuration for all translation tests,
following pytest best practices and the Nook project's testing standards.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import json


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        project_path = Path(tmp_dir)

        # Create docs subdirectory
        docs_dir = project_path / "docs"
        docs_dir.mkdir()

        # Create sample files
        (docs_dir / "sample.md").write_text(
            "# Sample Document\n\nThis is a sample document for testing.",
            encoding='utf-8'
        )

        yield project_path


@pytest.fixture
def mock_anthropic_api():
    """Mock Anthropic API for Claude client testing."""
    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Mocked translation response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        yield mock_client


@pytest.fixture
def sample_english_content():
    """Sample English content for translation testing."""
    return """
    # Technical Documentation

    This document describes the system architecture.

    ## API Endpoints

    The system provides REST API endpoints for:
    - User authentication
    - Data retrieval
    - Configuration management

    ```python
    def authenticate_user(api_key: str) -> bool:
        return validate_api_key(api_key)
    ```

    See [Configuration Guide](config.md) for details.
    """


@pytest.fixture
def sample_japanese_content():
    """Sample Japanese content (expected translation output)."""
    return """
    # 技術ドキュメント

    この文書はシステムアーキテクチャについて説明します。

    ## APIエンドポイント

    システムは以下のREST APIエンドポイントを提供します：
    - ユーザー認証
    - データ取得
    - 設定管理

    ```python
    def authenticate_user(api_key: str) -> bool:
        return validate_api_key(api_key)
    ```

    詳細については[設定ガイド](config.md)を参照してください。
    """


@pytest.fixture
def standard_terminology_db():
    """Standard terminology database for consistent testing."""
    return {
        "API": {
            "japanese": "API",
            "context": "programming",
            "priority": "critical"
        },
        "endpoint": {
            "japanese": "エンドポイント",
            "context": "web_api",
            "priority": "high"
        },
        "authentication": {
            "japanese": "認証",
            "context": "security",
            "priority": "critical"
        },
        "configuration": {
            "japanese": "設定",
            "context": "system",
            "priority": "high"
        },
        "REST": {
            "japanese": "REST",
            "context": "web_api",
            "priority": "high"
        }
    }


@pytest.fixture
def quality_thresholds():
    """Standard quality thresholds based on detailed_design.md."""
    return {
        "technical_accuracy": 95.0,
        "terminology_consistency": 98.0,
        "linguistic_quality": 90.0,
        "structural_integrity": 100.0,
        "overall_quality": 95.0
    }


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data files."""
    current_dir = Path(__file__).parent
    test_data_path = current_dir / "test_data"
    test_data_path.mkdir(exist_ok=True)
    return test_data_path


# Environment setup for testing
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Automatically setup test environment for all tests."""
    # Set test environment variables
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setenv("AI_CLIENT_TYPE", "claude")
    monkeypatch.setenv("PYTEST_RUNNING", "true")


# Marks for different test categories
def pytest_configure(config):
    """Configure pytest marks for test categorization."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for system interactions"
    )
    config.addinivalue_line(
        "markers", "quality: Quality validation tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "red_phase: TDD Red phase tests (expected to fail initially)"
    )


# Skip conditions for RED phase testing
def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    if "red_phase" in item.keywords:
        # These tests are expected to fail in RED phase
        # Add special handling if needed
        pass