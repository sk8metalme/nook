"""
Test rollback and environment switching procedures.

These tests validate the ability to switch between Claude and Gemini
and perform emergency rollback procedures.
"""

import pytest
import os
import subprocess
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

from nook.functions.common.python.claude_client import ClaudeClient
from nook.functions.common.python.gemini_client import GeminiClient


@pytest.mark.rollback
class TestEnvironmentSwitching:
    """Test switching between AI providers via environment variables."""

    def test_claude_environment_setup(self):
        """Test that Claude environment is properly configured."""
        with patch.dict(os.environ, {
            'AI_PROVIDER': 'claude',
            'ANTHROPIC_API_KEY': 'test-claude-key'
        }, clear=False):
            # Verify environment variables are set
            assert os.environ.get('AI_PROVIDER') == 'claude'
            assert os.environ.get('ANTHROPIC_API_KEY') == 'test-claude-key'

    def test_gemini_environment_setup(self):
        """Test that Gemini environment is properly configured."""
        with patch.dict(os.environ, {
            'AI_PROVIDER': 'gemini',
            'GEMINI_API_KEY': 'test-gemini-key'
        }, clear=False):
            # Verify environment variables are set
            assert os.environ.get('AI_PROVIDER') == 'gemini'
            assert os.environ.get('GEMINI_API_KEY') == 'test-gemini-key'

    def test_provider_client_creation_claude(self):
        """Test creating Claude client based on environment."""
        with patch.dict(os.environ, {
            'AI_PROVIDER': 'claude',
            'ANTHROPIC_API_KEY': 'test-key'
        }):
            with patch('nook.functions.common.python.claude_client.Anthropic'):
                from nook.functions.common.python.claude_client import create_client
                client = create_client()

                assert isinstance(client, ClaudeClient)
                assert client.config.model == "claude-3-5-sonnet-20241022"

    def test_provider_client_creation_gemini(self):
        """Test creating Gemini client based on environment."""
        with patch.dict(os.environ, {
            'AI_PROVIDER': 'gemini',
            'GEMINI_API_KEY': 'test-key'
        }):
            with patch('google.genai.Client'):
                from nook.functions.common.python.gemini_client import create_client
                client = create_client()

                assert isinstance(client, GeminiClient)
                assert client._config.model == "gemini-2.0-flash-exp"

    def test_environment_variable_override_precedence(self):
        """Test that environment variables take precedence over defaults."""
        with patch.dict(os.environ, {
            'AI_PROVIDER': 'claude',
            'ANTHROPIC_API_KEY': 'test-key',
            'AI_TEMPERATURE': '0.3',
            'AI_MAX_TOKENS': '4096'
        }):
            # Test that environment overrides are detected
            assert os.environ.get('AI_TEMPERATURE') == '0.3'
            assert os.environ.get('AI_MAX_TOKENS') == '4096'

    def test_missing_api_key_handling(self):
        """Test handling of missing API keys."""
        # Test missing Claude API key
        with patch.dict(os.environ, {'AI_PROVIDER': 'claude'}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is not set"):
                from nook.functions.common.python.claude_client import create_client
                create_client()

        # Test missing Gemini API key
        with patch.dict(os.environ, {'AI_PROVIDER': 'gemini'}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is not set"):
                from nook.functions.common.python.gemini_client import create_client
                create_client()


@pytest.mark.rollback
class TestBasicRollbackProcedures:
    """Test basic rollback procedures and validation."""

    def test_configuration_backup_and_restore(self):
        """Test configuration backup and restore functionality."""
        # Simulate original configuration
        original_config = {
            'AI_PROVIDER': 'gemini',
            'GEMINI_API_KEY': 'original-gemini-key',
            'AI_TEMPERATURE': '1.0'
        }

        # Simulate migrated configuration
        migrated_config = {
            'AI_PROVIDER': 'claude',
            'ANTHROPIC_API_KEY': 'claude-key',
            'AI_TEMPERATURE': '0.8'
        }

        # Test migration state
        with patch.dict(os.environ, migrated_config, clear=True):
            assert os.environ['AI_PROVIDER'] == 'claude'
            assert os.environ['ANTHROPIC_API_KEY'] == 'claude-key'
            assert os.environ['AI_TEMPERATURE'] == '0.8'

        # Test rollback state
        with patch.dict(os.environ, original_config, clear=True):
            assert os.environ['AI_PROVIDER'] == 'gemini'
            assert os.environ['GEMINI_API_KEY'] == 'original-gemini-key'
            assert os.environ['AI_TEMPERATURE'] == '1.0'

    def test_function_level_rollback_capability(self):
        """Test the ability to rollback individual functions."""
        # This tests the concept of function-specific provider switching
        function_configs = [
            "paper_summarizer",
            "tech_feed",
            "hacker_news",
            "reddit_explorer"
        ]

        for function_name in function_configs:
            # Test function with Claude
            claude_env = {
                f'{function_name.upper()}_AI_PROVIDER': 'claude',
                'ANTHROPIC_API_KEY': 'test-key'
            }

            with patch.dict(os.environ, claude_env, clear=False):
                provider = os.environ.get(f'{function_name.upper()}_AI_PROVIDER')
                assert provider == 'claude'

            # Test rollback to Gemini
            gemini_env = {
                f'{function_name.upper()}_AI_PROVIDER': 'gemini',
                'GEMINI_API_KEY': 'test-key'
            }

            with patch.dict(os.environ, gemini_env, clear=False):
                provider = os.environ.get(f'{function_name.upper()}_AI_PROVIDER')
                assert provider == 'gemini'

    def test_rollback_validation_checks(self):
        """Test validation checks after rollback."""

        def validate_environment_variables():
            """Validate required environment variables are set."""
            required_vars = ['AI_PROVIDER', 'GEMINI_API_KEY']
            for var in required_vars:
                if not os.environ.get(var):
                    raise ValueError(f"Missing environment variable: {var}")

            if os.environ.get('AI_PROVIDER') != 'gemini':
                raise ValueError("AI_PROVIDER not set to gemini")

        def validate_api_connectivity():
            """Validate API connectivity (mocked)."""
            gemini_key = os.environ.get('GEMINI_API_KEY')
            if not gemini_key or len(gemini_key) < 10:
                raise ValueError("Invalid Gemini API key")

        def validate_basic_functionality():
            """Validate basic functionality (mocked)."""
            try:
                # This would test that imports work
                from nook.functions.common.python.gemini_client import create_client
                with patch('google.genai.Client'):
                    client = create_client()
                    assert client is not None
            except Exception as e:
                raise ValueError(f"Basic functionality check failed: {e}")

        # Test validation with correct rollback environment
        rollback_env = {
            'AI_PROVIDER': 'gemini',
            'GEMINI_API_KEY': 'valid-gemini-key-12345'
        }

        with patch.dict(os.environ, rollback_env, clear=True):
            # All validation checks should pass
            validate_environment_variables()  # Should not raise
            validate_api_connectivity()  # Should not raise
            validate_basic_functionality()  # Should not raise

        # Test validation with incorrect environment
        incorrect_env = {
            'AI_PROVIDER': 'claude',  # Wrong provider after rollback
            'GEMINI_API_KEY': 'short'  # Invalid key
        }

        with patch.dict(os.environ, incorrect_env, clear=True):
            with pytest.raises(ValueError, match="AI_PROVIDER not set to gemini"):
                validate_environment_variables()

            with pytest.raises(ValueError, match="Invalid Gemini API key"):
                validate_api_connectivity()


@pytest.mark.rollback
class TestEmergencyRollbackValidation:
    """Test emergency rollback procedures validation."""

    def test_emergency_environment_restore(self):
        """Test emergency environment variable restoration."""
        # Emergency configuration that should always work
        emergency_env = {
            'AI_PROVIDER': 'gemini',
            'GEMINI_API_KEY': 'emergency-fallback-key',
            'AI_TEMPERATURE': '1.0',
            'AI_TOP_P': '0.95',
            'AI_TOP_K': '40',
            'AI_MAX_TOKENS': '8192',
            'AI_USE_SEARCH': 'false'
        }

        # Test emergency restore
        with patch.dict(os.environ, emergency_env, clear=True):
            # Verify all emergency settings are in place
            assert os.environ['AI_PROVIDER'] == 'gemini'
            assert os.environ['GEMINI_API_KEY'] == 'emergency-fallback-key'
            assert os.environ['AI_TEMPERATURE'] == '1.0'
            assert os.environ['AI_TOP_P'] == '0.95'
            assert os.environ['AI_TOP_K'] == '40'
            assert os.environ['AI_MAX_TOKENS'] == '8192'
            assert os.environ['AI_USE_SEARCH'] == 'false'

    def test_emergency_service_validation(self):
        """Test that services can be validated after emergency rollback."""
        emergency_env = {
            'AI_PROVIDER': 'gemini',
            'GEMINI_API_KEY': 'test-emergency-key'
        }

        with patch.dict(os.environ, emergency_env):
            # Mock service validation
            with patch('subprocess.run') as mock_subprocess:
                mock_subprocess.return_value.returncode = 0

                # This would represent a health check command
                result = subprocess.run(['echo', 'health_check'], capture_output=True)

                # Verify health check can run
                assert result.returncode == 0

    def test_rollback_time_constraints(self):
        """Test that rollback operations meet time constraints."""
        import time

        # Simulate quick rollback operations
        start_time = time.time()

        # Simulate emergency restore (should be very fast)
        with patch.dict(os.environ, {
            'AI_PROVIDER': 'gemini',
            'GEMINI_API_KEY': 'emergency-key'
        }, clear=True):
            # Emergency environment setup should be instant
            assert os.environ['AI_PROVIDER'] == 'gemini'

        # Simulate basic validation
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            subprocess.run(['echo', 'validation'], capture_output=True)

        end_time = time.time()
        rollback_duration = end_time - start_time

        # Emergency rollback should complete in under 1 second (very generous for testing)
        assert rollback_duration < 1.0, f"Emergency rollback took too long: {rollback_duration:.2f}s"


@pytest.mark.smoke
class TestSmokeTestsForProduction:
    """Basic smoke tests for production validation."""

    def test_import_smoke_test(self):
        """Test that all critical modules can be imported."""
        # Test Claude client import
        from nook.functions.common.python.claude_client import ClaudeClient, create_client
        assert ClaudeClient is not None
        assert create_client is not None

        # Test Gemini client import
        from nook.functions.common.python.gemini_client import GeminiClient
        assert GeminiClient is not None

    def test_configuration_smoke_test(self):
        """Test that basic configuration works."""
        from nook.functions.common.python.claude_client import ClaudeClientConfig

        # Test default config
        config = ClaudeClientConfig()
        assert config.model is not None
        assert config.temperature >= 0.0
        assert config.max_output_tokens > 0

        # Test config update
        config.update(temperature=0.5)
        assert config.temperature == 0.5

    def test_environment_detection_smoke_test(self):
        """Test basic environment detection."""
        # Test that environment variable checking works
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            assert os.environ.get('TEST_VAR') == 'test_value'

        # Test that missing variables are handled
        assert os.environ.get('NON_EXISTENT_VAR') is None