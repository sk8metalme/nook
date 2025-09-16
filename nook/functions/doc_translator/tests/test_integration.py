"""
Integration Test Suite for Documentation Translation System - RED Phase (TDD).

This module implements failing integration tests that define the expected behavior
of the complete translation system, including end-to-end workflows,
system interactions, and real-world scenarios based on detailed_design.md.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import List, Dict, Optional
import tempfile
import json

# These imports will fail initially - this is expected in RED phase
try:
    from nook.functions.doc_translator.translation_orchestrator import (
        TranslationOrchestrator,
        TranslationProject,
        ProjectStatus,
        TranslationWorkflow
    )
except ImportError:
    # Expected to fail in RED phase
    TranslationOrchestrator = None
    TranslationProject = None
    ProjectStatus = None
    TranslationWorkflow = None

try:
    from nook.functions.doc_translator.quality_gate_orchestrator import (
        QualityGateOrchestrator,
        QualityGateResult,
        GateExecutionReport
    )
except ImportError:
    # Expected to fail in RED phase
    QualityGateOrchestrator = None
    QualityGateResult = None
    GateExecutionReport = None

try:
    from nook.functions.doc_translator.project_monitor import (
        ProjectMonitor,
        ProgressReport,
        QualityTrend,
        AlertSystem
    )
except ImportError:
    # Expected to fail in RED phase
    ProjectMonitor = None
    ProgressReport = None
    QualityTrend = None
    AlertSystem = None


class TestTranslationOrchestrator:
    """Integration tests for the main translation orchestrator."""

    @pytest.fixture
    def sample_project_structure(self, tmp_path):
        """Create a sample project structure for testing."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Create sample documentation files
        (docs_dir / "technical_design.md").write_text("""
        # Technical Design Document

        This document outlines the technical design of the system.

        ## Architecture Overview

        The system follows a microservices architecture with the following components:

        - **API Gateway**: Routes requests to appropriate services
        - **Authentication Service**: Handles user authentication
        - **Data Service**: Manages data persistence

        ```python
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/health")
        def health_check():
            return {"status": "healthy"}
        ```

        See [Configuration Guide](config.md) for setup details.
        """, encoding='utf-8')

        (docs_dir / "config.md").write_text("""
        # Configuration Guide

        This guide explains how to configure the system.

        ## Environment Variables

        Set the following environment variables:

        - `API_KEY`: Your API key
        - `DATABASE_URL`: Database connection string
        - `DEBUG`: Set to `true` for development

        ## Configuration File

        Create a `config.yaml` file:

        ```yaml
        database:
          host: localhost
          port: 5432
        ```
        """, encoding='utf-8')

        return docs_dir

    @pytest.fixture
    def mock_claude_client(self):
        """Mock Claude client with realistic responses."""
        mock_client = Mock()

        # Mock translation responses
        def mock_translate(content, **kwargs):
            # Simple mock translation for testing
            if "Technical Design Document" in content:
                return "技術設計書"
            elif "Configuration Guide" in content:
                return "設定ガイド"
            elif "Architecture Overview" in content:
                return "アーキテクチャ概要"
            else:
                return f"翻訳: {content[:50]}..."

        mock_client.generate_content.side_effect = mock_translate
        return mock_client

    def test_translation_orchestrator_initialization(self, sample_project_structure):
        """Test successful translation orchestrator initialization."""
        if TranslationOrchestrator is None:
            pytest.skip("TranslationOrchestrator not implemented yet - RED phase")

        orchestrator = TranslationOrchestrator(
            project_path=sample_project_structure,
            output_path=sample_project_structure / "translated"
        )

        assert orchestrator is not None
        assert orchestrator.project_path == sample_project_structure
        assert orchestrator.output_path == sample_project_structure / "translated"

    def test_full_project_translation_workflow(self, sample_project_structure, mock_claude_client):
        """Test complete project translation workflow integration."""
        if TranslationOrchestrator is None:
            pytest.skip("TranslationOrchestrator not implemented yet - RED phase")

        orchestrator = TranslationOrchestrator(
            project_path=sample_project_structure,
            output_path=sample_project_structure / "translated"
        )

        orchestrator.set_claude_client(mock_claude_client)

        # Execute full translation workflow
        project_result = orchestrator.execute_translation_workflow()

        assert isinstance(project_result, TranslationProject)
        assert project_result.status == ProjectStatus.COMPLETED
        assert project_result.total_documents == 2  # technical_design.md and config.md
        assert project_result.translated_documents == 2
        assert project_result.overall_quality >= 95.0

        # Check that output files are created
        output_path = sample_project_structure / "translated"
        assert (output_path / "technical_design.md").exists()
        assert (output_path / "config.md").exists()

    def test_staged_translation_with_context_preservation(self, sample_project_structure, mock_claude_client):
        """Test staged translation with context preservation between sections."""
        if TranslationOrchestrator is None:
            pytest.skip("TranslationOrchestrator not implemented yet - RED phase")

        orchestrator = TranslationOrchestrator(
            project_path=sample_project_structure,
            output_path=sample_project_structure / "translated"
        )

        orchestrator.set_claude_client(mock_claude_client)
        orchestrator.enable_context_preservation(window_size=2)  # Use 2-section context window

        # Execute staged translation
        stage_results = orchestrator.execute_staged_translation()

        assert len(stage_results) > 0

        for stage in stage_results:
            assert hasattr(stage, 'section_id')
            assert hasattr(stage, 'translation_result')
            assert hasattr(stage, 'context_used')
            assert stage.context_used is True  # Context should be preserved

    def test_quality_gate_integration_workflow(self, sample_project_structure, mock_claude_client):
        """Test integration with quality gate system."""
        if TranslationOrchestrator is None or QualityGateOrchestrator is None:
            pytest.skip("Components not implemented yet - RED phase")

        translation_orchestrator = TranslationOrchestrator(
            project_path=sample_project_structure,
            output_path=sample_project_structure / "translated"
        )

        quality_orchestrator = QualityGateOrchestrator()

        translation_orchestrator.set_claude_client(mock_claude_client)
        translation_orchestrator.set_quality_orchestrator(quality_orchestrator)

        # Execute with quality gates
        project_result = translation_orchestrator.execute_with_quality_gates()

        assert project_result.quality_gates_passed is True
        assert len(project_result.quality_gate_reports) == 3  # 3 phases from detailed_design.md

        # Verify each quality gate was executed
        gate_phases = [report.phase for report in project_result.quality_gate_reports]
        assert "initial" in gate_phases
        assert "improvement" in gate_phases
        assert "final" in gate_phases

    def test_terminology_consistency_across_documents(self, sample_project_structure, mock_claude_client):
        """Test terminology consistency across multiple documents."""
        if TranslationOrchestrator is None:
            pytest.skip("TranslationOrchestrator not implemented yet - RED phase")

        orchestrator = TranslationOrchestrator(
            project_path=sample_project_structure,
            output_path=sample_project_structure / "translated"
        )

        # Load terminology database
        terminology_db = {
            "API": "API",
            "service": "サービス",
            "authentication": "認証",
            "configuration": "設定"
        }
        orchestrator.load_terminology_database(terminology_db)

        orchestrator.set_claude_client(mock_claude_client)

        project_result = orchestrator.execute_translation_workflow()

        # Check terminology consistency across all translated documents
        consistency_report = orchestrator.get_cross_document_consistency_report()

        assert consistency_report.overall_consistency >= 99.0
        assert len(consistency_report.inconsistent_terms) == 0

    def test_cross_reference_integrity_preservation(self, sample_project_structure, mock_claude_client):
        """Test preservation of cross-references between documents."""
        if TranslationOrchestrator is None:
            pytest.skip("TranslationOrchestrator not implemented yet - RED phase")

        orchestrator = TranslationOrchestrator(
            project_path=sample_project_structure,
            output_path=sample_project_structure / "translated"
        )

        orchestrator.set_claude_client(mock_claude_client)
        orchestrator.enable_cross_reference_tracking()

        project_result = orchestrator.execute_translation_workflow()

        # Check cross-reference integrity
        reference_integrity = orchestrator.validate_cross_reference_integrity()

        assert reference_integrity.all_references_valid is True
        assert reference_integrity.broken_references == 0
        assert reference_integrity.integrity_score == 100.0

    def test_large_document_handling(self, tmp_path, mock_claude_client):
        """Test handling of large documents (>170,000 characters)."""
        if TranslationOrchestrator is None:
            pytest.skip("TranslationOrchestrator not implemented yet - RED phase")

        # Create a large document
        large_content = "# Large Document\n\n" + ("This is a section with content.\n\n" * 5000)
        large_doc_path = tmp_path / "large_document.md"
        large_doc_path.write_text(large_content, encoding='utf-8')

        orchestrator = TranslationOrchestrator(
            project_path=tmp_path,
            output_path=tmp_path / "translated"
        )

        orchestrator.set_claude_client(mock_claude_client)
        orchestrator.configure_for_large_documents(chunk_size=2000)

        project_result = orchestrator.execute_translation_workflow()

        assert project_result.status == ProjectStatus.COMPLETED
        assert project_result.total_character_count > 170000
        assert project_result.chunks_processed > 1  # Should be chunked

    def test_progress_monitoring_integration(self, sample_project_structure, mock_claude_client):
        """Test integration with progress monitoring system."""
        if TranslationOrchestrator is None or ProjectMonitor is None:
            pytest.skip("Components not implemented yet - RED phase")

        orchestrator = TranslationOrchestrator(
            project_path=sample_project_structure,
            output_path=sample_project_structure / "translated"
        )

        monitor = ProjectMonitor()
        orchestrator.set_monitor(monitor)
        orchestrator.set_claude_client(mock_claude_client)

        progress_updates = []
        def progress_callback(update):
            progress_updates.append(update)

        monitor.set_progress_callback(progress_callback)

        project_result = orchestrator.execute_translation_workflow()

        assert len(progress_updates) > 0
        assert progress_updates[-1].completion_percentage == 100.0
        assert any(update.stage == "translation" for update in progress_updates)
        assert any(update.stage == "quality_check" for update in progress_updates)

    def test_error_recovery_and_resilience(self, sample_project_structure, mock_claude_client):
        """Test error recovery and system resilience."""
        if TranslationOrchestrator is None:
            pytest.skip("TranslationOrchestrator not implemented yet - RED phase")

        orchestrator = TranslationOrchestrator(
            project_path=sample_project_structure,
            output_path=sample_project_structure / "translated"
        )

        # Configure client to fail on first attempt, succeed on retry
        call_count = 0
        def failing_translate(content, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 calls
                raise Exception("Simulated API failure")
            return f"翻訳: {content[:50]}..."

        mock_claude_client.generate_content.side_effect = failing_translate
        orchestrator.set_claude_client(mock_claude_client)
        orchestrator.enable_error_recovery(max_retries=3)

        project_result = orchestrator.execute_translation_workflow()

        # Should succeed despite initial failures
        assert project_result.status == ProjectStatus.COMPLETED
        assert project_result.retry_count > 0
        assert project_result.errors_recovered > 0

    def test_concurrent_translation_processing(self, sample_project_structure, mock_claude_client):
        """Test concurrent processing of multiple documents."""
        if TranslationOrchestrator is None:
            pytest.skip("TranslationOrchestrator not implemented yet - RED phase")

        orchestrator = TranslationOrchestrator(
            project_path=sample_project_structure,
            output_path=sample_project_structure / "translated"
        )

        orchestrator.set_claude_client(mock_claude_client)
        orchestrator.enable_concurrent_processing(max_workers=2)

        start_time = orchestrator.get_current_time()
        project_result = orchestrator.execute_translation_workflow()
        end_time = orchestrator.get_current_time()

        assert project_result.status == ProjectStatus.COMPLETED
        assert project_result.concurrent_processing_used is True
        # Should be faster than sequential processing (this is approximate)
        assert project_result.total_processing_time < (end_time - start_time) * 1.5


class TestQualityGateOrchestrator:
    """Integration tests for quality gate orchestration."""

    def test_sequential_quality_gate_execution(self):
        """Test sequential execution of quality gates."""
        if QualityGateOrchestrator is None:
            pytest.skip("QualityGateOrchestrator not implemented yet - RED phase")

        orchestrator = QualityGateOrchestrator()

        sample_translation = {
            "original": "The API provides authentication services.",
            "translated": "APIは認証サービスを提供します。"
        }

        gate_result = orchestrator.execute_quality_gates(sample_translation)

        assert isinstance(gate_result, QualityGateResult)
        assert gate_result.all_gates_passed is True
        assert len(gate_result.gate_reports) == 3  # 3 phases

        # Verify execution order
        phases = [report.phase for report in gate_result.gate_reports]
        assert phases == ["initial", "improvement", "final"]

    def test_quality_gate_failure_handling(self):
        """Test handling of quality gate failures."""
        if QualityGateOrchestrator is None:
            pytest.skip("QualityGateOrchestrator not implemented yet - RED phase")

        orchestrator = QualityGateOrchestrator()

        # Poor quality translation that should fail gates
        poor_translation = {
            "original": "The API provides authentication services with OAuth support.",
            "translated": "api provides auth services with oauth support."  # Poor quality
        }

        gate_result = orchestrator.execute_quality_gates(poor_translation)

        assert gate_result.all_gates_passed is False
        assert gate_result.failed_gate is not None
        assert len(gate_result.improvement_suggestions) > 0

    def test_quality_gate_metrics_aggregation(self):
        """Test aggregation of quality metrics across gates."""
        if QualityGateOrchestrator is None:
            pytest.skip("QualityGateOrchestrator not implemented yet - RED phase")

        orchestrator = QualityGateOrchestrator()

        translations = [
            {
                "original": "API authentication is required.",
                "translated": "API認証が必要です。"
            },
            {
                "original": "Configuration settings are stored in YAML.",
                "translated": "設定はYAMLファイルに保存されます。"
            }
        ]

        batch_results = orchestrator.execute_batch_quality_gates(translations)

        assert len(batch_results) == len(translations)

        aggregated_metrics = orchestrator.aggregate_quality_metrics(batch_results)
        assert hasattr(aggregated_metrics, 'average_technical_accuracy')
        assert hasattr(aggregated_metrics, 'average_terminology_consistency')
        assert hasattr(aggregated_metrics, 'overall_pass_rate')

        assert 0 <= aggregated_metrics.overall_pass_rate <= 100


class TestSystemIntegrationScenarios:
    """End-to-end system integration test scenarios."""

    def test_nook_project_documentation_translation_scenario(self, tmp_path):
        """Test realistic Nook project documentation translation scenario."""
        if TranslationOrchestrator is None:
            pytest.skip("TranslationOrchestrator not implemented yet - RED phase")

        # Create realistic Nook project structure
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Simulate actual Nook documentation files
        (docs_dir / "technical_design.md").write_text("""
        # Nook Technical Design

        Nook is a local/self-hosted fork of a news aggregation system.

        ## Architecture

        ### Components
        - Reddit Explorer
        - Hacker News Collector
        - GitHub Trending Monitor
        - Paper Summarizer

        ### Claude Integration
        The system uses Claude API for content summarization.

        ```python
        from nook.functions.common.python.claude_client import ClaudeClient

        client = ClaudeClient()
        summary = client.generate_content("Summarize this content...")
        ```
        """, encoding='utf-8')

        (docs_dir / "migration_status.md").write_text("""
        # Gemini to Claude Migration Status

        ## Completed Components
        - [x] Paper Summarizer: Migrated to use factory pattern
        - [ ] Tech Feed: Pending migration
        - [ ] Hacker News: Pending migration

        ## Migration Benefits
        - Improved error handling
        - Better rate limiting
        - Enhanced conversation capabilities
        """, encoding='utf-8')

        orchestrator = TranslationOrchestrator(
            project_path=docs_dir,
            output_path=tmp_path / "translated_docs"
        )

        # Configure for Nook-specific terminology
        nook_terminology = {
            "Nook": "Nook",  # Keep as-is, proper noun
            "Reddit": "Reddit",
            "Hacker News": "Hacker News",
            "GitHub": "GitHub",
            "Claude": "Claude",
            "Gemini": "Gemini",
            "API": "API",
            "migration": "移行",
            "factory pattern": "ファクトリーパターン"
        }

        orchestrator.load_terminology_database(nook_terminology)

        # Mock Claude client for this integration test
        mock_client = Mock()
        mock_client.generate_content.return_value = "技術設計書の翻訳内容"
        orchestrator.set_claude_client(mock_client)

        # Execute full workflow
        project_result = orchestrator.execute_translation_workflow()

        assert project_result.status == ProjectStatus.COMPLETED
        assert project_result.terminology_consistency >= 99.0
        assert project_result.nook_specific_terms_preserved is True

    def test_170k_character_document_translation_scenario(self, tmp_path):
        """Test translation of the actual 170,000 character requirement."""
        if TranslationOrchestrator is None:
            pytest.skip("TranslationOrchestrator not implemented yet - RED phase")

        # Create document that simulates the 170k character requirement
        large_content = self._create_realistic_170k_document()

        large_doc = tmp_path / "comprehensive_guide.md"
        large_doc.write_text(large_content, encoding='utf-8')

        orchestrator = TranslationOrchestrator(
            project_path=tmp_path,
            output_path=tmp_path / "translated"
        )

        # Configure for large document processing
        orchestrator.configure_for_large_documents(
            chunk_size=2000,
            overlap_size=200,
            preserve_context=True
        )

        mock_client = Mock()
        mock_client.generate_content.return_value = "大規模文書の翻訳セクション"
        orchestrator.set_claude_client(mock_client)

        project_result = orchestrator.execute_translation_workflow()

        assert project_result.status == ProjectStatus.COMPLETED
        assert project_result.total_character_count >= 170000
        assert project_result.processing_strategy == "chunked"
        assert project_result.context_preserved is True

    def test_quality_degradation_detection_and_recovery(self, tmp_path):
        """Test detection and recovery from quality degradation."""
        if TranslationOrchestrator is None or ProjectMonitor is None:
            pytest.skip("Components not implemented yet - RED phase")

        orchestrator = TranslationOrchestrator(
            project_path=tmp_path,
            output_path=tmp_path / "translated"
        )

        monitor = ProjectMonitor()
        orchestrator.set_monitor(monitor)

        # Configure monitor for quality degradation detection
        monitor.set_quality_thresholds({
            "terminology_consistency": 98.0,
            "technical_accuracy": 95.0,
            "linguistic_quality": 90.0
        })

        quality_alerts = []
        def alert_handler(alert):
            quality_alerts.append(alert)

        monitor.set_alert_handler(alert_handler)

        # Simulate quality degradation scenario
        mock_client = Mock()

        # First translations are good, then degrade
        translation_count = 0
        def degrading_translate(content, **kwargs):
            nonlocal translation_count
            translation_count += 1
            if translation_count <= 2:
                return "高品質な翻訳結果"
            else:
                return "poor quality translation"  # Intentionally poor

        mock_client.generate_content.side_effect = degrading_translate
        orchestrator.set_claude_client(mock_client)

        # Create sample documents
        for i in range(5):
            doc_path = tmp_path / f"doc_{i}.md"
            doc_path.write_text(f"# Document {i}\n\nContent for document {i}.")

        project_result = orchestrator.execute_translation_workflow()

        # Should detect quality degradation and trigger recovery
        assert len(quality_alerts) > 0
        assert any(alert.type == "quality_degradation" for alert in quality_alerts)
        assert project_result.quality_recovery_triggered is True

    def _create_realistic_170k_document(self) -> str:
        """Create a realistic 170,000+ character technical document."""
        content_parts = [
            "# Comprehensive Technical Documentation\n\n",
            "This document provides comprehensive technical documentation for the system.\n\n"
        ]

        # Add multiple sections to reach 170k characters
        sections = [
            "Architecture Overview",
            "API Reference",
            "Configuration Guide",
            "Deployment Instructions",
            "Security Considerations",
            "Performance Optimization",
            "Troubleshooting Guide",
            "Integration Examples",
            "Best Practices",
            "Migration Guide"
        ]

        for section in sections:
            section_content = f"""
## {section}

This section covers {section.lower()} in detail. The system architecture follows
modern best practices with microservices design, containerization, and cloud-native
deployment strategies.

### Key Components

The main components include:
- Service layer for business logic
- Data access layer for persistence
- API gateway for request routing
- Authentication and authorization services
- Monitoring and logging infrastructure

### Implementation Details

```python
class SystemComponent:
    def __init__(self, config):
        self.config = config
        self.initialized = False

    def initialize(self):
        \"\"\"Initialize the component with configuration.\"\"\"
        try:
            self._setup_connections()
            self._configure_logging()
            self._validate_configuration()
            self.initialized = True
            return True
        except Exception as e:
            self._handle_initialization_error(e)
            return False

    def _setup_connections(self):
        \"\"\"Setup database and external service connections.\"\"\"
        pass

    def _configure_logging(self):
        \"\"\"Configure component-specific logging.\"\"\"
        pass
```

### Configuration Options

The component supports various configuration options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| timeout | int | 30 | Connection timeout in seconds |
| retries | int | 3 | Number of retry attempts |
| debug | bool | false | Enable debug logging |
| cache_size | int | 1000 | Maximum cache entries |

### Error Handling

Error handling is implemented at multiple levels:
1. Connection-level errors with automatic retry
2. Business logic errors with proper exception propagation
3. System-level errors with graceful degradation
4. User-facing errors with helpful messages

""" * 3  # Multiply to increase content size

            content_parts.append(section_content)

        full_content = "".join(content_parts)

        # Ensure we reach at least 170k characters
        while len(full_content) < 170000:
            full_content += "\n\nAdditional content to reach the required character count. " * 100

        return full_content


# These integration tests define the expected end-to-end system behavior
# They will fail initially (RED phase) as the system components don't exist yet
# Implementation will be built to satisfy these integration requirements