"""
Documentation Translation System for Nook.

This module provides comprehensive translation capabilities for technical documentation,
including quality control, terminology management, and progress monitoring.
"""

from .translation_engine import TranslationEngine, TranslationConfig, TranslationResult, TranslationQualityReport
from .quality_validator import QualityValidator, QualityReport, QualityGate, QualityMetrics, ValidationError
from .terminology_manager import TerminologyManager, TerminologyDatabase, TerminologyEntry, TerminologyConsistencyReport
from .document_processor import DocumentProcessor, MarkdownDocument, DocumentSection, CodeBlock, CrossReference
from .translation_orchestrator import TranslationOrchestrator, TranslationProject, ProjectStatus, TranslationWorkflow
from .quality_gate_orchestrator import QualityGateOrchestrator, QualityGateResult, GateExecutionReport
from .project_monitor import ProjectMonitor, ProgressReport, QualityTrend, AlertSystem

__version__ = "1.0.0"
__author__ = "Nook Translation System"

__all__ = [
    # Core translation components
    "TranslationEngine",
    "TranslationConfig",
    "TranslationResult",
    "TranslationQualityReport",

    # Quality validation
    "QualityValidator",
    "QualityReport",
    "QualityGate",
    "QualityMetrics",
    "ValidationError",

    # Terminology management
    "TerminologyManager",
    "TerminologyDatabase",
    "TerminologyEntry",
    "TerminologyConsistencyReport",

    # Document processing
    "DocumentProcessor",
    "MarkdownDocument",
    "DocumentSection",
    "CodeBlock",
    "CrossReference",

    # Orchestration
    "TranslationOrchestrator",
    "TranslationProject",
    "ProjectStatus",
    "TranslationWorkflow",

    # Quality gates
    "QualityGateOrchestrator",
    "QualityGateResult",
    "GateExecutionReport",

    # Monitoring
    "ProjectMonitor",
    "ProgressReport",
    "QualityTrend",
    "AlertSystem",
]