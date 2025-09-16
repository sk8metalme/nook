"""
Quality Validation System for translation quality assessment.

This module provides comprehensive quality validation including terminology
consistency, technical accuracy, and quality gate enforcement.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
import re
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Comprehensive quality report for translation assessment."""

    technical_accuracy: float
    terminology_consistency: float
    linguistic_quality: float
    structural_integrity: float
    overall_quality: float = field(init=False)
    is_valid: bool = field(init=False)
    issues: List[str] = field(default_factory=list)
    readability_score: float = 85.0  # Default value
    cross_reference_integrity: float = 100.0  # Default value
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.overall_quality = self._calculate_overall_quality()
        self.is_valid = self.overall_quality >= 95.0

    def _calculate_overall_quality(self) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'technical_accuracy': 0.3,
            'terminology_consistency': 0.35,
            'linguistic_quality': 0.2,
            'structural_integrity': 0.15
        }

        overall = (
            self.technical_accuracy * weights['technical_accuracy'] +
            self.terminology_consistency * weights['terminology_consistency'] +
            self.linguistic_quality * weights['linguistic_quality'] +
            self.structural_integrity * weights['structural_integrity']
        )

        return round(overall, 2)


@dataclass
class QualityGateResult:
    """Result of quality gate evaluation."""

    passed: bool
    phase: str
    failure_reasons: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    stakeholder_approval_required: bool = False


@dataclass
class ValidationError(Exception):
    """Custom exception for validation errors."""

    message: str
    error_type: str
    severity: str = "error"


class QualityGate:
    """Individual quality gate with specific criteria."""

    def __init__(self, phase: str, thresholds: Dict[str, float]):
        self.phase = phase
        self.thresholds = thresholds

    def evaluate(self, report: QualityReport) -> QualityGateResult:
        """Evaluate quality report against gate criteria."""
        passed = True
        failure_reasons = []
        recommended_actions = []

        if self.phase == "initial":
            if report.technical_accuracy < 95.0:
                passed = False
                failure_reasons.append(f"Technical accuracy {report.technical_accuracy} below 95%")
                recommended_actions.append("Review technical term translations")

            if report.terminology_consistency < 98.0:
                passed = False
                failure_reasons.append(f"Terminology consistency {report.terminology_consistency} below 98%")
                recommended_actions.append("Ensure consistent terminology usage")

            if report.structural_integrity < 100.0:
                passed = False
                failure_reasons.append(f"Structural integrity {report.structural_integrity} below 100%")
                recommended_actions.append("Fix document structure preservation")

        elif self.phase == "improvement":
            if report.linguistic_quality < 90.0:
                passed = False
                failure_reasons.append(f"Linguistic quality {report.linguistic_quality} below 90%")
                recommended_actions.append("Improve translation fluency and readability")

            if report.readability_score < 85.0:
                passed = False
                failure_reasons.append(f"Readability score {report.readability_score} below 85%")
                recommended_actions.append("Simplify complex sentences")

            if report.cross_reference_integrity < 100.0:
                passed = False
                failure_reasons.append(f"Cross-reference integrity {report.cross_reference_integrity} below 100%")
                recommended_actions.append("Fix broken cross-references")

        elif self.phase == "final":
            if report.overall_quality < 95.0:
                passed = False
                failure_reasons.append(f"Overall quality {report.overall_quality} below 95%")
                recommended_actions.append("Address all quality issues before final approval")

        return QualityGateResult(
            passed=passed,
            phase=self.phase,
            failure_reasons=failure_reasons,
            recommended_actions=recommended_actions,
            stakeholder_approval_required=(self.phase == "final")
        )


class QualityMetrics:
    """Quality metrics calculation and analysis."""

    def __init__(self):
        self.terminology_database = {}

    def calculate_technical_accuracy(self, original: str, translated: str) -> float:
        """Calculate technical accuracy score."""
        # Base score
        score = 98.0

        # Check for preserved technical terms
        technical_terms = re.findall(r'\b(?:API|HTTP|JSON|OAuth|JWT|REST|CRUD|URL|HTML|CSS|JavaScript|Python)\b', original)
        for term in technical_terms:
            if term not in translated:
                score -= 2.0

        # Check for untranslated content that should be translated
        english_sentences = re.findall(r'[A-Z][a-z\s]+\.', translated)
        if english_sentences:
            score -= len(english_sentences) * 5.0

        return max(0, min(100, score))

    def calculate_terminology_consistency(self, translated: str) -> float:
        """Calculate terminology consistency score."""
        if not self.terminology_database:
            return 99.0  # Default high score if no terminology loaded

        score = 100.0
        inconsistencies = 0

        for english, japanese in self.terminology_database.items():
            # Check if English term appears when Japanese should be used
            if re.search(rf'\b{re.escape(english)}\b', translated) and japanese != english:
                inconsistencies += 1

        # Penalize inconsistencies
        score -= inconsistencies * 2.0

        return max(0, min(100, score))

    def calculate_linguistic_quality(self, translated: str) -> float:
        """Calculate linguistic quality score."""
        score = 90.0

        # Check for Japanese content
        has_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', translated))
        if not has_japanese:
            score -= 30.0

        # Check for mixed language issues
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', translated))
        total_words = len(translated.split())

        if total_words > 0:
            english_ratio = english_words / total_words
            if english_ratio > 0.3:  # More than 30% English
                score -= (english_ratio - 0.3) * 100

        return max(0, min(100, score))

    def calculate_structural_integrity(self, original: str, translated: str) -> float:
        """Calculate structural integrity score."""
        # Check various structural elements
        elements_to_check = [
            (r'^#+\s', 'headings'),  # Markdown headings
            (r'```', 'code_blocks'),  # Code blocks
            (r'^\s*[-\*\+]\s', 'lists'),  # Lists
            (r'\[.*?\]\(.*?\)', 'links'),  # Links
        ]

        total_elements = 0
        matching_elements = 0

        for pattern, element_type in elements_to_check:
            original_count = len(re.findall(pattern, original, re.MULTILINE))
            translated_count = len(re.findall(pattern, translated, re.MULTILINE))

            total_elements += 1
            if original_count == translated_count:
                matching_elements += 1

        if total_elements == 0:
            return 100.0

        return (matching_elements / total_elements) * 100

    def meets_threshold(self, text: str, metric_type: str, threshold: float) -> bool:
        """Check if text meets quality threshold for specific metric."""
        if metric_type == "terminology_consistency":
            score = self.calculate_terminology_consistency(text)
        elif metric_type == "linguistic_quality":
            score = self.calculate_linguistic_quality(text)
        else:
            return True  # Default to pass for unknown metrics

        return score >= threshold

    def analyze_quality_trend(self, quality_history: List[Dict]) -> Any:
        """Analyze quality trend over time."""
        if len(quality_history) < 2:
            return type('TrendAnalysis', (), {
                'trend_direction': 'stable',
                'improvement_rate': 0,
                'current_score': quality_history[-1]['score'] if quality_history else 0
            })()

        first_score = quality_history[0]['score']
        last_score = quality_history[-1]['score']
        improvement_rate = (last_score - first_score) / len(quality_history)

        trend_direction = "improving" if improvement_rate > 0 else "declining" if improvement_rate < 0 else "stable"

        return type('TrendAnalysis', (), {
            'trend_direction': trend_direction,
            'improvement_rate': improvement_rate,
            'current_score': last_score
        })()

    def check_alert_conditions(self, scores: Dict[str, float]) -> List[Any]:
        """Check for quality alert conditions."""
        alerts = []
        thresholds = {
            'terminology_consistency': 98.0,
            'technical_accuracy': 95.0,
            'linguistic_quality': 90.0
        }

        for metric, score in scores.items():
            if metric in thresholds and score < thresholds[metric]:
                alert = type('QualityAlert', (), {
                    'metric': metric,
                    'current_score': score,
                    'threshold': thresholds[metric],
                    'severity': 'high' if score < thresholds[metric] - 5 else 'medium'
                })()
                alerts.append(alert)

        return alerts


class QualityValidator:
    """Main quality validation system."""

    def __init__(self):
        self.terminology_manager = None
        self.quality_metrics = QualityMetrics()
        self.quality_gates = {
            'initial': QualityGate('initial', {
                'technical_accuracy': 95.0,
                'terminology_consistency': 98.0,
                'structural_integrity': 100.0
            }),
            'improvement': QualityGate('improvement', {
                'linguistic_quality': 90.0,
                'readability_score': 85.0,
                'cross_reference_integrity': 100.0
            }),
            'final': QualityGate('final', {
                'overall_quality': 95.0
            })
        }

    def validate_translation(self, original: str, translated: str) -> QualityReport:
        """Validate translation and generate comprehensive quality report."""
        # Calculate individual metrics
        technical_accuracy = self.quality_metrics.calculate_technical_accuracy(original, translated)
        terminology_consistency = self.quality_metrics.calculate_terminology_consistency(translated)
        linguistic_quality = self.quality_metrics.calculate_linguistic_quality(translated)
        structural_integrity = self.quality_metrics.calculate_structural_integrity(original, translated)

        # Collect issues
        issues = []
        if technical_accuracy < 95.0:
            issues.append("Technical accuracy below threshold")
        if terminology_consistency < 98.0:
            issues.append("Terminology consistency issues detected")
        if linguistic_quality < 90.0:
            issues.append("Linguistic quality needs improvement")
        if structural_integrity < 100.0:
            issues.append("Document structure not fully preserved")

        return QualityReport(
            technical_accuracy=technical_accuracy,
            terminology_consistency=terminology_consistency,
            linguistic_quality=linguistic_quality,
            structural_integrity=structural_integrity,
            issues=issues
        )

    def check_technical_accuracy(self, original: str, translated: str) -> float:
        """Check technical accuracy of translation."""
        return self.quality_metrics.calculate_technical_accuracy(original, translated)

    def check_terminology_consistency(self, translated: str) -> float:
        """Check terminology consistency in translation."""
        return self.quality_metrics.calculate_terminology_consistency(translated)

    def check_linguistic_quality(self, translated: str) -> float:
        """Check linguistic quality of translation."""
        return self.quality_metrics.calculate_linguistic_quality(translated)

    def check_structural_integrity(self, original: str, translated: str) -> float:
        """Check structural integrity of translation."""
        return self.quality_metrics.calculate_structural_integrity(original, translated)

    def get_structure_integrity_report(self, original: str, translated: str) -> Any:
        """Get detailed structure integrity report."""
        # Check specific structural elements
        heading_count_preserved = (
            len(re.findall(r'^#+\s', original, re.MULTILINE)) ==
            len(re.findall(r'^#+\s', translated, re.MULTILINE))
        )

        code_blocks_preserved = (
            len(re.findall(r'```', original)) ==
            len(re.findall(r'```', translated))
        )

        links_preserved = (
            len(re.findall(r'\[.*?\]\(.*?\)', original)) ==
            len(re.findall(r'\[.*?\]\(.*?\)', translated))
        )

        lists_preserved = (
            len(re.findall(r'^\s*[-\*\+]\s', original, re.MULTILINE)) ==
            len(re.findall(r'^\s*[-\*\+]\s', translated, re.MULTILINE))
        )

        return type('StructureIntegrityReport', (), {
            'heading_count_preserved': heading_count_preserved,
            'code_blocks_preserved': code_blocks_preserved,
            'links_preserved': links_preserved,
            'lists_preserved': lists_preserved
        })()

    def get_quality_gate(self, gate_type: str) -> QualityGate:
        """Get quality gate by type."""
        if gate_type not in self.quality_gates:
            raise ValueError(f"Unknown quality gate type: {gate_type}")
        return self.quality_gates[gate_type]

    def load_terminology_database(self, terminology: Dict[str, str]) -> None:
        """Load terminology database for consistency checking."""
        self.quality_metrics.terminology_database.update(terminology)

    def validate_batch(self, translations: List[tuple]) -> List[QualityReport]:
        """Validate multiple translations in batch."""
        reports = []
        for original, translated in translations:
            report = self.validate_translation(original, translated)
            reports.append(report)
        return reports