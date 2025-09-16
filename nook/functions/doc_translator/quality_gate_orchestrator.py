"""
Quality Gate Orchestrator for managing translation quality gates.

This module manages the sequential execution of quality gates and
provides comprehensive quality assessment across translation workflows.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from .quality_validator import QualityValidator, QualityReport, QualityGate

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of quality gate execution."""

    all_gates_passed: bool
    gate_reports: List[Any] = field(default_factory=list)
    failed_gate: Optional[str] = None
    improvement_suggestions: List[str] = field(default_factory=list)
    execution_time: float = 0.0


@dataclass
class GateExecutionReport:
    """Detailed report of gate execution."""

    gate_name: str
    passed: bool
    execution_time: float
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AggregatedQualityMetrics:
    """Aggregated quality metrics across multiple translations."""

    average_technical_accuracy: float
    average_terminology_consistency: float
    average_linguistic_quality: float
    average_structural_integrity: float
    overall_pass_rate: float
    total_translations_processed: int


class QualityGateOrchestrator:
    """Orchestrator for managing quality gate execution."""

    def __init__(self):
        """Initialize quality gate orchestrator."""
        self.quality_validator = QualityValidator()
        self.execution_history = []

        # Define quality gates in execution order
        self.quality_gates = [
            ("initial", self.quality_validator.get_quality_gate("initial")),
            ("improvement", self.quality_validator.get_quality_gate("improvement")),
            ("final", self.quality_validator.get_quality_gate("final"))
        ]

    def execute_quality_gates(self, translation_data: Dict[str, str]) -> QualityGateResult:
        """Execute all quality gates sequentially."""
        import time

        start_time = time.time()
        gate_reports = []
        all_gates_passed = True
        failed_gate = None
        improvement_suggestions = []

        original = translation_data.get("original", "")
        translated = translation_data.get("translated", "")

        # Execute each gate in sequence
        for gate_name, gate in self.quality_gates:
            # Generate quality report
            quality_report = self.quality_validator.validate_translation(original, translated)

            # Evaluate against gate criteria
            gate_result = gate.evaluate(quality_report)

            # Create execution report
            execution_report = GateExecutionReport(
                gate_name=gate_name,
                passed=gate_result.passed,
                execution_time=0.1,  # Simplified for implementation
                quality_metrics={
                    "technical_accuracy": quality_report.technical_accuracy,
                    "terminology_consistency": quality_report.terminology_consistency,
                    "linguistic_quality": quality_report.linguistic_quality,
                    "structural_integrity": quality_report.structural_integrity
                },
                failure_reasons=gate_result.failure_reasons,
                recommendations=gate_result.recommended_actions
            )

            gate_reports.append(execution_report)

            # Check if gate failed
            if not gate_result.passed:
                all_gates_passed = False
                if failed_gate is None:  # Record first failure
                    failed_gate = gate_name
                improvement_suggestions.extend(gate_result.recommended_actions)

                # Stop execution on first failure
                break

        execution_time = time.time() - start_time

        result = QualityGateResult(
            all_gates_passed=all_gates_passed,
            gate_reports=gate_reports,
            failed_gate=failed_gate,
            improvement_suggestions=improvement_suggestions,
            execution_time=execution_time
        )

        # Record execution
        self.execution_history.append(result)

        return result

    def execute_batch_quality_gates(self, translations: List[Dict[str, str]]) -> List[QualityGateResult]:
        """Execute quality gates for multiple translations."""
        results = []

        for translation_data in translations:
            try:
                result = self.execute_quality_gates(translation_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Quality gate execution failed: {e}")
                # Create failed result
                failed_result = QualityGateResult(
                    all_gates_passed=False,
                    failed_gate="execution_error",
                    improvement_suggestions=[f"Fix execution error: {str(e)}"]
                )
                results.append(failed_result)

        return results

    def aggregate_quality_metrics(self, batch_results: List[QualityGateResult]) -> AggregatedQualityMetrics:
        """Aggregate quality metrics from batch results."""
        if not batch_results:
            return AggregatedQualityMetrics(
                average_technical_accuracy=0.0,
                average_terminology_consistency=0.0,
                average_linguistic_quality=0.0,
                average_structural_integrity=0.0,
                overall_pass_rate=0.0,
                total_translations_processed=0
            )

        # Collect metrics from all reports
        technical_scores = []
        terminology_scores = []
        linguistic_scores = []
        structural_scores = []
        pass_count = 0

        for result in batch_results:
            if result.all_gates_passed:
                pass_count += 1

            # Extract metrics from gate reports
            for report in result.gate_reports:
                if hasattr(report, 'quality_metrics'):
                    metrics = report.quality_metrics
                    technical_scores.append(metrics.get("technical_accuracy", 0.0))
                    terminology_scores.append(metrics.get("terminology_consistency", 0.0))
                    linguistic_scores.append(metrics.get("linguistic_quality", 0.0))
                    structural_scores.append(metrics.get("structural_integrity", 0.0))

        # Calculate averages
        def safe_average(scores):
            return sum(scores) / len(scores) if scores else 0.0

        return AggregatedQualityMetrics(
            average_technical_accuracy=safe_average(technical_scores),
            average_terminology_consistency=safe_average(terminology_scores),
            average_linguistic_quality=safe_average(linguistic_scores),
            average_structural_integrity=safe_average(structural_scores),
            overall_pass_rate=(pass_count / len(batch_results)) * 100,
            total_translations_processed=len(batch_results)
        )

    def get_quality_gate_statistics(self) -> Dict[str, Any]:
        """Get statistics on quality gate performance."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "pass_rate": 0.0,
                "common_failure_reasons": [],
                "average_execution_time": 0.0
            }

        total_executions = len(self.execution_history)
        passed_executions = sum(1 for result in self.execution_history if result.all_gates_passed)
        pass_rate = (passed_executions / total_executions) * 100

        # Collect failure reasons
        failure_reasons = []
        execution_times = []

        for result in self.execution_history:
            execution_times.append(result.execution_time)
            failure_reasons.extend(result.improvement_suggestions)

        # Find most common failure reasons
        from collections import Counter
        common_failures = Counter(failure_reasons).most_common(5)

        return {
            "total_executions": total_executions,
            "pass_rate": pass_rate,
            "common_failure_reasons": [reason for reason, count in common_failures],
            "average_execution_time": sum(execution_times) / len(execution_times)
        }

    def configure_quality_thresholds(self, gate_name: str, thresholds: Dict[str, float]) -> None:
        """Configure quality thresholds for a specific gate."""
        for name, gate in self.quality_gates:
            if name == gate_name:
                gate.thresholds.update(thresholds)
                logger.info(f"Updated thresholds for gate {gate_name}: {thresholds}")
                break
        else:
            logger.warning(f"Gate {gate_name} not found")

    def get_gate_performance_trend(self, gate_name: str) -> Dict[str, Any]:
        """Get performance trend for a specific gate."""
        gate_executions = []

        for result in self.execution_history:
            for report in result.gate_reports:
                if report.gate_name == gate_name:
                    gate_executions.append(report)

        if not gate_executions:
            return {"trend": "no_data", "executions": 0}

        # Calculate trend metrics
        recent_executions = gate_executions[-10:]  # Last 10 executions
        recent_pass_rate = sum(1 for exec in recent_executions if exec.passed) / len(recent_executions) * 100

        all_pass_rate = sum(1 for exec in gate_executions if exec.passed) / len(gate_executions) * 100

        trend_direction = "improving" if recent_pass_rate > all_pass_rate else "declining" if recent_pass_rate < all_pass_rate else "stable"

        return {
            "trend": trend_direction,
            "executions": len(gate_executions),
            "recent_pass_rate": recent_pass_rate,
            "overall_pass_rate": all_pass_rate,
            "improvement_needed": recent_pass_rate < 90.0
        }

    def generate_quality_improvement_plan(self, failed_results: List[QualityGateResult]) -> Dict[str, Any]:
        """Generate improvement plan based on failed quality gate results."""
        if not failed_results:
            return {"status": "no_improvements_needed", "recommendations": []}

        # Analyze failure patterns
        failure_analysis = {}
        all_suggestions = []

        for result in failed_results:
            if result.failed_gate:
                if result.failed_gate not in failure_analysis:
                    failure_analysis[result.failed_gate] = []
                failure_analysis[result.failed_gate].extend(result.improvement_suggestions)
                all_suggestions.extend(result.improvement_suggestions)

        # Prioritize recommendations
        from collections import Counter
        suggestion_counts = Counter(all_suggestions)
        priority_recommendations = [suggestion for suggestion, count in suggestion_counts.most_common()]

        improvement_plan = {
            "status": "improvements_needed",
            "failed_gates": list(failure_analysis.keys()),
            "priority_recommendations": priority_recommendations[:5],  # Top 5
            "gate_specific_actions": failure_analysis,
            "estimated_improvement_time": self._estimate_improvement_time(len(priority_recommendations)),
            "success_probability": self._estimate_success_probability(failure_analysis)
        }

        return improvement_plan

    def _estimate_improvement_time(self, recommendation_count: int) -> str:
        """Estimate time needed for improvements."""
        if recommendation_count <= 2:
            return "1-2 hours"
        elif recommendation_count <= 5:
            return "4-6 hours"
        else:
            return "1-2 days"

    def _estimate_success_probability(self, failure_analysis: Dict[str, List[str]]) -> str:
        """Estimate probability of success after improvements."""
        total_issues = sum(len(issues) for issues in failure_analysis.values())

        if total_issues <= 3:
            return "high"
        elif total_issues <= 7:
            return "medium"
        else:
            return "low"

    def reset_execution_history(self) -> None:
        """Reset execution history (useful for testing)."""
        self.execution_history.clear()
        logger.info("Quality gate execution history reset")