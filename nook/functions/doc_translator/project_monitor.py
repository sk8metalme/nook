"""
Project Monitor for translation progress tracking and quality monitoring.

This module provides comprehensive monitoring capabilities including progress
tracking, quality trend analysis, and alert management.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProgressReport:
    """Progress report for translation project."""

    completion_percentage: float
    current_item: int
    total_items: int
    stage: str
    estimated_time_remaining: float = 0.0
    current_quality_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityTrend:
    """Quality trend analysis over time."""

    trend_direction: str  # improving, declining, stable
    current_score: float
    previous_score: float
    change_percentage: float
    trend_confidence: float
    data_points: int


@dataclass
class QualityAlert:
    """Quality alert notification."""

    alert_id: str
    type: str
    severity: AlertSeverity
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""

    overall_health: str  # healthy, warning, critical
    quality_score: float
    performance_score: float
    error_rate: float
    active_alerts: int
    last_update: datetime = field(default_factory=datetime.now)


class AlertSystem:
    """Alert management system."""

    def __init__(self):
        """Initialize alert system."""
        self.alerts = []
        self.alert_handlers = []
        self.alert_thresholds = {
            'terminology_consistency': 98.0,
            'technical_accuracy': 95.0,
            'linguistic_quality': 90.0,
            'structural_integrity': 100.0,
            'processing_speed': 0.1,  # seconds per character
            'error_rate': 5.0  # percentage
        }

    def add_alert_handler(self, handler: Callable[[QualityAlert], None]) -> None:
        """Add alert handler callback."""
        self.alert_handlers.append(handler)

    def trigger_alert(self, alert: QualityAlert) -> None:
        """Trigger an alert and notify handlers."""
        self.alerts.append(alert)
        logger.warning(f"Quality alert triggered: {alert.type} - {alert.message}")

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def check_quality_thresholds(self, metrics: Dict[str, float]) -> List[QualityAlert]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []

        for metric_name, current_value in metrics.items():
            if metric_name in self.alert_thresholds:
                threshold = self.alert_thresholds[metric_name]

                # For most metrics, alert if below threshold
                if current_value < threshold:
                    severity = self._determine_alert_severity(metric_name, current_value, threshold)

                    alert = QualityAlert(
                        alert_id=f"{metric_name}_{datetime.now().timestamp()}",
                        type="quality_degradation",
                        severity=severity,
                        message=f"{metric_name} dropped to {current_value:.1f}% (threshold: {threshold:.1f}%)",
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold_value=threshold
                    )

                    alerts.append(alert)
                    self.trigger_alert(alert)

        return alerts

    def _determine_alert_severity(self, metric_name: str, current_value: float, threshold: float) -> AlertSeverity:
        """Determine alert severity based on deviation from threshold."""
        deviation_percentage = ((threshold - current_value) / threshold) * 100

        if metric_name in ['terminology_consistency', 'technical_accuracy']:
            # Critical metrics
            if deviation_percentage > 5:
                return AlertSeverity.CRITICAL
            elif deviation_percentage > 2:
                return AlertSeverity.HIGH
            else:
                return AlertSeverity.MEDIUM
        else:
            # Non-critical metrics
            if deviation_percentage > 10:
                return AlertSeverity.HIGH
            elif deviation_percentage > 5:
                return AlertSeverity.MEDIUM
            else:
                return AlertSeverity.LOW

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert by ID."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_active_alerts(self) -> List[QualityAlert]:
        """Get all unacknowledged alerts."""
        return [alert for alert in self.alerts if not alert.acknowledged]

    def clear_old_alerts(self, hours: int = 24) -> None:
        """Clear alerts older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]


class ProjectMonitor:
    """Main project monitoring system."""

    def __init__(self):
        """Initialize project monitor."""
        self.progress_history = []
        self.quality_history = []
        self.performance_metrics = {}
        self.alert_system = AlertSystem()
        self._progress_callback = None
        self.quality_thresholds = {}

    def set_progress_callback(self, callback: Callable) -> None:
        """Set progress update callback."""
        self._progress_callback = callback

    def set_alert_handler(self, handler: Callable[[QualityAlert], None]) -> None:
        """Set alert handler."""
        self.alert_system.add_alert_handler(handler)

    def set_quality_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Set quality thresholds for monitoring."""
        self.quality_thresholds.update(thresholds)
        self.alert_system.alert_thresholds.update(thresholds)

    def update_progress(self, progress: ProgressReport) -> None:
        """Update project progress."""
        self.progress_history.append(progress)

        # Trigger progress callback
        if self._progress_callback:
            self._progress_callback(progress)

        # Check for progress-related alerts
        self._check_progress_alerts(progress)

        logger.info(f"Progress updated: {progress.completion_percentage:.1f}% complete")

    def update_quality_metrics(self, metrics: Dict[str, float]) -> None:
        """Update quality metrics and check for alerts."""
        # Record quality history
        quality_entry = {
            'timestamp': datetime.now(),
            'metrics': metrics.copy(),
            'overall_score': sum(metrics.values()) / len(metrics) if metrics else 0.0
        }
        self.quality_history.append(quality_entry)

        # Check quality thresholds
        alerts = self.alert_system.check_quality_thresholds(metrics)

        # Log quality update
        overall_quality = quality_entry['overall_score']
        logger.info(f"Quality metrics updated: overall score {overall_quality:.1f}%")

    def analyze_quality_trend(self, metric_name: str = "overall") -> QualityTrend:
        """Analyze quality trend for a specific metric."""
        if len(self.quality_history) < 2:
            return QualityTrend(
                trend_direction="stable",
                current_score=0.0,
                previous_score=0.0,
                change_percentage=0.0,
                trend_confidence=0.0,
                data_points=len(self.quality_history)
            )

        # Get recent data points
        recent_entries = self.quality_history[-10:]  # Last 10 entries

        if metric_name == "overall":
            current_score = recent_entries[-1]['overall_score']
            previous_score = recent_entries[-2]['overall_score'] if len(recent_entries) >= 2 else current_score
        else:
            current_score = recent_entries[-1]['metrics'].get(metric_name, 0.0)
            previous_score = recent_entries[-2]['metrics'].get(metric_name, 0.0) if len(recent_entries) >= 2 else current_score

        # Calculate change
        change_percentage = ((current_score - previous_score) / max(previous_score, 1.0)) * 100

        # Determine trend direction
        if abs(change_percentage) < 1.0:
            trend_direction = "stable"
        elif change_percentage > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"

        # Calculate confidence based on data consistency
        trend_confidence = min(len(recent_entries) / 10.0, 1.0)

        return QualityTrend(
            trend_direction=trend_direction,
            current_score=current_score,
            previous_score=previous_score,
            change_percentage=change_percentage,
            trend_confidence=trend_confidence,
            data_points=len(self.quality_history)
        )

    def get_system_health_report(self) -> SystemHealthReport:
        """Generate comprehensive system health report."""
        # Calculate overall health
        active_alerts = len(self.alert_system.get_active_alerts())

        if not self.quality_history:
            quality_score = 0.0
        else:
            quality_score = self.quality_history[-1]['overall_score']

        # Determine health status
        if active_alerts == 0 and quality_score >= 95.0:
            health_status = "healthy"
        elif active_alerts <= 2 and quality_score >= 90.0:
            health_status = "warning"
        else:
            health_status = "critical"

        # Calculate performance score (simplified)
        performance_score = max(100.0 - (active_alerts * 10), 0.0)

        # Calculate error rate (simplified)
        error_rate = min(active_alerts * 2.0, 100.0)

        return SystemHealthReport(
            overall_health=health_status,
            quality_score=quality_score,
            performance_score=performance_score,
            error_rate=error_rate,
            active_alerts=active_alerts
        )

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.progress_history:
            return {
                "average_completion_rate": 0.0,
                "estimated_total_time": 0.0,
                "current_throughput": 0.0,
                "efficiency_score": 0.0
            }

        # Calculate statistics from progress history
        recent_progress = self.progress_history[-10:]  # Last 10 updates

        completion_rates = []
        for i in range(1, len(recent_progress)):
            time_diff = (recent_progress[i].timestamp - recent_progress[i-1].timestamp).total_seconds()
            progress_diff = recent_progress[i].completion_percentage - recent_progress[i-1].completion_percentage

            if time_diff > 0:
                rate = progress_diff / time_diff  # percentage per second
                completion_rates.append(rate)

        average_rate = sum(completion_rates) / len(completion_rates) if completion_rates else 0.0
        current_progress = recent_progress[-1].completion_percentage

        # Estimate remaining time
        remaining_progress = 100.0 - current_progress
        estimated_remaining_time = remaining_progress / max(average_rate, 0.001)

        return {
            "average_completion_rate": average_rate,
            "estimated_total_time": estimated_remaining_time,
            "current_throughput": len(recent_progress),
            "efficiency_score": min(max(average_rate * 100, 0), 100)
        }

    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        health_report = self.get_system_health_report()
        performance_stats = self.get_performance_statistics()
        quality_trend = self.analyze_quality_trend()

        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": {
                "status": health_report.overall_health,
                "quality_score": health_report.quality_score,
                "active_alerts": health_report.active_alerts
            },
            "performance": performance_stats,
            "quality_trend": {
                "direction": quality_trend.trend_direction,
                "current_score": quality_trend.current_score,
                "change_percentage": quality_trend.change_percentage
            },
            "alerts": [
                {
                    "type": alert.type,
                    "severity": alert.severity.value,
                    "message": alert.message
                }
                for alert in self.alert_system.get_active_alerts()
            ],
            "recommendations": self._generate_recommendations(health_report, quality_trend)
        }

    def _check_progress_alerts(self, progress: ProgressReport) -> None:
        """Check progress for potential alerts."""
        # Check for stalled progress
        if len(self.progress_history) >= 3:
            recent_progress = [p.completion_percentage for p in self.progress_history[-3:]]
            if len(set(recent_progress)) == 1:  # No progress change
                alert = QualityAlert(
                    alert_id=f"progress_stalled_{datetime.now().timestamp()}",
                    type="progress_stalled",
                    severity=AlertSeverity.MEDIUM,
                    message="Translation progress appears to have stalled",
                    metric_name="progress_rate",
                    current_value=0.0,
                    threshold_value=1.0
                )
                self.alert_system.trigger_alert(alert)

    def _generate_recommendations(self, health_report: SystemHealthReport,
                                quality_trend: QualityTrend) -> List[str]:
        """Generate recommendations based on current state."""
        recommendations = []

        if health_report.overall_health == "critical":
            recommendations.append("Immediate attention required - multiple quality issues detected")
        elif health_report.overall_health == "warning":
            recommendations.append("Monitor quality metrics closely and address any declining trends")

        if quality_trend.trend_direction == "declining":
            recommendations.append("Quality trend is declining - review translation parameters and terminology consistency")

        if health_report.active_alerts > 0:
            recommendations.append("Review and address active quality alerts")

        if not recommendations:
            recommendations.append("System operating within normal parameters")

        return recommendations