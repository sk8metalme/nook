"""
Terminology Management System for translation consistency.

This module provides comprehensive terminology management including
database operations, consistency checking, and auto-correction.
"""

import logging
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TerminologyEntry:
    """Single terminology entry with metadata."""

    english: str
    japanese: str
    context: str
    priority: str
    usage_notes: str = ""

    def __post_init__(self):
        """Validate terminology entry."""
        if self.priority not in ["critical", "high", "medium", "low"]:
            raise ValueError(f"Invalid priority level: {self.priority}")


@dataclass
class TerminologyConsistencyReport:
    """Report on terminology consistency across documents."""

    overall_consistency: float
    inconsistencies: List[Any] = field(default_factory=list)
    total_terms_checked: int = 0
    consistent_terms: int = 0

    def __post_init__(self):
        """Calculate derived fields."""
        if self.total_terms_checked == 0:
            self.overall_consistency = 100.0
        else:
            self.overall_consistency = (self.consistent_terms / self.total_terms_checked) * 100


class TerminologyDatabase:
    """Database for managing terminology entries."""

    def __init__(self, terminology_data: Dict[str, Any] = None):
        """Initialize terminology database."""
        self._terms = {}
        self._load_terminology_data(terminology_data or {})

    def _load_terminology_data(self, data: Dict[str, Any]) -> None:
        """Load terminology data from dictionary."""
        for english, term_data in data.items():
            if isinstance(term_data, str):
                # Simple string mapping
                self._terms[english] = TerminologyEntry(
                    english=english,
                    japanese=term_data,
                    context="general",
                    priority="medium"
                )
            elif isinstance(term_data, dict):
                # Detailed term data
                self._terms[english] = TerminologyEntry(
                    english=english,
                    japanese=term_data.get("japanese", english),
                    context=term_data.get("context", "general"),
                    priority=term_data.get("priority", "medium"),
                    usage_notes=term_data.get("usage_notes", "")
                )

    @property
    def term_count(self) -> int:
        """Get total number of terms in database."""
        return len(self._terms)

    def get_term(self, english: str) -> Optional[TerminologyEntry]:
        """Get terminology entry by English term."""
        return self._terms.get(english)

    def add_term(self, entry: TerminologyEntry) -> None:
        """Add terminology entry to database."""
        self._terms[entry.english] = entry

    def export_to_yaml(self) -> str:
        """Export terminology database to YAML format."""
        export_data = {}
        for english, entry in self._terms.items():
            export_data[english] = {
                "japanese": entry.japanese,
                "context": entry.context,
                "priority": entry.priority,
                "usage_notes": entry.usage_notes
            }
        return yaml.dump(export_data, allow_unicode=True, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_content: str) -> 'TerminologyDatabase':
        """Create terminology database from YAML content."""
        data = yaml.safe_load(yaml_content)
        return cls(data)

    def get_terms_by_context(self, context: str) -> List[TerminologyEntry]:
        """Get all terms for a specific context."""
        return [entry for entry in self._terms.values() if entry.context == context]

    def get_critical_terms(self) -> List[TerminologyEntry]:
        """Get all critical priority terms."""
        return [entry for entry in self._terms.values() if entry.priority == "critical"]


class TerminologyManager:
    """Main terminology management system."""

    def __init__(self, terminology_data: Dict[str, Any] = None):
        """Initialize terminology manager."""
        self.database = TerminologyDatabase(terminology_data)
        self._context_patterns = {
            "programming": r"\b(?:code|function|class|method|variable|API|framework)\b",
            "web_api": r"\b(?:endpoint|REST|HTTP|JSON|API|request|response)\b",
            "security": r"\b(?:authentication|authorization|token|key|certificate|encrypt)\b",
            "database": r"\b(?:database|query|table|schema|index|transaction)\b"
        }

    def detect_terms_in_text(self, text: str) -> List[TerminologyEntry]:
        """Detect terminology entries present in text."""
        detected = []
        text_lower = text.lower()

        for english, entry in self.database._terms.items():
            # Check for exact word boundaries
            if re.search(rf'\b{re.escape(english.lower())}\b', text_lower):
                detected.append(entry)

        return detected

    def check_consistency_across_documents(self, documents: List[str]) -> TerminologyConsistencyReport:
        """Check terminology consistency across multiple documents."""
        term_usage = defaultdict(set)
        total_terms_checked = 0
        consistent_terms = 0
        inconsistencies = []

        # Analyze each document
        for doc_index, document in enumerate(documents):
            detected_terms = self.detect_terms_in_text(document)

            for term in detected_terms:
                # Check how the term is used in this document
                japanese_usage = term.japanese in document
                english_usage = term.english in document

                usage_pattern = (japanese_usage, english_usage)
                term_usage[term.english].add((doc_index, usage_pattern))

        # Check for consistency
        for english_term, usage_set in term_usage.items():
            total_terms_checked += 1
            usage_patterns = [pattern for _, pattern in usage_set]

            # If all documents use the same pattern, it's consistent
            if len(set(usage_patterns)) <= 1:
                consistent_terms += 1
            else:
                # Create inconsistency record
                inconsistency = type('TerminologyInconsistency', (), {
                    'term': english_term,
                    'document_usages': dict(usage_set),
                    'severity': 'high' if self.database.get_term(english_term).priority == 'critical' else 'medium'
                })()
                inconsistencies.append(inconsistency)

        return TerminologyConsistencyReport(
            overall_consistency=0.0,  # Will be calculated in __post_init__
            inconsistencies=inconsistencies,
            total_terms_checked=total_terms_checked,
            consistent_terms=consistent_terms
        )

    def auto_correct_terminology(self, text: str) -> str:
        """Automatically correct terminology inconsistencies in text."""
        corrected_text = text

        for english, entry in self.database._terms.items():
            # Apply case-insensitive correction while preserving intended usage
            if entry.japanese != english:  # Only correct if translation differs
                # Replace English with Japanese equivalent
                pattern = rf'\b{re.escape(english)}\b'
                corrected_text = re.sub(pattern, entry.japanese, corrected_text, flags=re.IGNORECASE)

        return corrected_text

    def analyze_terminology_context(self, text: str) -> Any:
        """Analyze terminology context in text."""
        detected_contexts = set()

        for context, pattern in self._context_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                detected_contexts.add(context)

        return type('TerminologyContextAnalysis', (), {
            'detected_contexts': list(detected_contexts),
            'primary_context': list(detected_contexts)[0] if detected_contexts else 'general',
            'context_confidence': len(detected_contexts) / len(self._context_patterns)
        })()

    def validate_critical_terminology(self, text: str) -> Any:
        """Validate that critical terminology is correctly used."""
        violations = []
        critical_terms = self.database.get_critical_terms()

        for term in critical_terms:
            # Check if critical term appears in incorrect form
            if term.english.lower() in text.lower():
                # Check if it should be translated but isn't
                if term.japanese != term.english and term.japanese not in text:
                    violation = type('CriticalTermViolation', (), {
                        'term': term.english,
                        'expected': term.japanese,
                        'priority': term.priority,
                        'context': term.context
                    })()
                    violations.append(violation)

        return type('CriticalTerminologyReport', (), {
            'violations': violations,
            'critical_terms_checked': len(critical_terms),
            'violations_found': len(violations)
        })()

    def get_terminology_suggestions(self, text: str) -> List[Dict[str, str]]:
        """Get terminology suggestions for text."""
        suggestions = []
        detected_terms = self.detect_terms_in_text(text)

        for term in detected_terms:
            # Suggest proper usage based on database entry
            suggestion = {
                'english': term.english,
                'suggested_japanese': term.japanese,
                'context': term.context,
                'priority': term.priority,
                'usage_notes': term.usage_notes
            }
            suggestions.append(suggestion)

        return suggestions

    def update_terminology_database(self, updates: Dict[str, Any]) -> None:
        """Update terminology database with new entries or modifications."""
        for english, term_data in updates.items():
            if isinstance(term_data, dict):
                entry = TerminologyEntry(
                    english=english,
                    japanese=term_data.get("japanese", english),
                    context=term_data.get("context", "general"),
                    priority=term_data.get("priority", "medium"),
                    usage_notes=term_data.get("usage_notes", "")
                )
                self.database.add_term(entry)

    def export_terminology_report(self) -> Dict[str, Any]:
        """Export comprehensive terminology report."""
        return {
            'total_terms': self.database.term_count,
            'critical_terms': len(self.database.get_critical_terms()),
            'contexts': list(self._context_patterns.keys()),
            'database_health': 'healthy' if self.database.term_count > 0 else 'empty'
        }