"""
Translation Engine for documentation translation using Claude API.

This module provides the core translation functionality with quality control,
terminology consistency, and structure preservation.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
import re

from nook.functions.common.python.claude_client import ClaudeClient, ClaudeClientConfig

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    """Configuration for translation engine."""

    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.1  # Low temperature for consistency
    max_tokens: int = 8192
    system_prompt: str = field(default_factory=lambda: """
あなたは専門技術翻訳者です。以下の原則に従って英語の技術文書を正確な日本語に翻訳してください：

1. 技術用語の統一性を保つ
2. Markdownの構造を完全に保持する
3. コードブロックは翻訳せず、そのまま保持する
4. リンクや参照は構造を保持して翻訳する
5. 自然で読みやすい日本語にする
6. 専門用語は適切に日本語化するか、英語のまま保持する

翻訳のみを返し、説明や追加のコメントは含めないでください。
""")
    quality_threshold: float = 95.0
    terminology_database: Dict[str, str] = field(default_factory=dict)

    def update(self, **kwargs) -> None:
        """Update configuration with given parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")


@dataclass
class TranslationResult:
    """Result of a translation operation."""

    original_text: str
    translated_text: str
    quality_score: float
    terminology_matches: Dict[str, str] = field(default_factory=dict)
    structure_preserved: bool = True
    processing_time: float = 0.0
    context_used: bool = False


@dataclass
class TranslationQualityReport:
    """Quality report for translation assessment."""

    technical_accuracy: float
    terminology_consistency: float
    linguistic_quality: float
    structural_integrity: float

    def __post_init__(self):
        """Validate quality metrics are within valid ranges."""
        for field_name in ['technical_accuracy', 'terminology_consistency',
                          'linguistic_quality', 'structural_integrity']:
            value = getattr(self, field_name)
            if not (0 <= value <= 100):
                raise ValueError(f"{field_name} must be between 0 and 100, got {value}")

    def calculate_overall_score(self) -> float:
        """Calculate weighted overall quality score."""
        # Weighted average based on importance
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

    def passes_quality_gate(self, gate_type: str) -> bool:
        """Check if translation passes quality gate requirements."""
        if gate_type == "initial":
            return (
                self.technical_accuracy >= 95.0 and
                self.terminology_consistency >= 98.0 and
                self.structural_integrity >= 100.0
            )
        elif gate_type == "improvement":
            return (
                self.linguistic_quality >= 90.0 and
                self.calculate_overall_score() >= 92.0
            )
        elif gate_type == "final":
            return self.calculate_overall_score() >= 95.0
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")


class TranslationEngine:
    """Core translation engine using Claude API."""

    def __init__(self, claude_client: ClaudeClient = None, config: TranslationConfig = None):
        """Initialize translation engine."""
        self.claude_client = claude_client or ClaudeClient(
            config=ClaudeClientConfig(
                model="claude-3-5-sonnet-20241022",
                temperature=0.1,
                max_output_tokens=8192
            )
        )
        self.config = config or TranslationConfig()
        self.terminology_database = {}

    def translate_text(self, text: str) -> str:
        """Translate text while preserving structure."""
        try:
            # Apply terminology preprocessing
            processed_text = self._preprocess_text(text)

            # Perform translation
            translated = self.claude_client.generate_content(
                processed_text,
                system_instruction=self.config.system_prompt,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens
            )

            # Post-process to ensure consistency
            final_text = self._postprocess_translation(text, translated)

            return final_text

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise Exception(f"Translation failed: {str(e)}")

    def translate_with_context(self, text: str, previous_context: str = "",
                             next_context: str = "") -> str:
        """Translate text with surrounding context for better consistency."""
        context_prompt = f"""
前のセクション: {previous_context[:200]}...
翻訳対象: {text}
次のセクション: {next_context[:200]}...

上記の文脈を考慮して、翻訳対象のテキストを翻訳してください。
"""

        return self.claude_client.generate_content(
            context_prompt,
            system_instruction=self.config.system_prompt,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens
        )

    def translate_document_section(self, text: str) -> TranslationResult:
        """Translate a document section with quality assessment."""
        import time

        start_time = time.time()

        # Perform translation
        translated = self.translate_text(text)

        # Calculate quality metrics
        quality_score = self._calculate_quality_score(text, translated)

        # Check terminology matches
        terminology_matches = self._check_terminology_matches(translated)

        # Verify structure preservation
        structure_preserved = self._verify_structure_preservation(text, translated)

        processing_time = time.time() - start_time

        result = TranslationResult(
            original_text=text,
            translated_text=translated,
            quality_score=quality_score,
            terminology_matches=terminology_matches,
            structure_preserved=structure_preserved,
            processing_time=processing_time
        )

        # Enforce quality threshold
        if quality_score < self.config.quality_threshold:
            raise Exception(f"Translation quality {quality_score} below threshold {self.config.quality_threshold}")

        return result

    def translate_batch(self, sections: List[str],
                       progress_callback: Optional[Callable] = None) -> List[TranslationResult]:
        """Translate multiple sections with progress tracking."""
        results = []
        total = len(sections)

        for i, section in enumerate(sections):
            if progress_callback:
                progress_callback(i, total, f"section_{i}")

            result = self.translate_document_section(section)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, f"section_{i}")

        return results

    def load_terminology_database(self, terminology: Dict[str, str]) -> None:
        """Load terminology database for consistency enforcement."""
        self.terminology_database.update(terminology)
        self.config.terminology_database.update(terminology)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before translation."""
        # Preserve code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        preserved_text = text

        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            preserved_text = preserved_text.replace(block, placeholder, 1)

        return preserved_text

    def _postprocess_translation(self, original: str, translated: str) -> str:
        """Post-process translation to restore preserved elements."""
        # Restore code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', original)
        result = translated

        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            if placeholder in result:
                result = result.replace(placeholder, block, 1)
            elif len(code_blocks) > i:
                # If placeholder not found but code block exists, preserve it
                if block not in result:
                    result = result + "\n\n" + block

        # Apply terminology consistency
        for english, japanese in self.terminology_database.items():
            # Simple replacement - in production, would need more sophisticated NLP
            result = re.sub(rf'\b{re.escape(english)}\b', japanese, result, flags=re.IGNORECASE)

        return result

    def _calculate_quality_score(self, original: str, translated: str) -> float:
        """Calculate quality score based on various metrics."""
        # Simple quality calculation - in production would use more sophisticated metrics
        score = 95.0  # Base score

        # Check for untranslated English (should penalize)
        english_words = re.findall(r'[a-zA-Z]+', translated)
        # Allow technical terms and code
        allowed_english = ['API', 'HTTP', 'JSON', 'OAuth', 'JWT', 'URL', 'HTML', 'CSS', 'JavaScript']
        problematic_english = [word for word in english_words
                             if word not in allowed_english and len(word) > 2]

        if problematic_english:
            score -= min(10, len(problematic_english) * 2)

        # Check for Japanese content
        has_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', translated))
        if not has_japanese:
            score -= 20

        return max(0, min(100, score))

    def _check_terminology_matches(self, translated: str) -> Dict[str, str]:
        """Check for terminology consistency in translation."""
        matches = {}
        for english, japanese in self.terminology_database.items():
            if japanese in translated:
                matches[english] = japanese
        return matches

    def _verify_structure_preservation(self, original: str, translated: str) -> bool:
        """Verify that document structure is preserved."""
        # Check heading count
        original_headings = len(re.findall(r'^#+\s', original, re.MULTILINE))
        translated_headings = len(re.findall(r'^#+\s', translated, re.MULTILINE))

        # Check code block count
        original_code_blocks = len(re.findall(r'```', original))
        translated_code_blocks = len(re.findall(r'```', translated))

        # Check list count
        original_lists = len(re.findall(r'^\s*[-\*\+]\s', original, re.MULTILINE))
        translated_lists = len(re.findall(r'^\s*[-\*\+]\s', translated, re.MULTILINE))

        # Structure is preserved if counts match
        return (original_headings == translated_headings and
                original_code_blocks == translated_code_blocks and
                original_lists == translated_lists)