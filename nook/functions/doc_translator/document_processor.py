"""
Document Processing System for Markdown document handling.

This module provides comprehensive document parsing, structure analysis,
and cross-reference management for translation workflows.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Individual section within a document."""

    title: str
    content: str
    level: int = 1
    line_start: int = 0
    line_end: int = 0
    section_id: str = ""

    def __post_init__(self):
        """Generate section ID if not provided."""
        if not self.section_id:
            self.section_id = self.title.lower().replace(' ', '_').replace('#', '')


@dataclass
class CodeBlock:
    """Code block within a document."""

    language: str
    content: str
    line_start: int = 0
    line_end: int = 0


@dataclass
class CrossReference:
    """Cross-reference link within or between documents."""

    source_text: str
    target: str
    reference_type: str = "internal"  # internal, external, file
    line_number: int = 0


@dataclass
class StructureIntegrityResult:
    """Result of document structure integrity validation."""

    is_valid: bool
    heading_count_match: bool
    heading_levels_match: bool
    issues: List[str] = field(default_factory=list)


class MarkdownDocument:
    """Parsed Markdown document with structure analysis."""

    def __init__(self, content: str, file_path: str = ""):
        """Initialize markdown document from content."""
        self.content = content
        self.file_path = file_path
        self.sections = []
        self.title = ""
        self.code_blocks = []
        self.cross_references = []

        self._parse_document()

    def _parse_document(self) -> None:
        """Parse document content into structured sections."""
        lines = self.content.split('\n')
        current_section = None
        current_content = []
        line_number = 0

        for line in lines:
            line_number += 1

            # Check for headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content)
                    current_section.line_end = line_number - 1
                    self.sections.append(current_section)

                # Start new section
                level = len(heading_match.group(1))
                title = heading_match.group(2)

                # Set document title from first heading
                if not self.title and level == 1:
                    self.title = title

                current_section = DocumentSection(
                    title=title,
                    content="",
                    level=level,
                    line_start=line_number
                )
                current_content = []
            else:
                # Add to current section content
                current_content.append(line)

        # Save last section
        if current_section:
            current_section.content = '\n'.join(current_content)
            current_section.line_end = line_number
            self.sections.append(current_section)

        # If no sections were found, create a default section
        if not self.sections and self.content.strip():
            # Find title from first line if it exists
            first_lines = self.content.split('\n')[:5]
            title = "Untitled Document"
            for line in first_lines:
                if line.strip() and not line.startswith('#'):
                    title = line.strip()[:50] + "..." if len(line.strip()) > 50 else line.strip()
                    break

            self.sections.append(DocumentSection(
                title=title,
                content=self.content,
                level=1,
                line_start=1,
                line_end=len(self.content.split('\n'))
            ))
            self.title = title

        # Extract code blocks and cross-references
        self._extract_code_blocks()
        self._extract_cross_references()

    def _extract_code_blocks(self) -> None:
        """Extract code blocks from document content."""
        code_block_pattern = r'```(\w*)\n(.*?)\n```'
        matches = re.finditer(code_block_pattern, self.content, re.DOTALL)

        for match in matches:
            language = match.group(1) or 'text'
            content = match.group(2)

            # Calculate line numbers
            start_pos = match.start()
            line_start = self.content[:start_pos].count('\n') + 1
            line_end = line_start + content.count('\n') + 2  # +2 for opening and closing ```

            code_block = CodeBlock(
                language=language,
                content=content,
                line_start=line_start,
                line_end=line_end
            )
            self.code_blocks.append(code_block)

    def _extract_cross_references(self) -> None:
        """Extract cross-references from document content."""
        # Markdown link pattern: [text](url)
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.finditer(link_pattern, self.content)

        for match in matches:
            source_text = match.group(1)
            target = match.group(2)

            # Determine reference type
            if target.startswith('http'):
                ref_type = "external"
            elif target.endswith('.md'):
                ref_type = "file"
            else:
                ref_type = "internal"

            # Calculate line number
            start_pos = match.start()
            line_number = self.content[:start_pos].count('\n') + 1

            cross_ref = CrossReference(
                source_text=source_text,
                target=target,
                reference_type=ref_type,
                line_number=line_number
            )
            self.cross_references.append(cross_ref)

    def get_section_by_title(self, title: str) -> Optional[DocumentSection]:
        """Get section by title."""
        for section in self.sections:
            if section.title == title:
                return section
        return None

    def get_sections_by_level(self, level: int) -> List[DocumentSection]:
        """Get all sections at specific heading level."""
        return [section for section in self.sections if section.level == level]


class DocumentProcessor:
    """Main document processing system."""

    def __init__(self):
        """Initialize document processor."""
        self.processed_documents = {}
        self.cross_reference_map = {}

    def parse_document(self, content: str, file_path: str = "") -> MarkdownDocument:
        """Parse markdown document from content."""
        document = MarkdownDocument(content, file_path)
        if file_path:
            self.processed_documents[file_path] = document
        return document

    def extract_code_blocks(self, document: MarkdownDocument) -> List[CodeBlock]:
        """Extract all code blocks from document."""
        return document.code_blocks

    def extract_cross_references(self, document: MarkdownDocument) -> List[CrossReference]:
        """Extract all cross-references from document."""
        return document.cross_references

    def validate_structure_integrity(self, original_content: str,
                                   translated_content: str) -> StructureIntegrityResult:
        """Validate that document structure is preserved in translation."""
        original_doc = MarkdownDocument(original_content)
        translated_doc = MarkdownDocument(translated_content)

        issues = []
        is_valid = True

        # Check heading counts
        original_headings = len(original_doc.sections)
        translated_headings = len(translated_doc.sections)
        heading_count_match = original_headings == translated_headings

        if not heading_count_match:
            issues.append(f"Heading count mismatch: original {original_headings}, translated {translated_headings}")
            is_valid = False

        # Check heading levels
        original_levels = [section.level for section in original_doc.sections]
        translated_levels = [section.level for section in translated_doc.sections]
        heading_levels_match = original_levels == translated_levels

        if not heading_levels_match:
            issues.append("heading level mismatch detected")
            is_valid = False

        # Check code block preservation
        original_code_count = len(original_doc.code_blocks)
        translated_code_count = len(translated_doc.code_blocks)

        if original_code_count != translated_code_count:
            issues.append(f"Code block count mismatch: original {original_code_count}, translated {translated_code_count}")
            is_valid = False

        # Check cross-reference preservation
        original_ref_count = len(original_doc.cross_references)
        translated_ref_count = len(translated_doc.cross_references)

        if original_ref_count != translated_ref_count:
            issues.append(f"Cross-reference count mismatch: original {original_ref_count}, translated {translated_ref_count}")
            is_valid = False

        return StructureIntegrityResult(
            is_valid=is_valid,
            heading_count_match=heading_count_match,
            heading_levels_match=heading_levels_match,
            issues=issues
        )

    def analyze_document_structure(self, document: MarkdownDocument) -> Dict[str, Any]:
        """Analyze document structure and return detailed metrics."""
        return {
            'total_sections': len(document.sections),
            'heading_levels': list(set(section.level for section in document.sections)),
            'code_blocks': len(document.code_blocks),
            'cross_references': len(document.cross_references),
            'document_title': document.title,
            'structure_complexity': self._calculate_structure_complexity(document)
        }

    def _calculate_structure_complexity(self, document: MarkdownDocument) -> str:
        """Calculate structure complexity rating."""
        complexity_score = 0

        # Add points for different structural elements
        complexity_score += len(document.sections) * 1
        complexity_score += len(document.code_blocks) * 2
        complexity_score += len(document.cross_references) * 1

        # Factor in heading level depth
        max_level = max((section.level for section in document.sections), default=0)
        complexity_score += max_level * 2

        if complexity_score < 10:
            return "simple"
        elif complexity_score < 25:
            return "moderate"
        else:
            return "complex"

    def extract_terminology_context(self, document: MarkdownDocument) -> Dict[str, int]:
        """Extract terminology context from document."""
        context_patterns = {
            "programming": r'\b(?:code|function|class|method|variable|API|framework|library)\b',
            "web_api": r'\b(?:endpoint|REST|HTTP|JSON|API|request|response|server)\b',
            "security": r'\b(?:authentication|authorization|token|key|certificate|encrypt)\b',
            "database": r'\b(?:database|query|table|schema|index|transaction|SQL)\b',
            "deployment": r'\b(?:deploy|deployment|container|docker|kubernetes|cloud)\b'
        }

        context_counts = {}
        full_content = document.content.lower()

        for context, pattern in context_patterns.items():
            matches = re.findall(pattern, full_content, re.IGNORECASE)
            context_counts[context] = len(matches)

        return context_counts

    def build_cross_reference_map(self, documents: List[MarkdownDocument]) -> Dict[str, List[str]]:
        """Build cross-reference map between documents."""
        reference_map = {}

        for document in documents:
            doc_refs = []
            for ref in document.cross_references:
                if ref.reference_type == "file":
                    doc_refs.append(ref.target)

            if document.file_path:
                reference_map[document.file_path] = doc_refs

        return reference_map

    def validate_cross_reference_integrity(self, documents: List[MarkdownDocument]) -> Dict[str, Any]:
        """Validate cross-reference integrity across documents."""
        reference_map = self.build_cross_reference_map(documents)
        available_files = {doc.file_path for doc in documents if doc.file_path}

        broken_references = []
        total_references = 0

        for source_file, references in reference_map.items():
            for ref_target in references:
                total_references += 1
                # Remove anchors and normalize path
                clean_target = ref_target.split('#')[0]
                if clean_target not in available_files:
                    broken_references.append({
                        'source': source_file,
                        'target': clean_target,
                        'type': 'missing_file'
                    })

        return {
            'total_references': total_references,
            'broken_references': len(broken_references),
            'integrity_score': ((total_references - len(broken_references)) / max(total_references, 1)) * 100,
            'broken_reference_details': broken_references
        }

    def extract_translation_units(self, document: MarkdownDocument) -> List[Dict[str, Any]]:
        """Extract translatable units from document."""
        units = []

        for i, section in enumerate(document.sections):
            # Skip code-heavy sections
            code_density = section.content.count('```') / max(len(section.content), 1)
            if code_density > 0.3:  # Skip sections that are >30% code
                continue

            unit = {
                'unit_id': f"section_{i}",
                'type': 'section',
                'title': section.title,
                'content': section.content.strip(),
                'level': section.level,
                'context': self._determine_section_context(section),
                'priority': self._determine_translation_priority(section)
            }
            units.append(unit)

        return units

    def _determine_section_context(self, section: DocumentSection) -> str:
        """Determine the context/domain of a section."""
        content_lower = section.content.lower()

        if any(term in content_lower for term in ['api', 'endpoint', 'http', 'rest']):
            return 'api_reference'
        elif any(term in content_lower for term in ['install', 'setup', 'configure', 'deployment']):
            return 'setup_guide'
        elif any(term in content_lower for term in ['example', 'tutorial', 'how to']):
            return 'tutorial'
        elif any(term in content_lower for term in ['architecture', 'design', 'system']):
            return 'technical_design'
        else:
            return 'general'

    def _determine_translation_priority(self, section: DocumentSection) -> str:
        """Determine translation priority based on section characteristics."""
        if section.level == 1:  # Main headings are high priority
            return 'high'
        elif 'important' in section.content.lower() or 'note' in section.content.lower():
            return 'high'
        elif section.level <= 3:  # Sub-headings are medium priority
            return 'medium'
        else:
            return 'low'