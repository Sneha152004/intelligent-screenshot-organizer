"""
Chunker Module
==============
Splits raw OCR text into clean paragraph chunks for downstream processing and indexing.
"""

import re
from typing import List


class ParagraphChunker:
    """
    Paragraph-based text chunker that splits text by paragraph breaks (double newlines)
    and removes empty or whitespace-only chunks.
    """

    def chunk(self, text: str) -> List[str]:
        """
        Splits text into non-empty paragraph chunks.

        Args:
            text: Raw extracted OCR text.

        Returns:
            List of non-empty paragraph strings.
        """
        if not text or not text.strip():
            return []

        # Split on double or multiple newlines (with optional whitespace in between)
        raw_paragraphs = re.split(r"\n\s*\n", text)

        # Strip whitespace and exclude empty paragraphs
        chunks = [p.strip() for p in raw_paragraphs if p and p.strip()]

        return chunks


def chunk_text(text: str) -> List[str]:
    """
    Convenience function for paragraph text chunking.

    Args:
        text: Raw extracted OCR text.

    Returns:
        List of non-empty paragraph strings.
    """
    return ParagraphChunker().chunk(text)
