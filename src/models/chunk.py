"""
Screenshot Chunk Model
======================
Represents a text chunk extracted from a screenshot image and its vector embedding.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ScreenshotChunk:
    """Represents a text chunk belonging to a screenshot."""

    chunk_id: str
    screenshot_id: str
    chunk_index: int
    text: str
    category: str = "General"
    embedding: Optional[List[float]] = None
