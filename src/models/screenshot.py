"""
Screenshot Model
================
Represents a raw screenshot image file and its physical storage reference.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Screenshot:
    """Represents a screenshot file entity."""

    screenshot_id: str
    filename: str
    image_uri: str
    file_size_bytes: Optional[int] = None
