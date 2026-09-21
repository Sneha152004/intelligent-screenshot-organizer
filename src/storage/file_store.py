"""
File Storage Layer (Layer 1)
=============================
Abstracts raw screenshot image file storage operations.
Enables local filesystem storage for prototype and cloud object storage (S3, R2, MinIO) in the future.
"""

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseFileStore(ABC):
    """Abstract Base Class for Image / Object Storage."""

    @abstractmethod
    def store_image(self, source_path: str, screenshot_id: str) -> str:
        """
        Registers/stores an image file and returns its URI or canonical path.

        Args:
            source_path: Source image file path.
            screenshot_id: Unique screenshot identifier.

        Returns:
            Canonical image URI or absolute path.
        """
        pass

    @abstractmethod
    def get_image_uri(self, screenshot_id: str) -> Optional[str]:
        """
        Resolves a screenshot_id to its image URI or path.

        Args:
            screenshot_id: Unique screenshot identifier.

        Returns:
            Image URI string if found, else None.
        """
        pass

    @abstractmethod
    def exists(self, screenshot_id: str) -> bool:
        """Checks whether an image file exists in storage."""
        pass


class LocalFileStore(BaseFileStore):
    """Local filesystem implementation of screenshot object storage."""

    def __init__(self, storage_dir: str = "data/screenshots"):
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)

    def store_image(self, source_path: str, screenshot_id: str) -> str:
        src = Path(source_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Source image file not found: {source_path}")

        ext = src.suffix or ".jpg"
        target_name = f"{screenshot_id}{ext}"
        target_path = os.path.join(self.storage_dir, target_name)

        # If source is already at target_path, no need to copy
        if src.resolve() != Path(target_path).resolve():
            shutil.copy2(src, target_path)

        return os.path.abspath(target_path)

    def get_image_uri(self, screenshot_id: str) -> Optional[str]:
        # Search for file with matching screenshot_id stem in storage directory
        for ext in [".jpg", ".png", ".jpeg", ".webp"]:
            candidate = os.path.join(self.storage_dir, f"{screenshot_id}{ext}")
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        return None

    def exists(self, screenshot_id: str) -> bool:
        return self.get_image_uri(screenshot_id) is not None
