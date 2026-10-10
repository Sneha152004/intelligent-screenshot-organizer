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
from typing import Optional, Tuple


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

    def store_image_in_category(
        self,
        source_path: str,
        screenshot_id: str,
        category: str,
        filename: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """
        Copies source_path to <storage_dir>/<category>/<target_name>.
        Returns tuple: (absolute_target_path, is_newly_copied_boolean).
        Idempotent: If identical file already exists at target_path, skips copy and returns (target_path, False).
        Collision-safe: If different file exists with same filename, appends collision index (_1, _2, etc.).
        """
        src = Path(source_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Source image file not found: {source_path}")

        category_dir = os.path.join(self.storage_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        ext = src.suffix or ".jpg"
        base_name = filename if filename else f"{screenshot_id}{ext}"
        stem = Path(base_name).stem
        target_name = base_name
        target_path = os.path.join(category_dir, target_name)

        # Idempotent & collision safety check
        counter = 1
        while os.path.exists(target_path):
            # Check if source and target are identical file path
            if src.resolve() == Path(target_path).resolve():
                return (os.path.abspath(target_path), False)

            # Check file size & content SHA-256
            if os.path.getsize(src) == os.path.getsize(target_path):
                if self._file_hash(src) == self._file_hash(target_path):
                    return (os.path.abspath(target_path), False)

            # Different content collision -> resolve with suffix
            target_name = f"{stem}_{counter}{ext}"
            target_path = os.path.join(category_dir, target_name)
            counter += 1

        shutil.copy2(src, target_path)
        return (os.path.abspath(target_path), True)

    @staticmethod
    def _file_hash(path: Path) -> str:
        import hashlib
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_image_uri(self, screenshot_id: str) -> Optional[str]:
        # Search root storage_dir and category subdirectories for matching file
        valid_exts = {".jpg", ".png", ".jpeg", ".webp", ".bmp", ".tiff"}
        for root, _, files in os.walk(self.storage_dir):
            for f in files:
                p = Path(f)
                if p.stem == screenshot_id and p.suffix.lower() in valid_exts:
                    return os.path.abspath(os.path.join(root, f))
        return None

    def exists(self, screenshot_id: str) -> bool:
        return self.get_image_uri(screenshot_id) is not None

