"""
Duplicate Screenshot Detector Module
====================================
Provides exact duplicate detection (SHA-256) and near-duplicate visual similarity
detection using perceptual hashing (pHash) and difference hashing (dHash) with
Hamming distance comparison.

Key capabilities:
- SHA-256 exact content hash computation.
- 64-bit DCT perceptual hash (pHash) and difference hash (dHash).
- Configurable Hamming distance threshold for visual similarity.
- Duplicate clustering without automatic deletion.
- Graceful handling of missing or corrupted image files.
"""

import hashlib
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    from scipy.fftpack import dct
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """
    Reusable engine for exact and near-duplicate screenshot detection.
    """

    def __init__(self, default_max_distance: int = 6):
        """
        Args:
            default_max_distance: Maximum Hamming distance to consider two images
                                  near-duplicates (out of 64 bits). Default 6 (~90.6% similarity).
        """
        self.default_max_distance = default_max_distance

    @staticmethod
    def compute_sha256(image_source: Union[str, Path, bytes, io.BytesIO]) -> Optional[str]:
        """
        Computes SHA-256 hex digest for an image file or raw bytes.
        Returns 64-character lowercase hexadecimal string, or None if reading fails.
        """
        try:
            h = hashlib.sha256()
            if isinstance(image_source, (str, Path)):
                p = Path(image_source)
                if not p.is_file():
                    return None
                with open(p, "rb") as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
            elif isinstance(image_source, bytes):
                h.update(image_source)
            elif hasattr(image_source, "read"):
                curr_pos = image_source.tell() if hasattr(image_source, "tell") else None
                while chunk := image_source.read(65536):
                    h.update(chunk)
                if curr_pos is not None and hasattr(image_source, "seek"):
                    image_source.seek(curr_pos)
            else:
                return None
            return h.hexdigest().lower()
        except Exception as e:
            logger.warning(f"Failed to compute SHA-256 hash: {e}")
            return None

    @staticmethod
    def compute_dhash(
        image_source: Union[str, Path, bytes, io.BytesIO, Image.Image],
        hash_size: int = 8,
    ) -> Optional[str]:
        """
        Computes 64-bit difference hash (dHash) based on horizontal luminance gradients.
        Returns 16-character hexadecimal string.
        """
        try:
            if isinstance(image_source, Image.Image):
                img = image_source
                should_close = False
            else:
                if isinstance(image_source, (str, Path)):
                    p = Path(image_source)
                    if not p.is_file():
                        return None
                    img = Image.open(p)
                elif isinstance(image_source, bytes):
                    img = Image.open(io.BytesIO(image_source))
                elif hasattr(image_source, "read"):
                    img = Image.open(image_source)
                else:
                    return None
                should_close = True

            try:
                # Resize to (width = hash_size + 1, height = hash_size)
                resized = img.convert("L").resize(
                    (hash_size + 1, hash_size), Image.Resampling.LANCZOS
                )
                pixels = np.array(resized, dtype=np.int32)
                # Compare adjacent pixels
                diff = pixels[:, 1:] > pixels[:, :-1]
                val = 0
                for bit in diff.flatten():
                    val = (val << 1) | int(bit)
                return f"{val:016x}"
            finally:
                if should_close:
                    img.close()
        except Exception as e:
            logger.warning(f"Failed to compute dHash: {e}")
            return None

    @classmethod
    def compute_phash(
        cls,
        image_source: Union[str, Path, bytes, io.BytesIO, Image.Image],
        hash_size: int = 8,
        highfreq_factor: int = 4,
    ) -> Optional[str]:
        """
        Computes 64-bit Discrete Cosine Transform (DCT) perceptual hash (pHash).
        Falls back to dHash if scipy is unavailable.
        Returns 16-character hexadecimal string.
        """
        if not _HAS_SCIPY:
            return cls.compute_dhash(image_source, hash_size=hash_size)

        try:
            if isinstance(image_source, Image.Image):
                img = image_source
                should_close = False
            else:
                if isinstance(image_source, (str, Path)):
                    p = Path(image_source)
                    if not p.is_file():
                        return None
                    img = Image.open(p)
                elif isinstance(image_source, bytes):
                    img = Image.open(io.BytesIO(image_source))
                elif hasattr(image_source, "read"):
                    img = Image.open(image_source)
                else:
                    return None
                should_close = True

            try:
                img_size = hash_size * highfreq_factor  # 32
                resized = img.convert("L").resize(
                    (img_size, img_size), Image.Resampling.LANCZOS
                )
                pixels = np.array(resized, dtype=np.float32)

                # 2D Discrete Cosine Transform
                dct_matrix = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
                # Top-left hash_size x hash_size low frequencies (skip DC [0,0] for variance)
                sub_matrix = dct_matrix[:hash_size, :hash_size]
                med = np.median(sub_matrix)
                diff = sub_matrix > med
                val = 0
                for bit in diff.flatten():
                    val = (val << 1) | int(bit)
                return f"{val:016x}"
            finally:
                if should_close:
                    img.close()
        except Exception as e:
            logger.warning(f"Failed to compute pHash: {e}")
            return cls.compute_dhash(image_source, hash_size=hash_size)

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """
        Calculates Hamming distance (number of bit differences) between two hex hashes.
        """
        if not hash1 or not hash2:
            return 64
        try:
            val1 = int(hash1, 16)
            val2 = int(hash2, 16)
            return bin(val1 ^ val2).count("1")
        except (ValueError, TypeError):
            return 64

    @classmethod
    def similarity_score(cls, hash1: str, hash2: str) -> float:
        """
        Returns normalized similarity score in range [0.0, 1.0].
        1.0 means identical hash.
        """
        dist = cls.hamming_distance(hash1, hash2)
        return round(max(0.0, 1.0 - (dist / 64.0)), 4)

    def find_duplicates(
        self,
        screenshots: List[Dict[str, Any]],
        max_distance: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Identifies exact and visually similar duplicate screenshot groups.

        Args:
            screenshots: List of screenshot dicts containing at minimum:
                         'screenshot_id', 'filename', and optionally 'file_hash',
                         'perceptual_hash', 'created_at', 'category', 'image_url'.
            max_distance: Max Hamming distance threshold for near-duplicates (default: 6).
            category_filter: Optional category to restrict comparison to.

        Returns:
            Dictionary containing structured duplicate clusters and summary metrics.
        """
        threshold = max_distance if max_distance is not None else self.default_max_distance

        items = screenshots
        if category_filter:
            items = [s for s in items if str(s.get("category", "")).lower() == category_filter.lower()]

        exact_groups: Dict[str, List[Dict[str, Any]]] = {}
        processed_exact_ids = set()

        # Step 1: Find Exact Duplicates via SHA-256
        for s in items:
            f_hash = s.get("file_hash") or s.get("sha256")
            if f_hash and len(str(f_hash)) == 64:
                exact_groups.setdefault(f_hash, []).append(s)

        structured_groups = []
        group_idx = 1

        for f_hash, members in exact_groups.items():
            if len(members) > 1:
                # Sort deterministically by creation date (oldest first as primary)
                members.sort(key=lambda x: (x.get("created_at") or "", x.get("screenshot_id") or ""))
                primary = members[0]
                duplicates = members[1:]

                dup_entries = []
                for d in duplicates:
                    processed_exact_ids.add(d.get("screenshot_id"))
                    dup_entries.append({
                        "screenshot_id": d.get("screenshot_id"),
                        "filename": d.get("filename"),
                        "image_url": d.get("image_url") or d.get("image_uri"),
                        "category": d.get("category"),
                        "created_at": d.get("created_at"),
                        "file_hash": f_hash,
                        "perceptual_hash": d.get("perceptual_hash") or d.get("phash"),
                        "similarity_type": "exact",
                        "similarity_score": 1.0,
                        "distance": 0,
                    })

                processed_exact_ids.add(primary.get("screenshot_id"))

                structured_groups.append({
                    "group_id": f"grp_exact_{f_hash[:8]}_{group_idx}",
                    "type": "exact",
                    "similarity_score": 1.0,
                    "distance": 0,
                    "hash": f_hash,
                    "primary": {
                        "screenshot_id": primary.get("screenshot_id"),
                        "filename": primary.get("filename"),
                        "image_url": primary.get("image_url") or primary.get("image_uri"),
                        "category": primary.get("category"),
                        "created_at": primary.get("created_at"),
                        "file_hash": f_hash,
                        "perceptual_hash": primary.get("perceptual_hash") or primary.get("phash"),
                    },
                    "duplicates": dup_entries,
                    "total_in_group": len(members),
                })
                group_idx += 1

        # Step 2: Find Visually Near-Duplicates via pHash for non-exact items
        # To avoid redundant quadratic clustering, group pairwise by distance
        remaining_items = [
            s for s in items
            if s.get("screenshot_id") not in processed_exact_ids
            and (s.get("perceptual_hash") or s.get("phash"))
        ]

        assigned_near_ids = set()
        for i in range(len(remaining_items)):
            s1 = remaining_items[i]
            sid1 = s1.get("screenshot_id")
            if sid1 in assigned_near_ids:
                continue

            ph1 = s1.get("perceptual_hash") or s1.get("phash")
            if not ph1 or len(str(ph1)) < 16:
                continue

            near_matches = []
            for j in range(i + 1, len(remaining_items)):
                s2 = remaining_items[j]
                sid2 = s2.get("screenshot_id")
                if sid2 in assigned_near_ids:
                    continue

                ph2 = s2.get("perceptual_hash") or s2.get("phash")
                if not ph2 or len(str(ph2)) < 16:
                    continue

                dist = self.hamming_distance(ph1, ph2)
                if dist <= threshold:
                    sim = self.similarity_score(ph1, ph2)
                    near_matches.append((s2, dist, sim))

            if near_matches:
                assigned_near_ids.add(sid1)
                dup_entries = []
                # Compute average similarity for group
                distances = []
                similarities = []

                for match_item, dist, sim in near_matches:
                    assigned_near_ids.add(match_item.get("screenshot_id"))
                    distances.append(dist)
                    similarities.append(sim)
                    dup_entries.append({
                        "screenshot_id": match_item.get("screenshot_id"),
                        "filename": match_item.get("filename"),
                        "image_url": match_item.get("image_url") or match_item.get("image_uri"),
                        "category": match_item.get("category"),
                        "created_at": match_item.get("created_at"),
                        "file_hash": match_item.get("file_hash") or match_item.get("sha256"),
                        "perceptual_hash": match_item.get("perceptual_hash") or match_item.get("phash"),
                        "similarity_type": "visual",
                        "similarity_score": sim,
                        "distance": dist,
                    })

                avg_sim = round(float(np.mean(similarities)), 4) if similarities else 1.0
                min_dist = min(distances) if distances else 0

                structured_groups.append({
                    "group_id": f"grp_visual_{ph1[:8]}_{group_idx}",
                    "type": "visual",
                    "similarity_score": avg_sim,
                    "distance": min_dist,
                    "hash": ph1,
                    "primary": {
                        "screenshot_id": s1.get("screenshot_id"),
                        "filename": s1.get("filename"),
                        "image_url": s1.get("image_url") or s1.get("image_uri"),
                        "category": s1.get("category"),
                        "created_at": s1.get("created_at"),
                        "file_hash": s1.get("file_hash") or s1.get("sha256"),
                        "perceptual_hash": ph1,
                    },
                    "duplicates": dup_entries,
                    "total_in_group": len(dup_entries) + 1,
                })
                group_idx += 1

        exact_count = sum(1 for g in structured_groups if g["type"] == "exact")
        visual_count = sum(1 for g in structured_groups if g["type"] == "visual")
        total_dup_candidates = sum(len(g["duplicates"]) for g in structured_groups)

        return {
            "total_groups": len(structured_groups),
            "exact_groups_count": exact_count,
            "visual_groups_count": visual_count,
            "total_duplicate_candidates": total_dup_candidates,
            "max_distance_threshold": threshold,
            "groups": structured_groups,
        }
