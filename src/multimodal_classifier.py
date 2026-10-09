"""
Multimodal Screenshot Classifier.
Combines existing Explainable OCR + MiniLM Text Classifier with Visual Category Evidence Engine
using confidence-aware dynamic score fusion and graceful visual fallbacks.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.visual.visual_evidence import (
    VisualCategoryEvidenceEngine,
    CANONICAL_CATEGORIES,
)


class MultimodalScreenshotClassifier:
    """
    Experimental Multimodal Classifier wrapping the text classifier and visual evidence engine.
    Preserves original text classifier completely while adding visual evidence branch signals.
    """

    def __init__(
        self,
        text_classifier: Optional[Any] = None,
        visual_engine: Optional[VisualCategoryEvidenceEngine] = None,
        enable_visual: bool = True,
        w_text_default: float = 0.70,
        w_visual_default: float = 0.30,
    ):
        """
        Args:
            text_classifier: Instance of FinalRefinedClassifier or compatible text classifier.
            visual_engine: Instance of VisualCategoryEvidenceEngine (optional, created lazily if enabled).
            enable_visual: If False, disables TinyCLIP visual branch and runs in pure text-only mode.
            w_text_default: Default fusion weight for text branch.
            w_visual_default: Default fusion weight for visual branch.
        """
        self._text_classifier = text_classifier
        self._visual_engine = visual_engine
        self.enable_visual = enable_visual
        self.w_text_default = w_text_default
        self.w_visual_default = w_visual_default

    @property
    def text_classifier(self) -> Any:
        """Lazily loads and returns original text classifier."""
        if self._text_classifier is None:
            from experiments.intelligence.final_refinement.final_refinement_experiment import (
                FinalRefinedClassifier,
            )
            self._text_classifier = FinalRefinedClassifier()
        return self._text_classifier

    @property
    def visual_engine(self) -> Optional[VisualCategoryEvidenceEngine]:
        """Lazily loads and returns visual category evidence engine if enabled."""
        if not self.enable_visual:
            return None
        if self._visual_engine is None:
            from src.visual.tiny_clip_encoder import TinyCLIPEncoder
            encoder = TinyCLIPEncoder(device="cpu")
            self._visual_engine = VisualCategoryEvidenceEngine(encoder=encoder)
        return self._visual_engine

    def _min_max_scale_dict(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Scales dictionary values to [0.0, 1.0] range using min-max normalization."""
        vals = list(scores.values())
        min_v, max_v = min(vals), max(vals)
        denom = max_v - min_v
        if denom < 1e-6:
            return {k: 0.5 for k in scores}
        return {k: float((v - min_v) / denom) for k, v in scores.items()}

    def _compute_fusion_weights(
        self,
        ocr_text: str,
        text_result: Dict[str, Any],
        visual_result: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Computes dynamic, confidence-aware fusion weights (w_text, w_visual)
        based on OCR text density, text margin, and visual margin.
        """
        tokens = ocr_text.strip().split()
        num_chars = len(ocr_text.strip())
        is_sparse = (len(tokens) < 5 or num_chars < 20)

        text_margin = text_result.get("margin", 0.0)
        text_conf = text_result.get("confidence", "Low")

        vis_raw = visual_result.get("category_raw_scores", {})
        sorted_vis = sorted(vis_raw.values(), reverse=True)
        vis_margin = (sorted_vis[0] - sorted_vis[1]) if len(sorted_vis) > 1 else 0.0

        # Case 1: Sparse / No OCR Text -> Rely heavily on visual branch
        if is_sparse:
            return 0.15, 0.85

        # Case 2: High Confidence Text Evidence -> Rely primarily on text branch
        if text_conf == "High" and text_margin >= 0.18:
            return 0.80, 0.20

        # Case 3: Ambiguous Text vs Strong Visual Separation -> Balanced 50/50
        if text_margin < 0.08 and vis_margin >= 0.04:
            return 0.50, 0.50

        # Case 4: Default Configured Weights
        return self.w_text_default, self.w_visual_default

    def predict(
        self,
        ocr_text: str = "",
        image_path: Optional[Union[str, Path]] = None,
        notes: str = "",
        variant: str = "C",
    ) -> Dict[str, Any]:
        """
        Executes multimodal prediction combining text and visual evidence.
        Args:
            ocr_text: Raw or normalized OCR text extracted from screenshot.
            image_path: File path to screenshot image (required for visual branch).
            notes: Optional user notes or annotations.
            variant: Text classifier variant parameter ("C" by default).
        Returns:
            Dict containing unified prediction, category_scores, top_candidates,
            tags, text_prediction, visual_prediction, contributions, and evidence breakdown.
        """
        t0 = time.perf_counter()
        # 1. Execute Original Text Classifier Branch (Mandatory Baseline)
        text_result = self.text_classifier.predict_final(ocr_text, notes=notes, variant=variant)

        visual_result: Optional[Dict[str, Any]] = None
        visual_fallback_used = True
        visual_error_msg: Optional[str] = None

        # 2. Execute Visual Evidence Branch (If enabled and valid image path provided)
        if self.enable_visual and image_path is not None:
            try:
                img_path_obj = Path(image_path)
                if img_path_obj.exists() and img_path_obj.is_file():
                    v_engine = self.visual_engine
                    if v_engine is not None:
                        visual_result = v_engine.extract_evidence(img_path_obj)
                        visual_fallback_used = False
                else:
                    visual_error_msg = f"Image path does not exist or is not a file: '{image_path}'"
            except Exception as e:
                visual_fallback_used = True
                visual_error_msg = f"Visual branch exception: {e}"

        # 3. Perform Score Fusion if Visual Evidence is Available
        if not visual_fallback_used and visual_result is not None:
            w_text, w_visual = self._compute_fusion_weights(ocr_text, text_result, visual_result)

            text_raw = text_result.get("category_scores", {})
            vis_raw = visual_result.get("category_raw_scores", {})

            text_cal = self._min_max_scale_dict(text_raw)
            vis_cal = self._min_max_scale_dict(vis_raw)

            fused_scores = {}
            for cat in CANONICAL_CATEGORIES:
                s_t = text_cal.get(cat, 0.0)
                s_v = vis_cal.get(cat, 0.0)
                fused_scores[cat] = float(w_text * s_t + w_visual * s_v)

            # Rank candidates
            sorted_cats = sorted(CANONICAL_CATEGORIES, key=lambda c: fused_scores[c], reverse=True)
            top1_cat = sorted_cats[0]
            top2_cat = sorted_cats[1] if len(sorted_cats) > 1 else sorted_cats[0]
            top3_cat = sorted_cats[2] if len(sorted_cats) > 2 else sorted_cats[0]

            top1_score = fused_scores[top1_cat]
            top2_score = fused_scores[top2_cat]
            margin = top1_score - top2_score

            # Fused Confidence Logic
            if top1_score > 0.50 and margin >= 0.12:
                confidence = "High"
            elif top1_score > 0.30 and margin >= 0.06:
                confidence = "Medium"
            else:
                confidence = "Low"

            # Combine Evidence
            evidence_matches = list(text_result.get("evidence_matches", []))
            vis_prompt_ev = visual_result.get("prompt_evidence", {}).get(top1_cat, {})
            best_prompt = vis_prompt_ev.get("best_prompt", "")
            best_sim = vis_prompt_ev.get("best_similarity", 0.0)
            if best_prompt:
                evidence_matches.append(
                    f"Visual TinyCLIP Match: {top1_cat} (sim: {best_sim:.3f}, prompt: '{best_prompt[:40]}...')"
                )

            vis_pred = visual_result.get("top_category")
            vis_scores = visual_result.get("category_raw_scores", {})

        else:
            # Fallback to Text Classifier Output directly
            fused_scores = dict(text_result.get("category_scores", {}))
            top1_cat = text_result.get("prediction", "General")
            confidence = text_result.get("confidence", "Low")
            margin = text_result.get("margin", 0.0)
            sorted_cats = [c["category"] for c in text_result.get("top_candidates", [])]
            top2_cat = sorted_cats[1] if len(sorted_cats) > 1 else top1_cat
            top3_cat = sorted_cats[2] if len(sorted_cats) > 2 else top1_cat

            w_text, w_visual = 1.0, 0.0
            evidence_matches = list(text_result.get("evidence_matches", []))
            if visual_error_msg:
                evidence_matches.append(f"[Visual Fallback]: {visual_error_msg}")
            else:
                evidence_matches.append("[Visual Fallback]: Visual branch disabled or image path omitted.")

            vis_pred = None
            vis_scores = {c: 0.0 for c in CANONICAL_CATEGORIES}

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        return {
            "prediction": top1_cat,
            "confidence": confidence,
            "margin": round(margin, 4),
            "is_ambiguous": (margin < 0.08),
            "is_sparse": text_result.get("is_sparse", False),
            "category_scores": {c: round(fused_scores[c], 4) for c in CANONICAL_CATEGORIES},
            "top_candidates": [
                {"category": top1_cat, "score": round(fused_scores[top1_cat], 4)},
                {"category": top2_cat, "score": round(fused_scores[top2_cat], 4)},
                {"category": top3_cat, "score": round(fused_scores[top3_cat], 4)},
            ],
            "tags": text_result.get("tags", []),
            "evidence_matches": evidence_matches,
            "text_prediction": text_result.get("prediction"),
            "text_confidence": text_result.get("confidence"),
            "text_scores": text_result.get("category_scores", {}),
            "text_weight": round(w_text, 2),
            "visual_prediction": vis_pred,
            "visual_scores": vis_scores,
            "visual_weight": round(w_visual, 2),
            "visual_fallback_used": visual_fallback_used,
            "ocr_available": bool(ocr_text and ocr_text.strip()),
            "text_length": len(ocr_text.strip()),
            "reasoning_ms": round(total_ms, 3),
        }
