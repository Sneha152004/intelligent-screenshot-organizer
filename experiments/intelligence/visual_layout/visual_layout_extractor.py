"""
Lightweight Visual & Layout Feature Extractor for Screenshot Intelligence
========================================================================
Extracts lightweight, CPU-friendly visual and structural layout features directly
from screenshot images using OpenCV, PIL, and NumPy without any pretrained visual neural networks.

Feature Families Extracted:
1. Image Geometry & Orientation (width, height, aspect ratio, orientation)
2. Color, Brightness & Dark/Light Mode Statistics (mean/std brightness, dark/light ratios, saturation)
3. Edge, Line & Region Structure (Canny edge density, horizontal/vertical Sobel edge ratio, divider lines)
4. OCR Text Region Spatial Geometry (text box count, coverage, vertical distribution, left margin std, indentation)
5. Heuristic Layout Signals (IDE/Editor, Terminal/Console, Slide/Document, Browser/Webpage, Settings UI, Product Grid, Messaging Chat, Media/Video, Map/Travel, Generic/Unknown)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# Canonical 10 Categories
CANONICAL_CATEGORIES = [
    "Coding",
    "Communication",
    "Connectivity",
    "Education",
    "Entertainment",
    "General",
    "Payment",
    "Shopping",
    "Travel",
    "Work & Career",
]


def extract_visual_layout_features(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extracts lightweight visual and layout features from an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Dict containing raw feature metrics, layout heuristic scores (0.0 to 1.0),
        visual evidence category scores, and feature extraction latency in milliseconds.
    """
    t0 = time.perf_counter()
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Screenshot image not found: {img_path}")

    # Load image using OpenCV
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Failed to read image file: {img_path}")

    h, w, c = img_bgr.shape

    # -------------------------------------------------------------------------
    # 1. Image Geometry & Orientation
    # -------------------------------------------------------------------------
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0
    if aspect_ratio < 0.85:
        orientation = "portrait"
    elif aspect_ratio > 1.20:
        orientation = "landscape"
    else:
        orientation = "square"

    area_pixels = int(w * h)

    # -------------------------------------------------------------------------
    # 2. Color, Brightness & Contrast Statistics
    # -------------------------------------------------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))

    dark_pixel_ratio = float(np.sum(gray < 60)) / float(h * w)
    light_pixel_ratio = float(np.sum(gray > 200)) / float(h * w)

    is_dark_mode = 1.0 if (mean_brightness < 90.0 or dark_pixel_ratio > 0.55) else 0.0
    is_light_mode = 1.0 if (mean_brightness > 175.0 or light_pixel_ratio > 0.55) else 0.0

    mean_saturation = float(np.mean(hsv[:, :, 1])) / 255.0

    # Top vs Middle Brightness Split (Header vs Body contrast)
    top_split_h = max(1, int(h * 0.20))
    top_gray = gray[:top_split_h, :]
    mid_gray = gray[top_split_h:int(h * 0.80), :]
    top_vs_mid_contrast = float(abs(np.mean(top_gray) - np.mean(mid_gray)))

    # -------------------------------------------------------------------------
    # 3. Edge, Line & Structural Region Features
    # -------------------------------------------------------------------------
    edges = cv2.Canny(gray, 50, 150)
    canny_edge_density = float(np.sum(edges > 0)) / float(h * w)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sum_abs_x = float(np.sum(np.abs(sobel_x)))
    sum_abs_y = float(np.sum(np.abs(sobel_y)))
    h_v_edge_ratio = (sum_abs_y + 1e-5) / (sum_abs_x + 1e-5)

    # Horizontal Lines Detection (dividers, row separators, code lines)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(w * 0.15)), 1))
    horiz_lines_img = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horiz_kernel)
    horiz_contours, _ = cv2.findContours(horiz_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal_lines_count = len([c for c in horiz_contours if cv2.boundingRect(c)[2] > int(w * 0.12)])

    # Card / Grid Region Count
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    card_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(w * 0.08)), max(5, int(h * 0.04))))
    card_dilated = cv2.dilate(thresh_inv, card_kernel, iterations=1)
    card_contours, _ = cv2.findContours(card_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_boxes = [cv2.boundingRect(c) for c in card_contours if cv2.boundingRect(c)[2] > int(w * 0.15) and cv2.boundingRect(c)[3] > int(h * 0.03)]
    grid_card_count = len(card_boxes)

    # -------------------------------------------------------------------------
    # 4. OCR Text Region Spatial Geometry (Morphological Text Box Extractor)
    # -------------------------------------------------------------------------
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(w * 0.03)), max(2, int(h * 0.005))))
    text_dilated = cv2.dilate(thresh_inv, text_kernel, iterations=1)
    text_contours, _ = cv2.findContours(text_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_boxes = []
    for c_cnt in text_contours:
        bx, by, bw, bh = cv2.boundingRect(c_cnt)
        if bw > 10 and bh > 5 and bw < int(w * 0.98):
            text_boxes.append((bx, by, bw, bh))

    text_box_count = len(text_boxes)

    if text_box_count > 0:
        total_text_area = sum(bw * bh for _, _, bw, bh in text_boxes)
        text_area_coverage = float(total_text_area) / float(h * w)

        avg_box_height = float(np.mean([bh for _, _, _, bh in text_boxes]))
        avg_box_width = float(np.mean([bw for _, _, _, bw in text_boxes]))

        # Vertical Zone Distribution
        top_boxes = sum(1 for _, by, _, _ in text_boxes if by < int(h * 0.20))
        mid_boxes = sum(1 for _, by, _, _ in text_boxes if int(h * 0.20) <= by <= int(h * 0.80))
        bot_boxes = sum(1 for _, by, _, _ in text_boxes if by > int(h * 0.80))

        top_zone_ratio = float(top_boxes) / float(text_box_count)
        mid_zone_ratio = float(mid_boxes) / float(text_box_count)
        bot_zone_ratio = float(bot_boxes) / float(text_box_count)

        # Left Margin Alignment & Indentation
        left_margins_norm = np.array([float(bx) / float(w) for bx, _, _, _ in text_boxes])
        left_margin_std = float(np.std(left_margins_norm))

        # Code Indentation Pattern (Check for left margin clusters near 0.02, 0.06, 0.10, 0.14)
        left_margin_counts, _ = np.histogram(left_margins_norm, bins=10, range=(0.0, 0.5))
        active_indent_clusters = int(np.sum(left_margin_counts > 1))
        code_indentation_score = min(1.0, active_indent_clusters / 4.0)
    else:
        text_area_coverage = 0.0
        avg_box_height = 0.0
        avg_box_width = 0.0
        top_zone_ratio = 0.0
        mid_zone_ratio = 0.0
        bot_zone_ratio = 0.0
        left_margin_std = 0.5
        code_indentation_score = 0.0

    # -------------------------------------------------------------------------
    # 5. Heuristic Layout Signals (0.0 to 1.0)
    # -------------------------------------------------------------------------
    # A. IDE / Editor Layout (Dark mode + left-aligned text + code indentations + top header + low saturation)
    ide_score = 0.0
    if is_dark_mode > 0.5:
        ide_score += 0.35
    if left_margin_std < 0.15 and text_box_count >= 8:
        ide_score += 0.25
    if code_indentation_score >= 0.50:
        ide_score += 0.25
    if mean_saturation < 0.25:
        ide_score += 0.15
    ide_editor_layout = float(min(1.0, ide_score))

    # B. Terminal / Console Layout (Very dark background + uniform dark background + dense left-aligned text)
    term_score = 0.0
    if mean_brightness < 60.0 and dark_pixel_ratio > 0.65:
        term_score += 0.45
    if std_brightness < 65.0:
        term_score += 0.25
    if left_margin_std < 0.12:
        term_score += 0.30
    terminal_console_layout = float(min(1.0, term_score))

    # C. Slide / Course / Document Layout (Light background + large top title box + centered/formatted body + low edge density)
    slide_score = 0.0
    if is_light_mode > 0.5:
        slide_score += 0.35
    if top_zone_ratio > 0.15 and avg_box_height > 20.0:
        slide_score += 0.30
    if canny_edge_density < 0.035:
        slide_score += 0.20
    if left_margin_std > 0.10 and text_box_count >= 5:
        slide_score += 0.15
    slide_course_document_layout = float(min(1.0, slide_score))

    # D. Browser / Webpage Layout (Top header bar + light background + mixed card edges + vertical layout)
    browser_score = 0.0
    if top_zone_ratio > 0.20:
        browser_score += 0.35
    if is_light_mode > 0.5:
        browser_score += 0.25
    if top_vs_mid_contrast > 20.0:
        browser_score += 0.25
    if canny_edge_density >= 0.02:
        browser_score += 0.15
    browser_webpage_layout = float(min(1.0, browser_score))

    # E. Settings / System UI Layout (Repeated horizontal dividers + clean background)
    settings_score = 0.0
    if horizontal_lines_count >= 4:
        settings_score += 0.45
    if canny_edge_density < 0.03 and is_light_mode > 0.5:
        settings_score += 0.30
    if left_margin_std < 0.12 and text_box_count >= 4:
        settings_score += 0.25
    settings_system_ui_layout = float(min(1.0, settings_score))

    # F. Product Grid / E-Commerce Layout (Multiple grid cards + mixed images + higher color saturation)
    product_score = 0.0
    if grid_card_count >= 3:
        product_score += 0.45
    if mean_saturation > 0.15:
        product_score += 0.30
    if canny_edge_density >= 0.03:
        product_score += 0.25
    product_grid_ecommerce_layout = float(min(1.0, product_score))

    # G. Messaging / Chat Layout (Alternating left/right margins + bottom input bar)
    chat_score = 0.0
    if bot_zone_ratio > 0.15:
        chat_score += 0.40
    if 0.10 <= left_margin_std <= 0.25:
        chat_score += 0.35
    if text_box_count >= 6:
        chat_score += 0.25
    messaging_chat_layout = float(min(1.0, chat_score))

    # H. Media / Video Layout (Landscape aspect ratio + large central area without text)
    media_score = 0.0
    if orientation == "landscape":
        media_score += 0.40
    if mid_zone_ratio < 0.40 and text_box_count < 10:
        media_score += 0.35
    if mean_saturation > 0.20:
        media_score += 0.25
    media_video_layout = float(min(1.0, media_score))

    # I. Map / Travel Layout (Non-standard graphic texture + cards)
    map_score = 0.0
    if grid_card_count >= 2 and mean_saturation > 0.18:
        map_score += 0.50
    if canny_edge_density >= 0.035:
        map_score += 0.50
    map_travel_layout = float(min(1.0, map_score))

    # J. Generic / Unknown Layout (Low edge density + sparse text + unstructured background)
    gen_score = 0.0
    if canny_edge_density < 0.015:
        gen_score += 0.45
    if text_box_count < 5:
        gen_score += 0.35
    if std_brightness < 40.0:
        gen_score += 0.20
    generic_unknown_layout = float(min(1.0, gen_score))

    # -------------------------------------------------------------------------
    # 6. Map Layout Signals to Visual Category Evidence Scores (0.0 to 1.0)
    # -------------------------------------------------------------------------
    visual_category_evidence: Dict[str, float] = {
        "Coding": float(min(1.0, 0.60 * ide_editor_layout + 0.40 * terminal_console_layout + 0.20 * code_indentation_score)),
        "Education": float(min(1.0, 0.70 * slide_course_document_layout + 0.30 * browser_webpage_layout)),
        "General": float(min(1.0, 0.75 * generic_unknown_layout + 0.25 * (1.0 - min(1.0, canny_edge_density * 30.0)))),
        "Shopping": float(min(1.0, 0.80 * product_grid_ecommerce_layout + 0.20 * browser_webpage_layout)),
        "Communication": float(min(1.0, 0.85 * messaging_chat_layout)),
        "Entertainment": float(min(1.0, 0.85 * media_video_layout)),
        "Travel": float(min(1.0, 0.60 * map_travel_layout + 0.40 * browser_webpage_layout)),
        "Connectivity": float(min(1.0, 0.85 * settings_system_ui_layout)),
        "Work & Career": float(min(1.0, 0.50 * slide_course_document_layout + 0.50 * browser_webpage_layout)),
        "Payment": float(min(1.0, 0.50 * product_grid_ecommerce_layout + 0.50 * settings_system_ui_layout)),
    }

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    return {
        "image_geometry": {
            "width": w,
            "height": h,
            "aspect_ratio": round(aspect_ratio, 4),
            "orientation": orientation,
            "area_pixels": area_pixels,
        },
        "color_brightness": {
            "mean_brightness": round(mean_brightness, 2),
            "std_brightness": round(std_brightness, 2),
            "dark_pixel_ratio": round(dark_pixel_ratio, 4),
            "light_pixel_ratio": round(light_pixel_ratio, 4),
            "is_dark_mode": is_dark_mode,
            "is_light_mode": is_light_mode,
            "mean_saturation": round(mean_saturation, 4),
            "top_vs_mid_contrast": round(top_vs_mid_contrast, 2),
        },
        "edge_structure": {
            "canny_edge_density": round(canny_edge_density, 6),
            "h_v_edge_ratio": round(h_v_edge_ratio, 4),
            "horizontal_lines_count": horizontal_lines_count,
            "grid_card_count": grid_card_count,
        },
        "text_geometry": {
            "text_box_count": text_box_count,
            "text_area_coverage": round(text_area_coverage, 4),
            "avg_box_height": round(avg_box_height, 2),
            "avg_box_width": round(avg_box_width, 2),
            "top_zone_ratio": round(top_zone_ratio, 4),
            "mid_zone_ratio": round(mid_zone_ratio, 4),
            "bot_zone_ratio": round(bot_zone_ratio, 4),
            "left_margin_std": round(left_margin_std, 4),
            "code_indentation_score": round(code_indentation_score, 4),
        },
        "layout_heuristics": {
            "ide_editor_layout": round(ide_editor_layout, 4),
            "terminal_console_layout": round(terminal_console_layout, 4),
            "slide_course_document_layout": round(slide_course_document_layout, 4),
            "browser_webpage_layout": round(browser_webpage_layout, 4),
            "settings_system_ui_layout": round(settings_system_ui_layout, 4),
            "product_grid_ecommerce_layout": round(product_grid_ecommerce_layout, 4),
            "messaging_chat_layout": round(messaging_chat_layout, 4),
            "media_video_layout": round(media_video_layout, 4),
            "map_travel_layout": round(map_travel_layout, 4),
            "generic_unknown_layout": round(generic_unknown_layout, 4),
        },
        "visual_category_evidence": {c: round(v, 4) for c, v in visual_category_evidence.items()},
        "latency_ms": round(latency_ms, 3),
    }
