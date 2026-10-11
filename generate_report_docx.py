"""
Script to generate Progress Report #1 DOCX for Intelligent Screenshot Organizer
Creates a professionally styled Microsoft Word document (.docx).
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets padding for a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color_hex="CBD5E0"):
    """Sets light gray borders for a table."""
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{color_hex}"/>'
            f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{color_hex}"/>'
            f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color_hex}"/>'
            f'<w:insideV w:val="none"/>'
            f'<w:left w:val="none"/>'
            f'<w:right w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)

def generate_docx_report(docx_path="Progress_Report_1_Intelligent_Screenshot_Organizer.docx"):
    doc = docx.Document()

    # Page Margins: 1 inch on all sides
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Color Palette Constants
    HEX_PRIMARY = "1A365D"   # Deep Navy
    HEX_SECONDARY = "2B6CB0" # Slate Blue
    HEX_DARK = "2D3748"      # Text Dark
    HEX_BG = "F7FAFC"        # Light Gray
    HEX_ACCENT = "319795"    # Teal Accent

    COLOR_PRIMARY = RGBColor(0x1A, 0x36, 0x5D)
    COLOR_SECONDARY = RGBColor(0x2B, 0x6C, 0xB0)
    COLOR_DARK = RGBColor(0x2D, 0x37, 0x48)

    # Set Default Normal Font
    style_normal = doc.styles['Normal']
    font_normal = style_normal.font
    font_normal.name = 'Calibri'
    font_normal.size = Pt(11)
    font_normal.color.rgb = COLOR_DARK

    # --- TITLE & HEADER BLOCK ---
    title_p = doc.add_paragraph()
    title_run = title_p.add_run("Intelligent Screenshot Organizer")
    title_run.font.name = 'Calibri'
    title_run.font.size = Pt(24)
    title_run.font.bold = True
    title_run.font.color.rgb = COLOR_PRIMARY
    title_p.paragraph_format.space_after = Pt(2)

    subtitle_p = doc.add_paragraph()
    subtitle_run = subtitle_p.add_run("PROJECT PROGRESS REPORT #1 OF 4: Automated Ingestion, Grounded OCR, and Vector Embeddings")
    subtitle_run.font.name = 'Calibri'
    subtitle_run.font.size = Pt(12)
    subtitle_run.font.bold = True
    subtitle_run.font.color.rgb = COLOR_SECONDARY
    subtitle_p.paragraph_format.space_after = Pt(12)

    # Metadata Info Table
    meta_table = doc.add_table(rows=3, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        [("Project Domain:", " AI / Computer Vision / Vector Search"), ("Institution:", " KIIT Deemed to be University")],
        [("Current Milestone:", " Milestone 3 (Embedding Integration)"), ("Department:", " School of Computer Engineering")],
        [("Git Branch:", " embeddings"), ("Date:", " August 2026")]
    ]

    for row_idx, row in enumerate(meta_data):
        for col_idx, (label, val) in enumerate(row):
            cell = meta_table.cell(row_idx, col_idx)
            set_cell_background(cell, HEX_BG)
            set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            
            run_lbl = p.add_run(label)
            run_lbl.bold = True
            run_lbl.font.size = Pt(9.5)
            run_lbl.font.color.rgb = COLOR_PRIMARY
            
            run_val = p.add_run(val)
            run_val.font.size = Pt(9.5)
            run_val.font.color.rgb = COLOR_DARK

    set_table_borders(meta_table, "CBD5E0")
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Helper function for Section Headings
    def add_heading_1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(16)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True
        run = h.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        return h

    def add_heading_2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        run = h.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = COLOR_SECONDARY
        return h

    # --- SECTION 1: EXECUTIVE SUMMARY ---
    add_heading_1("1. Executive Summary & Project Goal")
    p1 = doc.add_paragraph(
        "Digital clutter from unorganized screenshots is a pervasive problem. Screenshots contain rich visual and textual information—including Wi-Fi credentials, payment receipts, flight boarding passes, academic lecture slides, and code snippets—yet traditional file systems render them unsearchable."
    )
    p1.paragraph_format.space_after = Pt(8)

    p2 = doc.add_paragraph(
        "The Intelligent Screenshot Organizer is an enterprise-grade AI system being developed across 4 structured project milestones. This document serves as Progress Report #1 (of 4), documenting the successful completion of Milestones 1 through 3: dataset loading, grounded optical character recognition (OCR), paragraph chunking, local vector embeddings, and integration verification."
    )
    p2.paragraph_format.space_after = Pt(10)

    add_heading_2("Key Accomplishments in Progress Report #1:")
    bullet_items = [
        ("Benchmark Dataset & Metadata Loader: ", "Formatted 30 real-world screenshots across 9 categories in sample/ with automated metadata validation (src/loader.py)."),
        ("Grounded OCR Architecture: ", "Implemented a modular BaseOCREngine interface and MockOCREngine (src/mock_ocr.py) fully grounded in actual screenshot image content."),
        ("Paragraph Text Chunker: ", "Developed ParagraphChunker (src/chunker.py) to clean and partition raw extracted text into paragraph units."),
        ("Local Vector Embedding Layer: ", "Integrated SentenceTransformerEmbedder (src/embedder.py) utilizing the all-MiniLM-L6-v2 model to generate 384-dimensional dense vectors."),
        ("Verification & Traceability: ", "Executed complete end-to-end integration proving Screenshot -> Grounded OCR -> Chunks -> 384-Dim Vectors with 100% test coverage (15/15 Pytest checks passed)."),
    ]

    for title, desc in bullet_items:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(4)
        r1 = bp.add_run(title)
        r1.bold = True
        r1.font.color.rgb = COLOR_PRIMARY
        r2 = bp.add_run(desc)
        r2.font.color.rgb = COLOR_DARK

    # --- SECTION 2: SYSTEM ARCHITECTURE ---
    add_heading_1("2. System Architecture & Processing Workflow")
    p_arch = doc.add_paragraph(
        "The system adheres to modular object-oriented design patterns, enforcing strict separation of concerns via Abstract Base Classes (BaseOCREngine, BaseEmbedder). This guarantees that mock components can be seamlessly swapped with production OCR models (PaddleOCR / EasyOCR) and vector stores (ChromaDB / FAISS) without refactoring downstream pipeline code."
    )
    p_arch.paragraph_format.space_after = Pt(10)

    # Insert Architecture Flowchart Image if available
    flowchart_path = "docs/system_architecture_flowchart.png"
    if os.path.exists(flowchart_path):
        img_p = doc.add_paragraph()
        img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img_run = img_p.add_run()
        img_run.add_picture(flowchart_path, width=Inches(6.2))
        
        cap_p = doc.add_paragraph()
        cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_run = cap_p.add_run("Figure 1: End-to-End Modular Processing Architecture (Progress Report #1)")
        cap_run.font.size = Pt(9)
        cap_run.font.italic = True
        cap_run.font.color.rgb = RGBColor(0x71, 0x80, 0x96)
        cap_p.paragraph_format.space_after = Pt(12)

    # --- SECTION 3: DATASET INGESTION ---
    add_heading_1("3. Dataset Ingestion & Category Breakdown")
    doc.add_paragraph(
        "A benchmark dataset of 30 sample screenshots was created in sample/ alongside sample/metadata.csv. The dataset covers 9 major real-world functional categories:"
    ).paragraph_format.space_after = Pt(8)

    # Dataset Summary Table
    ds_data = [
        ["Category", "Count", "Sample Screenshots", "Key Information Extracted"],
        ["Education", "8", "ss_003, ss_006, ss_007, ss_012, ss_018, ss_023, ss_024, ss_027", "KIIT Deep Learning curriculum, handwritten calculus notes, exam notices, formulas slide."],
        ["Payment", "5", "ss_002, ss_005, ss_013, ss_017, ss_026", "UPI transaction receipts (₹689, ₹1800, ₹99), KIIT Indian Bank gateway slips."],
        ["Shopping", "5", "ss_008, ss_010, ss_015, ss_020, ss_022", "MD Combo flyer menu, Amazon/Flipkart order tracking, Zapvi phone cover invoice."],
        ["Travel", "4", "ss_011, ss_014, ss_016, ss_019", "IndiGo flight boarding passes (PNR, seat, QR codes), IRCTC train booking confirmation."],
        ["Coding", "2", "ss_028, ss_029", "VS Code Express.js asset controller, Terminal curl REST API JSON response."],
        ["Entertainment", "2", "ss_004, ss_025", "BookMyShow Spider-Man movie ticket, Wikipedia Power Rangers movie poster."],
        ["Connectivity", "1", "ss_001", "Wi-Fi network credentials (SSID: Airtel_runu_7550, Password: Air@53054)."],
        ["Work & Career", "2", "ss_009, ss_030", "UniConverter web page, OpenAI 2nd Global AI Hackathon announcement post."],
        ["Communication", "1", "ss_021", "Android notification shade (WhatsApp chat messages, Telegram alerts)."],
    ]

    table_ds = doc.add_table(rows=len(ds_data), cols=4)
    table_ds.alignment = WD_TABLE_ALIGNMENT.CENTER

    col_widths_ds = [Inches(1.1), Inches(0.6), Inches(2.2), Inches(2.6)]

    for row_idx, row in enumerate(ds_data):
        for col_idx, text in enumerate(row):
            cell = table_ds.cell(row_idx, col_idx)
            cell.width = col_widths_ds[col_idx]
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(text)
            
            if row_idx == 0:
                set_cell_background(cell, HEX_PRIMARY)
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9.5)
            else:
                bg = HEX_BG if row_idx % 2 == 1 else "FFFFFF"
                set_cell_background(cell, bg)
                run.font.size = Pt(9)
                if col_idx == 0:
                    run.bold = True
                    run.font.color.rgb = COLOR_PRIMARY
                else:
                    run.font.color.rgb = COLOR_DARK

    set_table_borders(table_ds, "CBD5E0")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # --- SECTION 4: GROUNDED OCR & CHUNKING ---
    add_heading_1("4. Grounded OCR Engine & Text Chunking Modules")
    doc.add_paragraph(
        "To allow rapid development of downstream vector indexing prior to heavy OCR model integration, a grounded mock OCR module (src/mock_ocr.py) was developed. All 30 text mappings were verified line-by-line against the actual visual text of the screenshots."
    ).paragraph_format.space_after = Pt(8)

    # Callout Box for Grounded Sample
    ocr_box_data = [
        [("Sample Grounded OCR Text (ss_001.jpg):", "\nAirtel_runu_7550\n\nPassword\nAir@53054\n\nShow password\n\nAdvanced options\n\nCancel\nConnect"),
         ("Extracted Paragraph Chunks (5 Chunks):", "\n1. 'Airtel_runu_7550'\n2. 'Password\\nAir@53054'\n3. 'Show password'\n4. 'Advanced options'\n5. 'Cancel\\nConnect'")]
    ]
    table_ocr = doc.add_table(rows=1, cols=2)
    table_ocr.alignment = WD_TABLE_ALIGNMENT.CENTER
    for col_idx, (t_head, t_body) in enumerate(ocr_box_data[0]):
        cell = table_ocr.cell(0, col_idx)
        cell.width = Inches(3.2)
        set_cell_background(cell, "EBF8FF")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        
        r_head = p.add_run(t_head)
        r_head.bold = True
        r_head.font.size = Pt(9.5)
        r_head.font.color.rgb = COLOR_PRIMARY
        
        r_body = p.add_run(t_body)
        r_body.font.name = 'Consolas'
        r_body.font.size = Pt(8.5)
        r_body.font.color.rgb = RGBColor(0x2C, 0x52, 0x82)

    set_table_borders(table_ocr, "3182CE")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # --- SECTION 5: EMBEDDING LAYER IMPLEMENTATION ---
    add_heading_1("5. Local SentenceTransformer Vector Embedding Layer")
    doc.add_paragraph(
        "Milestone 3 implemented the local embedding module (src/embedder.py). It uses the all-MiniLM-L6-v2 model from sentence-transformers, loaded once upon initialization."
    ).paragraph_format.space_after = Pt(6)

    embed_bullets = [
        ("Base Class Abstraction: ", "BaseEmbedder(ABC) defines embed(texts: List[str]) -> List[List[float]]."),
        ("Deterministic Local Inference: ", "Converts text chunks into dense 384-dimensional floating-point vectors deterministically without external API calls."),
        ("Input Order & Validation: ", "Preserves input text ordering, rejects non-string inputs with TypeError, and handles empty lists gracefully."),
        ("Empirical Execution Metrics: ", "Extracted 179 total paragraph chunks from 30 screenshots and generated 179 vectors of 384 dimensions."),
    ]
    for b_title, b_desc in embed_bullets:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(4)
        r1 = bp.add_run(b_title)
        r1.bold = True
        r1.font.color.rgb = COLOR_PRIMARY
        r2 = bp.add_run(b_desc)
        r2.font.color.rgb = COLOR_DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # --- SECTION 6: TESTING & VERIFICATION ---
    add_heading_1("6. Verification, Integration & Test Results")
    doc.add_paragraph(
        "The codebase was subjected to unit and integration testing using pytest. All 15 tests passed cleanly in 25.68 seconds:"
    ).paragraph_format.space_after = Pt(8)

    # Test Results Table
    test_data = [
        ["Test Suite Module", "Test Name", "Scope & Assertion Description", "Status"],
        ["test_embedder.py", "test_single_input", "Verifies 1 text input produces 1 non-empty float vector.", "PASSED"],
        ["test_embedder.py", "test_multiple_inputs", "Verifies 3 text inputs produce 3 vectors of uniform dim (384).", "PASSED"],
        ["test_embedder.py", "test_empty_input", "Verifies embed([]) cleanly returns empty list [].", "PASSED"],
        ["test_embedder.py", "test_determinism", "Verifies repeated embeddings are identical (atol=1e-6).", "PASSED"],
        ["test_embedder.py", "test_invalid_input", "Verifies TypeError on non-list or non-string elements.", "PASSED"],
        ["test_embedder.py", "test_base_interface", "Verifies custom BaseEmbedder subclassing flexibility.", "PASSED"],
        ["test_integration.py", "test_pipeline_integration", "End-to-end dataset -> OCR -> chunking -> 179 vectors of 384-dim.", "PASSED"],
        ["test_mock_pipeline.py", "8 Unit Tests", "Validates dataset loader, OCR lookup, chunking regex, pipeline doc structure.", "PASSED"],
    ]

    table_tests = doc.add_table(rows=len(test_data), cols=4)
    table_tests.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_widths_test = [Inches(1.3), Inches(1.6), Inches(2.8), Inches(0.8)]

    for row_idx, row in enumerate(test_data):
        for col_idx, text in enumerate(row):
            cell = table_tests.cell(row_idx, col_idx)
            cell.width = col_widths_test[col_idx]
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(text)
            
            if row_idx == 0:
                set_cell_background(cell, HEX_PRIMARY)
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9.5)
            else:
                bg = HEX_BG if row_idx % 2 == 1 else "FFFFFF"
                set_cell_background(cell, bg)
                run.font.size = Pt(9)
                if col_idx == 3:
                    run.bold = True
                    run.font.color.rgb = RGBColor(0x27, 0x67, 0x49) # Dark Green
                else:
                    if col_idx == 0:
                        run.bold = True
                    run.font.color.rgb = COLOR_DARK

    set_table_borders(table_tests, "CBD5E0")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # --- SECTION 7: PROJECT ROADMAP ---
    add_heading_1("7. Project Roadmap & Upcoming Milestones")
    doc.add_paragraph(
        "This project will be delivered across 4 sequential progress reports as scheduled below:"
    ).paragraph_format.space_after = Pt(8)

    roadmap_data = [
        ["Report #", "Milestone Focus", "Key Deliverables & Objectives", "Status"],
        ["Report 1 (Current)", "Ingestion, Grounded OCR & Embeddings", "Sample dataset, loader, grounded OCR engine, chunker, sentence-transformer embedder (15/15 tests passing).", "COMPLETED"],
        ["Report 2", "Vector Database & Persistent Indexing", "Integration of vector store (ChromaDB / FAISS), chunk storage with metadata payload, persistent indexing.", "UPCOMING"],
        ["Report 3", "Semantic Search & Hybrid Retrieval", "Natural language search pipeline, cosine similarity ranking, metadata filtering (by category/date), top-K retrieval.", "PLANNED"],
        ["Report 4", "Real OCR & Full Application UI", "Production PaddleOCR integration, FastAPI backend REST endpoints, interactive web interface (React / Streamlit).", "PLANNED"],
    ]

    table_road = doc.add_table(rows=len(roadmap_data), cols=4)
    table_road.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_widths_road = [Inches(1.2), Inches(1.8), Inches(2.6), Inches(0.9)]

    for row_idx, row in enumerate(roadmap_data):
        for col_idx, text in enumerate(row):
            cell = table_road.cell(row_idx, col_idx)
            cell.width = col_widths_road[col_idx]
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(text)
            
            if row_idx == 0:
                set_cell_background(cell, HEX_SECONDARY)
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9.5)
            else:
                bg = HEX_BG if row_idx % 2 == 1 else "FFFFFF"
                set_cell_background(cell, bg)
                run.font.size = Pt(9)
                if col_idx == 3 and row_idx == 1:
                    run.bold = True
                    run.font.color.rgb = RGBColor(0x27, 0x67, 0x49) # Green
                elif col_idx == 3 and row_idx == 2:
                    run.bold = True
                    run.font.color.rgb = RGBColor(0xDD, 0x6B, 0x20) # Orange
                else:
                    if col_idx == 1:
                        run.bold = True
                    run.font.color.rgb = COLOR_DARK

    set_table_borders(table_road, "CBD5E0")
    doc.add_paragraph().paragraph_format.space_after = Pt(20)

    # --- SIGN-OFF BLOCK ---
    sign_table = doc.add_table(rows=2, cols=2)
    sign_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    sign_table.autofit = False

    s_cell_1 = sign_table.cell(0, 0)
    s_cell_1.width = Inches(3.2)
    p_s1 = s_cell_1.paragraphs[0]
    p_s1.add_run("Student Signature:\n\nDate: ________________________").font.size = Pt(9.5)

    s_cell_2 = sign_table.cell(0, 1)
    s_cell_2.width = Inches(3.2)
    p_s2 = s_cell_2.paragraphs[0]
    p_s2.add_run("Faculty Guide Approval:\n\nDate: ________________________").font.size = Pt(9.5)

    doc.save(docx_path)
    print(f"DOCX Report successfully created at: {docx_path}")

if __name__ == "__main__":
    generate_docx_report()
