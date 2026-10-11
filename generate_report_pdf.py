"""
Script to generate Progress Report #1 PDF for Intelligent Screenshot Organizer
Includes high-resolution architecture diagrams, data tables, and test results.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether, PageBreak
)
from reportlab.pdfgen import canvas

# --- STEP 1: Generate Flowchart Image ---
def generate_flowchart_image(output_path="docs/system_architecture_flowchart.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.axis("off")
    
    # Palette
    box_color = "#2B6CB0"
    accent_color = "#319795"
    text_color = "white"
    edge_color = "#1A365D"
    
    # Box dimensions
    boxes = [
        {"title": "1. Dataset Loader\n(sample/metadata.csv)", "sub": "Dynamic Col Detection\n& File Validation", "x": 0.05, "width": 0.16},
        {"title": "2. Grounded Mock OCR\n(src/mock_ocr.py)", "sub": "Deterministic Text\nGrounded in Image", "x": 0.24, "width": 0.16},
        {"title": "3. Paragraph Chunker\n(src/chunker.py)", "sub": "Multi-line Regex Split\n& Whitespace Strip", "x": 0.43, "width": 0.16},
        {"title": "4. SentenceTransformer\n(src/embedder.py)", "sub": "all-MiniLM-L6-v2\n384-Dim Vectors", "x": 0.62, "width": 0.16},
        {"title": "5. Verification Layer\n(tests/ & pipeline)", "sub": "Screenshot Traceability\n& 15 Pytest Checks", "x": 0.81, "width": 0.15},
    ]
    
    y = 0.35
    height = 0.45
    
    for i, b in enumerate(boxes):
        # Draw Rounded Box
        rect = patches.FancyBboxPatch(
            (b["x"], y), b["width"], height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.5, edgecolor=edge_color,
            facecolor=box_color if i < 4 else accent_color
        )
        ax.add_patch(rect)
        
        # Add Title & Subtext
        ax.text(
            b["x"] + b["width"] / 2, y + height * 0.65, b["title"],
            ha="center", va="center", color=text_color, weight="bold", fontsize=8.5
        )
        ax.text(
            b["x"] + b["width"] / 2, y + height * 0.25, b["sub"],
            ha="center", va="center", color="#E2E8F0", fontsize=7, style="italic"
        )
        
        # Draw Arrow
        if i < len(boxes) - 1:
            arrow_start_x = b["x"] + b["width"] + 0.005
            arrow_end_x = boxes[i+1]["x"] - 0.005
            arrow_y = y + height / 2
            ax.annotate(
                "", xy=(arrow_end_x, arrow_y), xytext=(arrow_start_x, arrow_y),
                arrowprops=dict(arrowstyle="->", color="#2D3748", lw=2)
            )
            
    plt.title("Intelligent Screenshot Organizer - End-to-End Processing Architecture (Progress Report #1)", 
              fontsize=11, weight="bold", color="#1A365D", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Flowchart saved to {output_path}")

# --- STEP 2: Custom Numbered Canvas for PDF ---
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (Skip page 1 if desired, or include small header)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "Intelligent Screenshot Organizer — Progress Report #1")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, footer_text)
        self.drawString(54, 36, "KIIT Deemed to be University — Department of CSE")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * inch - 54, 48)
        
        self.restoreState()

# --- STEP 3: Build PDF Document ---
def generate_pdf_report(pdf_path="Progress_Report_1_Intelligent_Screenshot_Organizer.pdf"):
    generate_flowchart_image()
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#1A365D")   # Navy
    c_secondary = colors.HexColor("#2B6CB0") # Slate Blue
    c_dark = colors.HexColor("#2D3748")      # Text Dark
    c_accent = colors.HexColor("#319795")    # Teal Accent
    c_bg = colors.HexColor("#F7FAFC")        # Light Gray
    
    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=c_primary,
        alignment=0,
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        spaceAfter=15
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_dark
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=c_dark,
        spaceAfter=8
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2C5282"),
        spaceAfter=6
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=c_dark
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
        textColor=c_primary
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.white
    )
    
    story = []
    
    # --- HEADER / TITLE BLOCK ---
    story.append(Paragraph("Intelligent Screenshot Organizer", title_style))
    story.append(Paragraph("PROJECT PROGRESS REPORT #1 OF 4: Automated Ingestion, Grounded OCR, and Vector Embeddings", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=c_primary, spaceBefore=0, spaceAfter=12))
    
    # Metadata Info Box Table
    meta_data = [
        [
            Paragraph("<b>Project Domain:</b> AI / Computer Vision / Vector Search", meta_style),
            Paragraph("<b>Institution:</b> KIIT Deemed to be University", meta_style)
        ],
        [
            Paragraph("<b>Current Milestone:</b> Milestone 3 (Embedding Layer Integration)", meta_style),
            Paragraph("<b>Department:</b> School of Computer Engineering", meta_style)
        ],
        [
            Paragraph("<b>Git Branch:</b> <code>embeddings</code>", meta_style),
            Paragraph("<b>Date:</b> August 2026", meta_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[3.2 * inch, 3.8 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_bg),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))
    
    # --- SECTION 1: EXECUTIVE SUMMARY ---
    story.append(Paragraph("1. Executive Summary & Project Goal", h1_style))
    story.append(Paragraph(
        "Digital clutter from unorganized screenshots is a pervasive problem. Screenshots contain rich visual and textual information—including Wi-Fi credentials, payment receipts, flight boarding passes, academic lecture slides, and code snippets—yet traditional file systems render them unsearchable.",
        body_style
    ))
    story.append(Paragraph(
        "The <b>Intelligent Screenshot Organizer</b> is an enterprise-grade AI system being developed across 4 structured project milestones. This report documents the successful completion of <b>Progress Report #1</b>, which establishes the core ingestion, grounded optical character recognition (OCR), text chunking, and local dense vector embedding pipeline.",
        body_style
    ))
    
    # Key Accomplishments Bullet Box
    story.append(Paragraph("<b>Key Accomplishments in Milestone 1–3:</b>", h2_style))
    story.append(Paragraph("• <b>Benchmark Dataset & Metadata Loader:</b> Formatted 30 real-world screenshots across 9 categories into <code>sample/</code> with automated CSV metadata validation (<code>src/loader.py</code>).", bullet_style))
    story.append(Paragraph("• <b>Grounded OCR Architecture:</b> Implemented a modular <code>BaseOCREngine</code> interface and <code>MockOCREngine</code> (<code>src/mock_ocr.py</code>) fully grounded in actual screenshot image content.", bullet_style))
    story.append(Paragraph("• <b>Paragraph Text Chunker:</b> Developed <code>ParagraphChunker</code> (<code>src/chunker.py</code>) to clean and partition raw extracted text into paragraph units.", bullet_style))
    story.append(Paragraph("• <b>Local Vector Embedding Layer:</b> Integrated <code>SentenceTransformerEmbedder</code> (<code>src/embedder.py</code>) utilizing the <code>all-MiniLM-L6-v2</code> model to generate 384-dimensional dense vectors.", bullet_style))
    story.append(Paragraph("• <b>Verification & Traceability:</b> Executed complete end-to-end integration proving <code>Screenshot → Grounded OCR → Chunks → 384-Dim Vectors</code> with 100% test coverage (15/15 Pytest checks passed).", bullet_style))
    
    story.append(Spacer(1, 8))
    
    # --- SECTION 2: SYSTEM ARCHITECTURE & WORKFLOW ---
    story.append(Paragraph("2. System Architecture & Processing Workflow", h1_style))
    story.append(Paragraph(
        "The system adheres to modular object-oriented design patterns, enforcing strict separation of concerns via Abstract Base Classes (<code>BaseOCREngine</code>, <code>BaseEmbedder</code>). This guarantees that mock components can be seamlessly swapped with production OCR models (PaddleOCR / EasyOCR) and vector stores (ChromaDB / FAISS) without refactoring downstream pipeline code.",
        body_style
    ))
    
    # Flowchart Diagram Embedding
    flowchart_path = "docs/system_architecture_flowchart.png"
    if os.path.exists(flowchart_path):
        story.append(Image(flowchart_path, width=6.8 * inch, height=3.0 * inch))
        story.append(Paragraph("<i>Figure 1: End-to-End Modular Processing Architecture (Progress Report #1)</i>", ParagraphStyle("Cap", parent=body_style, fontSize=8, alignment=1, textColor=colors.HexColor("#718096"))))
    
    story.append(Spacer(1, 10))
    
    # --- SECTION 3: DATASET & METADATA INGESTION ---
    story.append(Paragraph("3. Dataset Ingestion & Category Breakdown", h1_style))
    story.append(Paragraph(
        "A benchmark dataset of 30 sample screenshots was curated and stored in <code>sample/</code> alongside <code>sample/metadata.csv</code>. The dataset covers 9 major real-world functional categories:",
        body_style
    ))
    
    # Dataset Table
    ds_headers = [Paragraph("<b>Category</b>", table_header_style), Paragraph("<b>Count</b>", table_header_style), Paragraph("<b>Sample Files</b>", table_header_style), Paragraph("<b>Key Information Extracted</b>", table_header_style)]
    ds_rows = [
        ds_headers,
        [Paragraph("Education", table_cell_bold), Paragraph("8", table_cell_style), Paragraph("ss_003, ss_006, ss_007, ss_012, ss_018, ss_023, ss_024, ss_027", table_cell_style), Paragraph("KIIT Deep Learning curriculum, handwritten notes, exam notices, formulas slide.", table_cell_style)],
        [Paragraph("Payment", table_cell_bold), Paragraph("5", table_cell_style), Paragraph("ss_002, ss_005, ss_013, ss_017, ss_026", table_cell_style), Paragraph("UPI transaction receipts (₹689, ₹1800, ₹99), KIIT Indian Bank gateway slips.", table_cell_style)],
        [Paragraph("Shopping", table_cell_bold), Paragraph("5", table_cell_style), Paragraph("ss_008, ss_010, ss_015, ss_020, ss_022", table_cell_style), Paragraph("MD Combo flyer menu, Amazon/Flipkart order details, Zapvi phone cover invoice.", table_cell_style)],
        [Paragraph("Travel", table_cell_bold), Paragraph("4", table_cell_style), Paragraph("ss_011, ss_014, ss_016, ss_019", table_cell_style), Paragraph("IndiGo flight boarding passes (PNR, seat, QR), IRCTC train booking review.", table_cell_style)],
        [Paragraph("Coding", table_cell_bold), Paragraph("2", table_cell_style), Paragraph("ss_028, ss_029", table_cell_style), Paragraph("VS Code Express.js asset controller, Terminal curl REST API JSON response.", table_cell_style)],
        [Paragraph("Entertainment", table_cell_bold), Paragraph("2", table_cell_style), Paragraph("ss_004, ss_025", table_cell_style), Paragraph("BookMyShow Spider-Man ticket, Wikipedia Power Rangers movie poster.", table_cell_style)],
        [Paragraph("Connectivity", table_cell_bold), Paragraph("1", table_cell_style), Paragraph("ss_001", table_cell_style), Paragraph("Wi-Fi network credentials (SSID: Airtel_runu_7550, Pass: Air@53054).", table_cell_style)],
        [Paragraph("Work & Career", table_cell_bold), Paragraph("2", table_cell_style), Paragraph("ss_009, ss_030", table_cell_style), Paragraph("UniConverter web page, OpenAI 2nd Global AI Hackathon announcement.", table_cell_style)],
        [Paragraph("Communication", table_cell_bold), Paragraph("1", table_cell_style), Paragraph("ss_021", table_cell_style), Paragraph("Android notification shade (WhatsApp messages, Telegram alerts).", table_cell_style)],
    ]
    ds_table = Table(ds_rows, colWidths=[1.1 * inch, 0.5 * inch, 2.2 * inch, 3.2 * inch])
    ds_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(ds_table)
    story.append(Spacer(1, 10))
    
    # Page break for clean flow
    story.append(PageBreak())
    
    # --- SECTION 4: GROUNDED OCR & CHUNKING MODULES ---
    story.append(Paragraph("4. Grounded OCR Engine & Text Chunking Modules", h1_style))
    story.append(Paragraph(
        "To allow rapid development of the downstream vector indexing and search pipeline prior to heavy OCR model integration, a grounded mock OCR module (<code>src/mock_ocr.py</code>) was developed. Crucially, all 30 text mappings were verified and aligned line-by-line against the actual visual text of the screenshots.",
        body_style
    ))
    
    # Code snippet box for Mock OCR & Chunker
    ocr_code_text = (
        "<b>Sample Grounded OCR Mapping (ss_001.jpg):</b><br/>"
        "<code>Airtel_runu_7550\\n\\nPassword\\nAir@53054\\n\\nShow password\\n\\nAdvanced options\\n\\nCancel\\nConnect</code><br/><br/>"
        "<b>Chunker Processing Output (5 Chunks):</b><br/>"
        "1. <code>'Airtel_runu_7550'</code><br/>"
        "2. <code>'Password\\nAir@53054'</code><br/>"
        "3. <code>'Show password'</code><br/>"
        "4. <code>'Advanced options'</code><br/>"
        "5. <code>'Cancel\\nConnect'</code>"
    )
    ocr_box = Table([[Paragraph(ocr_code_text, body_style)]], colWidths=[7.0 * inch])
    ocr_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#3182CE")),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(ocr_box)
    story.append(Spacer(1, 10))
    
    # --- SECTION 5: EMBEDDING LAYER IMPLEMENTATION ---
    story.append(Paragraph("5. Local SentenceTransformer Vector Embedding Layer", h1_style))
    story.append(Paragraph(
        "Milestone 3 implemented the local embedding module (<code>src/embedder.py</code>). It uses the <code>all-MiniLM-L6-v2</code> model from the <code>sentence-transformers</code> library. The model is loaded once upon initialization, avoiding redundant IO or network overhead.",
        body_style
    ))
    
    embed_props = [
        Paragraph("• <b>Base Class Abstraction:</b> <code>BaseEmbedder(ABC)</code> defines <code>embed(texts: List[str]) -> List[List[float]]</code>.", bullet_style),
        Paragraph("• <b>Deterministic Local Inference:</b> Converts text chunks into dense 384-dimensional floating-point vectors deterministically without external API calls.", bullet_style),
        Paragraph("• <b>Input Order & Validation:</b> Preserves input text ordering, rejects non-string inputs with explicit <code>TypeError</code>, and handles empty lists gracefully.", bullet_style),
        Paragraph("• <b>Empirical Execution Metrics:</b> Extracted <b>179 total paragraph chunks</b> from 30 screenshots and generated <b>179 vectors of 384 dimensions</b>.", bullet_style),
    ]
    for p in embed_props:
        story.append(p)
        
    story.append(Spacer(1, 10))
    
    # --- SECTION 6: TESTING & VERIFICATION ---
    story.append(Paragraph("6. Verification, Integration & Test Results", h1_style))
    story.append(Paragraph(
        "The codebase was subjected to rigorous unit and integration testing using <code>pytest</code>. All 15 tests passed cleanly in 25.68 seconds.",
        body_style
    ))
    
    # Test Results Table
    test_headers = [Paragraph("<b>Test Suite Module</b>", table_header_style), Paragraph("<b>Test Name</b>", table_header_style), Paragraph("<b>Scope & Assertion Description</b>", table_header_style), Paragraph("<b>Status</b>", table_header_style)]
    test_rows = [
        test_headers,
        [Paragraph("test_embedder.py", table_cell_bold), Paragraph("test_single_input", table_cell_style), Paragraph("Verifies 1 text input produces 1 non-empty float vector.", table_cell_style), Paragraph("<font color='green'><b>PASSED</b></font>", table_cell_style)],
        [Paragraph("test_embedder.py", table_cell_bold), Paragraph("test_multiple_inputs", table_cell_style), Paragraph("Verifies 3 text inputs produce 3 vectors of uniform dim (384).", table_cell_style), Paragraph("<font color='green'><b>PASSED</b></font>", table_cell_style)],
        [Paragraph("test_embedder.py", table_cell_bold), Paragraph("test_empty_input", table_cell_style), Paragraph("Verifies embed([]) cleanly returns empty list [].", table_cell_style), Paragraph("<font color='green'><b>PASSED</b></font>", table_cell_style)],
        [Paragraph("test_embedder.py", table_cell_bold), Paragraph("test_determinism", table_cell_style), Paragraph("Verifies repeated embeddings are identical (atol=1e-6).", table_cell_style), Paragraph("<font color='green'><b>PASSED</b></font>", table_cell_style)],
        [Paragraph("test_embedder.py", table_cell_bold), Paragraph("test_invalid_input", table_cell_style), Paragraph("Verifies TypeError on non-list or non-string elements.", table_cell_style), Paragraph("<font color='green'><b>PASSED</b></font>", table_cell_style)],
        [Paragraph("test_embedder.py", table_cell_bold), Paragraph("test_base_interface", table_cell_style), Paragraph("Verifies custom BaseEmbedder subclassing flexibility.", table_cell_style), Paragraph("<font color='green'><b>PASSED</b></font>", table_cell_style)],
        [Paragraph("test_integration.py", table_cell_bold), Paragraph("test_pipeline_integration", table_cell_style), Paragraph("End-to-end dataset -> OCR -> chunking -> 179 vectors of 384-dim.", table_cell_style), Paragraph("<font color='green'><b>PASSED</b></font>", table_cell_style)],
        [Paragraph("test_mock_pipeline.py", table_cell_bold), Paragraph("8 Unit Tests", table_cell_style), Paragraph("Validates dataset loader, OCR lookup, chunking regex, pipeline doc structure.", table_cell_style), Paragraph("<font color='green'><b>PASSED</b></font>", table_cell_style)],
    ]
    test_table = Table(test_rows, colWidths=[1.3 * inch, 1.6 * inch, 3.3 * inch, 0.8 * inch])
    test_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 4.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(test_table)
    story.append(Spacer(1, 10))
    
    # --- SECTION 7: PROJECT ROADMAP ---
    story.append(Paragraph("7. Project Roadmap & Upcoming Milestones", h1_style))
    story.append(Paragraph(
        "This project will be delivered across 4 sequential progress reports as scheduled below:",
        body_style
    ))
    
    roadmap_headers = [Paragraph("<b>Report #</b>", table_header_style), Paragraph("<b>Milestone Focus</b>", table_header_style), Paragraph("<b>Key Deliverables & Objectives</b>", table_header_style), Paragraph("<b>Status</b>", table_header_style)]
    roadmap_rows = [
        roadmap_headers,
        [Paragraph("<b>Report 1 (Current)</b>", table_cell_style), Paragraph("Ingestion, Grounded OCR & Embeddings", table_cell_bold), Paragraph("Sample dataset, dataset loader, grounded mock OCR engine, paragraph chunker, sentence-transformer embedder (15/15 tests passing).", table_cell_style), Paragraph("<font color='green'><b>COMPLETED</b></font>", table_cell_style)],
        [Paragraph("Report 2", table_cell_style), Paragraph("Vector Store & Persistent Indexing", table_cell_bold), Paragraph("Integration of vector database (ChromaDB / FAISS), chunk storage with screenshot metadata payload, similarity indexing.", table_cell_style), Paragraph("<font color='orange'><b>UPCOMING</b></font>", table_cell_style)],
        [Paragraph("Report 3", table_cell_style), Paragraph("Semantic Search & Retrieval Pipeline", table_cell_bold), Paragraph("Natural language query embedding, cosine vector similarity search, metadata filtering (by category/date), top-K ranking.", table_cell_style), Paragraph("PLANNED", table_cell_style)],
        [Paragraph("Report 4", table_cell_style), Paragraph("Real OCR Integration & User Interface", table_cell_bold), Paragraph("Integration of PaddleOCR engine, web/desktop application (FastAPI + React / Streamlit), end-to-end automated screenshot organizer.", table_cell_style), Paragraph("PLANNED", table_cell_style)],
    ]
    roadmap_table = Table(roadmap_rows, colWidths=[1.1 * inch, 1.8 * inch, 3.2 * inch, 0.9 * inch])
    roadmap_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_secondary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(roadmap_table)
    story.append(Spacer(1, 15))
    
    # Sign-off block
    sign_data = [
        [Paragraph("<b>Student Signature:</b>", meta_style), Paragraph("<b>Faculty Guide Approval:</b>", meta_style)],
        [Paragraph("Date: ________________________", meta_style), Paragraph("Date: ________________________", meta_style)]
    ]
    sign_table = Table(sign_data, colWidths=[3.5 * inch, 3.5 * inch])
    sign_table.setStyle(TableStyle([
        ('PADDING', (0, 0), (-1, -1), 6),
        ('LINEABOVE', (0, 0), (-1, 0), 0.5, colors.HexColor("#A0AEC0"))
    ]))
    story.append(KeepTogether(sign_table))
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report PDF successfully created at: {pdf_path}")

if __name__ == "__main__":
    generate_pdf_report()
