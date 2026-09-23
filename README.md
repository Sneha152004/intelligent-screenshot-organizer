# Intelligent Screenshot Organizer

An intelligent screenshot processing and retrieval system using a 3-Layer Storage Architecture (Local File Store, SQLite Metadata Store, ChromaDB Vector Store).

## OCR Engines & Swappable Architecture

The system supports a swappable OCR engine abstraction subclassed from `BaseOCREngine`:

1. **`MockOCREngine` (Default for Testing)**:
   - Provides fast, deterministic, offline mock text extraction grounded in sample screenshots.
   - Ideal for unit testing, CI/CD, and fast development iterations without model loading overhead.

2. **`EasyOCREngine` (Production Deep Learning OCR)**:
   - Leverages EasyOCR (PyTorch-based CRAFT detection + ResNet/LSTM recognition).
   - Features lazy initialization (models are only loaded when text extraction is called).
   - Requires zero C++ OS binary installations.

### How to Select & Swap OCR Engines

#### Option A: Direct Dependency Injection
```python
from src.easy_ocr import EasyOCREngine
from src.pipeline import OCRPipeline

# Instantiate EasyOCR Engine
easy_ocr = EasyOCREngine(languages=["en"], gpu=False)

# Inject into OCRPipeline
pipeline = OCRPipeline(ocr_engine=easy_ocr)
processed_docs = pipeline.process(metadata_path="sample/metadata.csv")
```

#### Option B: Factory Selection
```python
from src.easy_ocr import get_ocr_engine
from src.pipeline import OCRPipeline

# Select via factory ('mock' or 'easyocr')
ocr_engine = get_ocr_engine("easyocr", gpu=False)
pipeline = OCRPipeline(ocr_engine=ocr_engine)
```

#### Option C: AppConfig Selection
```python
from src.config import config
from src.pipeline import OCRPipeline

# Set application-wide default engine ('mock' or 'easyocr')
config.ocr_engine_type = "easyocr"

# OCRPipeline automatically selects EasyOCREngine via get_ocr_engine
pipeline = OCRPipeline()
```

## Screenshot Understanding / VLM

The project uses a swappable VLM abstraction for visual screenshot understanding.

Current implementation:
- MockVLM for deterministic testing

Future implementation:
- Real VLM evaluated separately before production integration

## Running Tests

Run the complete test suite (includes 34 core storage/retrieval tests + 12 OCR/EasyOCR integration and configuration tests + 11 VLM foundation tests):
```bash
python -m pytest -v
```

*Note: Unit tests mock the EasyOCR reader interface and VLM layer, ensuring fast execution without requiring automatic model downloads.*