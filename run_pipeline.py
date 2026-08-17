"""
Example Runner Script for Mock OCR Pipeline
===========================================
Executes the mock OCR pipeline on the sample dataset and displays sample output.
"""

import json
from src.pipeline import OCRPipeline


def main():
    # Initialize pipeline
    pipeline = OCRPipeline()

    # Process dataset
    processed_docs = pipeline.process(metadata_path="sample/metadata.csv")

    # Display sample output for the first processed document
    print("\n" + "=" * 50)
    print("Sample Processed Document Output (First Document):")
    print("=" * 50)
    print(json.dumps(processed_docs[0], indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
