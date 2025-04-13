#!/usr/bin/env python3
"""
Flashcard Generator for Textbooks.

Generates flashcards from PDF textbooks using GitHub Models API.
"""

import os
import sys
import json
import argparse
from datetime import datetime
import time
from loguru import logger
from tqdm import tqdm

# Add the current directory to the Python path
sys.path.append(os.getcwd())

from pdf_to_flashcards.extraction.pdf_extractor import extract_pdf
from pdf_to_flashcards.preprocessing.text_processor import preprocess_text
from pdf_to_flashcards.generation.github_models import generate_flashcards
from pdf_to_flashcards.postprocessing.flashcard_processor import process_flashcards
from pdf_to_flashcards.export.export_flashcards import export_flashcards
from pdf_to_flashcards.utils import Timer, setup_logging


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate flashcards from textbooks")
    
    parser.add_argument("pdf_file", help="Path to the PDF file")
    parser.add_argument("--model", default="meta/llama-3-8b-instruct", 
                        choices=["meta/llama-3-8b-instruct", "meta/llama-3-70b-instruct",
                                 "anthropic/claude-instant-1.2", "mistralai/mistral-small",
                                 "mistralai/mistral-medium", "mistralai/mistral-large"],
                        help="GitHub Models model to use")
    parser.add_argument("--cards-per-chunk", type=int, default=5,
                        help="Number of flashcards per chunk")
    parser.add_argument("--format", choices=["basic", "cloze"], default="basic",
                        help="Flashcard format type")
    parser.add_argument("--output", default="output",
                        help="Output directory for saving results")
    parser.add_argument("--max-chunks", type=int, 
                        help="Maximum number of chunks to process (for testing)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    return parser.parse_args()


def main():
    """Run the flashcard generation pipeline."""
    args = parse_arguments()
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level)
    
    # Create timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create output directory with timestamp
    output_dir = os.path.join(args.output, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate PDF file
    if not os.path.exists(args.pdf_file):
        logger.error(f"PDF file not found: {args.pdf_file}")
        sys.exit(1)
    
    # Save the run parameters
    with open(os.path.join(output_dir, "run_info.txt"), "w") as f:
        f.write(f"PDF File: {args.pdf_file}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Cards per chunk: {args.cards_per_chunk}\n")
        f.write(f"Format: {args.format}\n")
        f.write(f"Started at: {timestamp}\n")
        if args.max_chunks:
            f.write(f"Max chunks: {args.max_chunks}\n")
    
    print(f"\n===== FLASHCARD GENERATION =====")
    print(f"PDF File: {args.pdf_file}")
    print(f"Model: {args.model}")
    print(f"Cards per chunk: {args.cards_per_chunk}")
    print(f"Format: {args.format}")
    print(f"Output directory: {output_dir}")
    if args.max_chunks:
        print(f"Max chunks: {args.max_chunks}")
    print("=" * 33 + "\n")
    
    # Step 1: Extract text from PDF
    with Timer("PDF extraction"):
        raw_chunks = extract_pdf(args.pdf_file)
        logger.info(f"Extracted {len(raw_chunks)} raw text chunks from PDF")
        
        if args.max_chunks:
            raw_chunks = raw_chunks[:args.max_chunks]
            logger.info(f"Limited to {args.max_chunks} chunks for testing")
    
    # Step 2: Text preprocessing
    with Timer("Text preprocessing"):
        processed_chunks = preprocess_text(raw_chunks)
        logger.info(f"Preprocessed {len(processed_chunks)} text chunks")
    
    # Step 3: Generate flashcards
    with Timer("Flashcard generation"):
        flashcards = generate_flashcards(
            chunks=processed_chunks,
            model_id=args.model,
            cards_per_chunk=args.cards_per_chunk,
            format_type=args.format
        )
        logger.info(f"Generated {len(flashcards)} flashcards")
        
        # Save raw flashcards
        raw_cards_file = os.path.join(output_dir, "raw_flashcards.json")
        with open(raw_cards_file, "w", encoding="utf-8") as f:
            json.dump([{
                "question": card.question,
                "answer": card.answer,
                "source_pages": card.source_pages,
                "source_chunk_id": card.source_chunk_id,
                "quality_score": card.quality_score,
                "tags": card.tags
            } for card in flashcards], f, indent=2, ensure_ascii=False)
        
        print(f"Raw flashcards saved to: {raw_cards_file}")
    
    # Step 4: Post-process flashcards
    with Timer("Post-processing flashcards"):
        processed_cards = process_flashcards(
            flashcards=flashcards,
            min_question_length=10,
            min_answer_length=10,
            similarity_threshold=0.8
        )
        logger.info(f"After post-processing: {len(processed_cards)}/{len(flashcards)} flashcards retained")
        
        # Save processed flashcards
        processed_cards_file = os.path.join(output_dir, "processed_flashcards.json")
        with open(processed_cards_file, "w", encoding="utf-8") as f:
            json.dump([{
                "question": card.question,
                "answer": card.answer,
                "source_pages": card.source_pages,
                "quality_score": card.quality_score,
                "tags": card.tags
            } for card in processed_cards], f, indent=2, ensure_ascii=False)
        
        print(f"Processed flashcards saved to: {processed_cards_file}")
    
    # Step 5: Export in multiple formats
    with Timer("Exporting flashcards"):
        export_formats = ["csv", "json", "anki"]
        exported_files = export_flashcards(
            flashcards=processed_cards,
            output_dir=output_dir,
            formats=export_formats
        )
        
        # Log exported files
        logger.info(f"Exported flashcards to {len(exported_files)} formats:")
        for format_name, file_path in exported_files.items():
            if file_path:
                logger.info(f"  {format_name}: {os.path.basename(file_path)}")
    
    # Step 6: Save completion information
    completion_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(output_dir, "run_info.txt"), "a") as f:
        f.write(f"Completed at: {completion_time}\n")
        f.write(f"Total chunks processed: {len(processed_chunks)}\n")
        f.write(f"Total flashcards generated: {len(flashcards)}\n")
        f.write(f"Flashcards after quality filtering: {len(processed_cards)}\n")
        for format_name, file_path in exported_files.items():
            if file_path:
                f.write(f"Exported to {format_name}: {os.path.basename(file_path)}\n")
    
    print(f"\n===== PROCESSING COMPLETE =====")
    print(f"PDF File: {args.pdf_file}")
    print(f"Total chunks processed: {len(processed_chunks)}")
    print(f"Total flashcards generated: {len(flashcards)}")
    print(f"Flashcards after quality filtering: {len(processed_cards)}")
    print(f"Output directory: {output_dir}")
    
    print("\nExported formats:")
    for format_name, file_path in exported_files.items():
        if file_path:
            print(f"- {format_name.upper()}: {os.path.basename(file_path)}")
    
    print("\nNotes:")
    print(f"- Model used: {args.model}")
    print("- To use the Anki file, import it into Anki using File > Import")
    print("- CSV file can be imported into spreadsheet applications")
    print("=" * 33)


if __name__ == "__main__":
    main()
