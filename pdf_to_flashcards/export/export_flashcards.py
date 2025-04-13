#!/usr/bin/env python3
"""
Flashcard Export Module.

Handles exporting flashcards to various formats.
"""

import os
import json
import csv
import time
from typing import List, Dict, Any, Optional
from loguru import logger

from pdf_to_flashcards.generation.flashcard_generator import Flashcard


def export_to_json(flashcards: List[Flashcard], output_dir: str) -> str:
    """
    Export flashcards to JSON format.
    
    Args:
        flashcards: List of flashcards to export
        output_dir: Output directory
        
    Returns:
        Path to the exported file
    """
    if not flashcards:
        logger.warning("No flashcards to export to JSON")
        return ""
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "flashcards.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([card.to_dict() for card in flashcards], f, indent=2, ensure_ascii=False)
    
    logger.info(f"Exported {len(flashcards)} flashcards to JSON: {output_file}")
    
    return output_file


def export_to_csv(flashcards: List[Flashcard], output_dir: str) -> str:
    """
    Export flashcards to CSV format.
    
    Args:
        flashcards: List of flashcards to export
        output_dir: Output directory
        
    Returns:
        Path to the exported file
    """
    if not flashcards:
        logger.warning("No flashcards to export to CSV")
        return ""
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "flashcards.csv")
    
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(["Question", "Answer", "Pages", "Tags"])
        
        # Write flashcards
        for card in flashcards:
            pages_str = ",".join(map(str, card.source_pages))
            tags_str = ",".join(card.tags)
            writer.writerow([card.question, card.answer, pages_str, tags_str])
    
    logger.info(f"Exported {len(flashcards)} flashcards to CSV: {output_file}")
    
    return output_file


def export_to_anki(flashcards: List[Flashcard], output_dir: str) -> str:
    """
    Export flashcards to Anki-compatible format.
    
    Args:
        flashcards: List of flashcards to export
        output_dir: Output directory
        
    Returns:
        Path to the exported file
    """
    if not flashcards:
        logger.warning("No flashcards to export to Anki format")
        return ""
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "flashcards_anki.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for card in flashcards:
            # Format for Anki: question; answer; tags
            pages_str = ",".join(map(str, card.source_pages))
            tags_str = " ".join(card.tags)
            tags_with_pages = f"{tags_str} page_{pages_str.replace(',', '_')}"
            
            # Write in Anki format (question; answer; tags)
            f.write(f"{card.question}; {card.answer}; {tags_with_pages}\n")
    
    logger.info(f"Exported {len(flashcards)} flashcards to Anki format: {output_file}")
    
    return output_file


def export_flashcards(
    flashcards: List[Flashcard], 
    output_dir: str, 
    formats: List[str] = ["json", "csv", "anki"]
) -> Dict[str, str]:
    """
    Export flashcards to multiple formats.
    
    Args:
        flashcards: List of flashcards to export
        output_dir: Output directory
        formats: List of formats to export to
        
    Returns:
        Dictionary mapping format names to exported file paths
    """
    if not flashcards:
        logger.warning("No flashcards to export")
        return {}
    
    logger.info(f"Exporting {len(flashcards)} flashcards to formats: {', '.join(formats)}")
    
    result = {}
    
    # Export to each format
    for format_name in formats:
        if format_name.lower() == "json":
            result["json"] = export_to_json(flashcards, output_dir)
        elif format_name.lower() == "csv":
            result["csv"] = export_to_csv(flashcards, output_dir)
        elif format_name.lower() == "anki":
            result["anki"] = export_to_anki(flashcards, output_dir)
        else:
            logger.warning(f"Unsupported export format: {format_name}")
    
    return result
