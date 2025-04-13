#!/usr/bin/env python3
"""
Flashcard Postprocessing Module.

Handles filtering, deduplication, and quality assessment of generated flashcards.
"""

import re
import difflib
from typing import List, Dict, Any, Set, Tuple
from loguru import logger
from tqdm import tqdm

from pdf_to_flashcards.generation.flashcard_generator import Flashcard


def clean_flashcard_text(text: str) -> str:
    """
    Clean flashcard text for comparison.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation except in special cases
    text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()


def similarity_score(text1: str, text2: str) -> float:
    """
    Calculate similarity score between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Clean the texts
    clean1 = clean_flashcard_text(text1)
    clean2 = clean_flashcard_text(text2)
    
    # Calculate similarity
    if not clean1 or not clean2:
        return 0.0
    
    return difflib.SequenceMatcher(None, clean1, clean2).ratio()


def is_duplicate(
    flashcard: Flashcard, 
    existing_cards: List[Flashcard], 
    similarity_threshold: float = 0.8
) -> bool:
    """
    Check if a flashcard is a duplicate of any existing flashcard.
    
    Args:
        flashcard: Flashcard to check
        existing_cards: List of existing flashcards
        similarity_threshold: Similarity threshold for considering a duplicate
        
    Returns:
        True if the flashcard is a duplicate, False otherwise
    """
    for existing in existing_cards:
        # Calculate similarity for question and answer
        q_similarity = similarity_score(flashcard.question, existing.question)
        a_similarity = similarity_score(flashcard.answer, existing.answer)
        
        # Consider it a duplicate if either question or answer is very similar
        if q_similarity > similarity_threshold or a_similarity > similarity_threshold:
            return True
    
    return False


def is_valid_flashcard(
    flashcard: Flashcard,
    min_question_length: int = 10,
    min_answer_length: int = 5,
    max_question_length: int = 500,
    max_answer_length: int = 1000
) -> bool:
    """
    Check if a flashcard is valid based on basic criteria.
    
    Args:
        flashcard: Flashcard to check
        min_question_length: Minimum question length
        min_answer_length: Minimum answer length
        max_question_length: Maximum question length
        max_answer_length: Maximum answer length
        
    Returns:
        True if the flashcard is valid, False otherwise
    """
    # Check question and answer length
    if len(flashcard.question.strip()) < min_question_length:
        return False
    if len(flashcard.answer.strip()) < min_answer_length:
        return False
    
    # Check if question or answer is too long
    if len(flashcard.question) > max_question_length:
        return False
    if len(flashcard.answer) > max_answer_length:
        return False
    
    # Check for empty or nearly empty content
    if not flashcard.question.strip() or not flashcard.answer.strip():
        return False
    
    # Check for default, template or placeholder text
    placeholder_patterns = [
        r'^\[.*\]$',
        r'your (answer|question) here',
        r'enter .* here',
        r'insert .* here',
        r'fill in .* here'
    ]
    
    for pattern in placeholder_patterns:
        if re.search(pattern, flashcard.question.lower()) or re.search(pattern, flashcard.answer.lower()):
            return False
    
    return True


def process_flashcards(
    flashcards: List[Flashcard],
    min_question_length: int = 10,
    min_answer_length: int = 5,
    similarity_threshold: float = 0.8
) -> List[Flashcard]:
    """
    Process flashcards with filtering and deduplication.
    
    Args:
        flashcards: List of flashcards to process
        min_question_length: Minimum question length
        min_answer_length: Minimum answer length
        similarity_threshold: Similarity threshold for deduplication
        
    Returns:
        List of processed flashcards
    """
    if not flashcards:
        logger.warning("No flashcards to process")
        return []
    
    logger.info(f"Processing {len(flashcards)} flashcards")
    
    # Filter invalid flashcards
    valid_cards = []
    for card in tqdm(flashcards, desc="Validating flashcards"):
        if is_valid_flashcard(card, min_question_length, min_answer_length):
            valid_cards.append(card)
    
    logger.info(f"Valid flashcards: {len(valid_cards)}/{len(flashcards)}")
    
    # Deduplicate flashcards
    unique_cards = []
    for card in tqdm(valid_cards, desc="Deduplicating flashcards"):
        if not is_duplicate(card, unique_cards, similarity_threshold):
            unique_cards.append(card)
    
    logger.info(f"Unique flashcards: {len(unique_cards)}/{len(valid_cards)}")
    
    return unique_cards
