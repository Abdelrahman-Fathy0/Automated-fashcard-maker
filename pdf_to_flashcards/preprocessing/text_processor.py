#!/usr/bin/env python3
"""
Text Processing Module.

Handles cleaning and preprocessing text chunks before flashcard generation.
"""

import re
import os
import nltk
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from loguru import logger
from tqdm import tqdm

# Ensure NLTK data directory exists
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ['NLTK_DATA'] = nltk_data_dir

# Download necessary NLTK packages if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from pdf_to_flashcards.extraction.pdf_extractor import TextChunk


@dataclass
class ProcessedChunk:
    """
    Represents a processed text chunk.
    """
    id: int
    text: str
    source_pages: List[int]
    sentences: List[str] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text or not isinstance(text, str):
        return []
        
    # Clean the text
    text = re.sub(r'\s+', ' ', text)
    
    try:
        # Extract sentences using NLTK
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.warning(f"Error in sentence tokenization: {e}")
        # Fallback method if NLTK fails
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    # Clean sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def extract_key_terms(text: str, min_term_length: int = 4, max_terms: int = 50) -> List[str]:
    """
    Extract key terms from text.
    
    Args:
        text: Input text
        min_term_length: Minimum length of key terms
        max_terms: Maximum number of key terms to extract
        
    Returns:
        List of key terms
    """
    if not text or not isinstance(text, str):
        return []
        
    try:
        # Get stop words from NLTK
        stop_words = set(stopwords.words('english'))
    except Exception as e:
        logger.warning(f"Error loading stopwords: {e}")
        # Fallback basic stopwords if NLTK fails
        stop_words = {'the', 'a', 'an', 'and', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    # Clean the text
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text)
    
    # Extract words
    words = text.split()
    
    # Filter words
    filtered_words = [
        word for word in words 
        if word not in stop_words 
        and len(word) >= min_term_length
        and not word.isdigit()
    ]
    
    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Extract top terms
    key_terms = [word for word, count in sorted_words[:max_terms]]
    
    return key_terms


def process_chunk(chunk: TextChunk) -> ProcessedChunk:
    """
    Process a text chunk.
    
    Args:
        chunk: Input text chunk
        
    Returns:
        Processed chunk
    """
    # Extract sentences
    sentences = extract_sentences(chunk.text)
    
    # Extract key terms
    key_terms = extract_key_terms(chunk.text)
    
    # Create processed chunk
    processed_chunk = ProcessedChunk(
        id=chunk.id,
        text=chunk.text,
        source_pages=chunk.source_pages,
        sentences=sentences,
        key_terms=key_terms
    )
    
    return processed_chunk


def process_chunks(chunks: List[TextChunk]) -> List[ProcessedChunk]:
    """
    Process a list of text chunks.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        List of processed chunks
    """
    logger.info(f"Processing {len(chunks)} chunks...")
    
    processed_chunks = []
    
    for chunk in tqdm(chunks, desc="Processing chunks"):
        processed_chunk = process_chunk(chunk)
        processed_chunks.append(processed_chunk)
    
    logger.info(f"Processed {len(processed_chunks)} chunks")
    
    # Calculate some statistics
    total_sentences = sum(len(chunk.sentences) for chunk in processed_chunks)
    avg_sentences = total_sentences / len(processed_chunks) if processed_chunks else 0
    
    total_key_terms = sum(len(chunk.key_terms) for chunk in processed_chunks)
    
    logger.info(f"Average sentences per chunk: {avg_sentences:.1f}")
    logger.info(f"Total key terms identified: {total_key_terms}")
    
    return processed_chunks


def preprocess_text(chunks: List[TextChunk]) -> List[ProcessedChunk]:
    """
    Preprocess text chunks for flashcard generation.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        List of processed chunks
    """
    return process_chunks(chunks)
