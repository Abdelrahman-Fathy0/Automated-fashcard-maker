#!/usr/bin/env python3
"""
PDF Extraction Module.

Handles extracting text from PDF files with text chunking.
"""

import os
import re
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm


@dataclass
class TextChunk:
    """
    Represents a chunk of text extracted from a PDF.
    """
    id: int
    text: str
    source_pages: List[int]
    metadata: Dict[str, Any] = None


class PDFExtractor:
    """
    Extracts text from PDF files.
    """
    
    def __init__(self, pdf_path: str, chunk_size: int = 5000, chunk_overlap: int = 500):
        """
        Initialize the PDF extractor.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.doc = None
        self.toc = None
        
        # Validate PDF file
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Get file size in MB
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        logger.info(f"Initializing PDFExtractor for file: {pdf_path}")
        logger.info(f"File size: {file_size_mb:.2f} MB")
    
    def __enter__(self):
        """Open the PDF document when entering context."""
        self.doc = fitz.open(self.pdf_path)
        logger.info(f"PDF opened successfully. Pages: {len(self.doc)}")
        
        # Try to get table of contents
        try:
            self.toc = self.doc.get_toc()
            if self.toc:
                logger.info(f"PDF has a table of contents with {len(self.toc)} entries")
            else:
                logger.info("PDF does not have a table of contents")
        except Exception as e:
            logger.warning(f"Error getting table of contents: {e}")
            self.toc = None
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the PDF document when exiting context."""
        if self.doc:
            self.doc.close()
            logger.info("PDF document closed")
    
    def get_page_text(self, page_num: int) -> str:
        """
        Extract text from a specific page.
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            Extracted text from the page
        """
        if not self.doc:
            raise RuntimeError("PDF document not open. Use with context manager.")
        
        if page_num < 0 or page_num >= len(self.doc):
            raise ValueError(f"Invalid page number: {page_num}. PDF has {len(self.doc)} pages.")
        
        page = self.doc[page_num]
        text = page.get_text()
        
        # Clean the text
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean the extracted text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove header/footer patterns (customize based on your PDF)
        # text = re.sub(r'Page \d+ of \d+', '', text)
        
        # Remove other noise
        text = text.replace('\0', '')
        
        return text.strip()
    
    def extract_text_chunks(self) -> List[TextChunk]:
        """
        Extract text from the PDF in chunks.
        
        Returns:
            List of TextChunk objects
        """
        if not self.doc:
            raise RuntimeError("PDF document not open. Use with context manager.")
        
        logger.info(f"Extracting text in chunks of {self.chunk_size} characters with {self.chunk_overlap} character overlap")
        
        chunks = []
        current_chunk = ""
        current_pages = []
        chunk_id = 0
        
        # Process each page
        for page_num in tqdm(range(len(self.doc)), desc="Processing pages"):
            page_text = self.get_page_text(page_num)
            
            # Skip empty pages
            if not page_text.strip():
                continue
            
            # If adding this page would exceed the chunk size, finalize the current chunk
            if len(current_chunk) + len(page_text) > self.chunk_size and current_chunk:
                chunks.append(TextChunk(
                    id=chunk_id,
                    text=current_chunk.strip(),
                    source_pages=current_pages.copy()
                ))
                chunk_id += 1
                
                # Keep the overlap from the previous chunk
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                
                # Keep track of which pages the overlap came from
                # This is a bit tricky as the overlap might span multiple pages
                # For simplicity, we'll just include the last page in the overlap
                if current_pages:
                    current_pages = [current_pages[-1]]
                else:
                    current_pages = []
            
            # Add the current page text to the chunk
            current_chunk += " " + page_text
            current_pages.append(page_num + 1)  # Convert to 1-indexed for user-friendliness
            
            # If the current chunk exceeds the chunk size, create new chunks as needed
            while len(current_chunk) > self.chunk_size:
                chunk_text = current_chunk[:self.chunk_size]
                chunks.append(TextChunk(
                    id=chunk_id,
                    text=chunk_text.strip(),
                    source_pages=current_pages.copy()
                ))
                chunk_id += 1
                
                # Move to the next chunk with overlap
                current_chunk = current_chunk[self.chunk_size - self.chunk_overlap:]
                
                # Keep track of the last page for overlap
                if current_pages:
                    current_pages = [current_pages[-1]]
                else:
                    current_pages = []
        
        # Add the final chunk if there's anything left
        if current_chunk.strip():
            chunks.append(TextChunk(
                id=chunk_id,
                text=current_chunk.strip(),
                source_pages=current_pages
            ))
        
        logger.info(f"Extracted {len(chunks)} chunks from PDF")
        return chunks


def extract_pdf(pdf_path: str, chunk_size: int = 5000, chunk_overlap: int = 500) -> List[TextChunk]:
    """
    Extract text chunks from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum size of text chunks in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of TextChunk objects
    """
    with PDFExtractor(pdf_path, chunk_size, chunk_overlap) as extractor:
        return extractor.extract_text_chunks()
