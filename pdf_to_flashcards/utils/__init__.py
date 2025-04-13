#!/usr/bin/env python3
"""
Utility functions for PDF flashcard generation.
"""

import time
import os
import sys  # Add this missing import
import logging
from contextlib import contextmanager
from loguru import logger

class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        logger.info(f"{self.name} took {elapsed_time:.2f} seconds")


def setup_logging(level="INFO"):
    """Set up logging configuration."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level
    )
    
    # Also log to file
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/flashcards.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="1 week"
    )
