#!/usr/bin/env python3
"""
GitHub Models interface for flashcard generation.
"""

import os
import json
import time
import requests
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
from loguru import logger

from pdf_to_flashcards.preprocessing.text_processor import ProcessedChunk
from pdf_to_flashcards.generation.flashcard_generator import Flashcard
from pdf_to_flashcards.utils import Timer


class GitHubModelsClient:
    """Interface for GitHub Models API."""
    
    def __init__(self, model_id: str = "meta/llama-3-8b-instruct", temperature: float = 0.7):
        """
        Initialize GitHub Models interface.
        
        Args:
            model_id: ID of the model to use
            temperature: Temperature for generation (0.0 to 1.0)
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = 2000
        
        # Get the GitHub token from the environment
        self.token = os.environ.get("GITHUB_TOKEN", "")
        if not self.token:
            logger.warning("GITHUB_TOKEN not found in environment. Using default token.")
            # In GitHub Codespaces, this should work automatically
        
        # API endpoint for GitHub Models
        self.api_url = "https://api.github.com/github-models/chat/completions"
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized GitHub Models client with model: {model_id}")
    
    def generate_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate completion from GitHub Models.
        
        Args:
            messages: List of message objects (role and content)
            
        Returns:
            Generated text
        """
        data = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data
            )
            
            duration_ms = int((time.time() - start_time) * 1000)
            logger.debug(f"API response received in {duration_ms}ms")
            
            if response.status_code != 200:
                logger.error(f"GitHub Models API error: {response.status_code} - {response.text}")
                raise ValueError(f"GitHub Models API error: {response.status_code} - {response.text}")
            
            response_data = response.json()
            
            # Extract the generated text
            generated_text = response_data["choices"][0]["message"]["content"]
            
            return generated_text
        
        except Exception as e:
            logger.error(f"Error generating text with GitHub Models: {e}")
            raise
    
    def create_flashcards_for_chunk(
        self, 
        chunk: ProcessedChunk, 
        num_cards: int = 5,
        format_type: str = "basic"
    ) -> List[Flashcard]:
        """
        Generate flashcards for a text chunk.
        
        Args:
            chunk: Processed text chunk
            num_cards: Number of flashcards to generate
            format_type: Format type (basic or cloze)
            
        Returns:
            List of generated flashcards
        """
        # Format the page source as a string
        page_sources = ", ".join(str(page) for page in chunk.source_pages)
        
        # Create system message
        system_msg = "You are an experienced educator who creates high-quality flashcards from textbook content. Focus on important concepts and facts."
        
        if format_type.lower() == "basic":
            # Create basic flashcards prompt
            user_msg = f"""Text from textbook (from page {page_sources}):

{chunk.text}

Generate {num_cards} high-quality flashcards with questions and answers based on the text above.
Each flashcard should cover an important concept or fact from the text.

Format each flashcard like this:
Q: [Clear, specific question]
A: [Concise, accurate answer]

Guidelines:
- Focus on the most important concepts, definitions, and relationships
- Make questions clear and specific
- Answers should be concise but complete
- Cover different aspects of the material
- Include only information that is in the text
"""
        else:
            # Create cloze flashcards prompt
            user_msg = f"""Text from textbook (from page {page_sources}):

{chunk.text}

Generate {num_cards} high-quality cloze deletion flashcards based on the text above.
Each flashcard should cover an important concept or fact from the text.

Format each cloze flashcard like this:
Q: [Sentence with {{...}} around the deleted term]
A: [The term that was deleted]

Guidelines:
- Focus on the most important concepts, definitions, and relationships
- The cloze deletion should target a key term or concept
- The surrounding context should provide meaningful clues
- Cover different aspects of the material
- Include only information that is in the text
"""
        
        # Create messages
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        try:
            # Generate the flashcards
            response_text = self.generate_completion(messages)
            
            # Parse the flashcards from the response
            flashcards = self._parse_flashcards(response_text, chunk, format_type)
            
            return flashcards
            
        except Exception as e:
            logger.error(f"Error generating flashcards for chunk {chunk.id}: {e}")
            return []
    
    def _parse_flashcards(
        self, 
        response_text: str, 
        chunk: ProcessedChunk,
        format_type: str
    ) -> List[Flashcard]:
        """
        Parse flashcards from model response.
        
        Args:
            response_text: Text response from model
            chunk: Source chunk
            format_type: Format type (basic or cloze)
            
        Returns:
            List of flashcards
        """
        cards = []
        
        # Split the response into individual flashcards
        card_texts = response_text.split("Q: ")
        
        # Skip the first split if it doesn't contain a card
        start_idx = 0
        if not card_texts[0].strip() or "A: " not in card_texts[0]:
            start_idx = 1
        
        # Process each card
        for i in range(start_idx, len(card_texts)):
            card_text = card_texts[i].strip()
            if not card_text or "A: " not in card_text:
                continue
            
            # Split into question and answer
            qa_parts = card_text.split("A: ", 1)
            if len(qa_parts) != 2:
                continue
            
            question = qa_parts[0].strip()
            answer = qa_parts[1].strip()
            
            # Create a flashcard
            card_type = "cloze" if format_type.lower() == "cloze" else "basic"
            flashcard = Flashcard(
                question=question,
                answer=answer,
                source_chunk_id=chunk.id,
                source_pages=chunk.source_pages,
                tags=["github-models", card_type, f"page-{'-'.join(map(str, chunk.source_pages))}"],
                metadata={"type": card_type}
            )
            
            cards.append(flashcard)
        
        return cards


def generate_flashcards(
    chunks: List[ProcessedChunk],
    model_id: str = "meta/llama-3-8b-instruct",
    cards_per_chunk: int = 5,
    format_type: str = "basic"
) -> List[Flashcard]:
    """
    Generate flashcards for text chunks using GitHub Models.
    
    Args:
        chunks: List of processed text chunks
        model_id: ID of the model to use
        cards_per_chunk: Number of flashcards per chunk
        format_type: Format type (basic or cloze)
        
    Returns:
        List of generated flashcards
    """
    logger.info(f"Generating flashcards using GitHub Models with model: {model_id}")
    
    # Initialize GitHub Models client
    client = GitHubModelsClient(model_id=model_id)
    
    all_flashcards = []
    
    with Timer("Flashcard generation"):
        for i, chunk in enumerate(tqdm(chunks, desc="Generating flashcards")):
            logger.info(f"Generating flashcards for chunk {i+1}/{len(chunks)}")
            
            try:
                # Generate flashcards for this chunk
                cards = client.create_flashcards_for_chunk(
                    chunk=chunk,
                    num_cards=cards_per_chunk,
                    format_type=format_type
                )
                
                # Add the cards to the result
                all_flashcards.extend(cards)
                
                logger.info(f"Generated {len(cards)} flashcards for chunk {i+1}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Error generating flashcards for chunk {i+1}: {e}")
                # Continue with the next chunk
                time.sleep(1)
    
    logger.info(f"Generated {len(all_flashcards)} flashcards in total")
    
    return all_flashcards
