#!/usr/bin/env python3
"""
Flashcard Generator using GitHub Models via Azure AI Inference SDK.
Fixed version with correct model naming.
"""

import os
import sys
import json
import fitz  # PyMuPDF
import time
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any

# Azure AI Inference SDK imports
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Create output directory
OUTPUT_DIR = "github_flashcards"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TextChunk:
    """Represents a chunk of text from the PDF."""
    def __init__(self, id, text, source_pages):
        self.id = id
        self.text = text
        self.source_pages = source_pages

class Flashcard:
    """Represents a flashcard with question and answer."""
    def __init__(self, question, answer, source_pages=None):
        self.question = question
        self.answer = answer 
        self.source_pages = source_pages or []

def extract_text_from_pdf(pdf_path, chunk_size=5000, chunk_overlap=500):
    """Extract text from PDF in chunks."""
    print(f"Extracting text from PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Get PDF file size
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    print(f"PDF opened successfully. Pages: {len(doc)}")
    
    chunks = []
    current_chunk = ""
    current_pages = []
    chunk_id = 0
    
    # Process each page
    for page_num in tqdm(range(len(doc)), desc="Processing pages"):
        page = doc[page_num]
        page_text = page.get_text()
        
        # Skip empty pages
        if not page_text.strip():
            continue
        
        # If adding this page would exceed the chunk size, finalize the current chunk
        if len(current_chunk) + len(page_text) > chunk_size and current_chunk:
            chunks.append(TextChunk(
                id=chunk_id,
                text=current_chunk.strip(),
                source_pages=current_pages.copy()
            ))
            chunk_id += 1
            
            # Keep the overlap from the previous chunk
            overlap_start = max(0, len(current_chunk) - chunk_overlap)
            current_chunk = current_chunk[overlap_start:]
            
            # Keep track of which page the overlap came from
            if current_pages:
                current_pages = [current_pages[-1]]
            else:
                current_pages = []
        
        # Add the current page text to the chunk
        current_chunk += " " + page_text
        current_pages.append(page_num + 1)  # Convert to 1-indexed
        
        # If the current chunk exceeds the chunk size, create new chunks as needed
        while len(current_chunk) > chunk_size:
            chunk_text = current_chunk[:chunk_size]
            chunks.append(TextChunk(
                id=chunk_id,
                text=chunk_text.strip(),
                source_pages=current_pages.copy()
            ))
            chunk_id += 1
            
            # Move to the next chunk with overlap
            current_chunk = current_chunk[chunk_size - chunk_overlap:]
            
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
    
    # Close the PDF
    doc.close()
    print(f"Extracted {len(chunks)} chunks from PDF")
    
    return chunks

def generate_flashcards_for_chunk(chunk, client, model_id, num_cards=5):
    """Generate flashcards for a text chunk using GitHub Models."""
    # Format the page source as a string
    page_sources = ", ".join(str(page) for page in chunk.source_pages)
    
    # Create system message
    system_message = SystemMessage(
        content="You are an experienced educator who creates high-quality flashcards from textbook content."
    )
    
    # Create user message
    user_message = UserMessage(
        content=f"""Text from textbook (from page {page_sources}):

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
    )
    
    try:
        # Call the client to generate flashcards
        response = client.complete(
            model=model_id,
            messages=[system_message, user_message],
            temperature=0.7,
            max_tokens=1500
        )
        
        # Extract the generated text
        content = response.choices[0].message.content
        
        # Parse the flashcards from the response
        flashcards = parse_flashcards(content, chunk.source_pages)
        
        return flashcards
    
    except Exception as e:
        print(f"Error generating flashcards for chunk {chunk.id}: {e}")
        return []

def parse_flashcards(content, source_pages):
    """Parse flashcards from the model response."""
    flashcards = []
    
    # Split by Q: to get individual cards
    card_texts = content.split("Q: ")
    
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
        flashcard = Flashcard(
            question=question,
            answer=answer,
            source_pages=source_pages
        )
        
        flashcards.append(flashcard)
    
    return flashcards

def export_flashcards(flashcards, output_dir):
    """Export flashcards to multiple formats."""
    if not flashcards:
        print("No flashcards to export")
        return {}
    
    os.makedirs(output_dir, exist_ok=True)
    output_files = {}
    
    # Export to JSON
    json_file = os.path.join(output_dir, "flashcards.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump([{
            "question": card.question,
            "answer": card.answer,
            "source_pages": card.source_pages
        } for card in flashcards], f, indent=2)
    
    output_files["json"] = json_file
    
    # Export to CSV
    csv_file = os.path.join(output_dir, "flashcards.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("Question,Answer,Source Pages\n")
        for card in flashcards:
            pages = "-".join(str(p) for p in card.source_pages)
            f.write(f"\"{card.question}\",\"{card.answer}\",\"{pages}\"\n")
    
    output_files["csv"] = csv_file
    
    # Export to Anki format
    anki_file = os.path.join(output_dir, "flashcards_anki.txt")
    with open(anki_file, "w", encoding="utf-8") as f:
        for card in flashcards:
            pages = "-".join(str(p) for p in card.source_pages)
            f.write(f"{card.question}; {card.answer}; page-{pages}\n")
    
    output_files["anki"] = anki_file
    
    return output_files

def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python fixed_github_flashcard.py <pdf_file> [--max-chunks N] [--cards-per-chunk N] [--model MODEL_ID]")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    # Parse optional arguments
    max_chunks = None
    cards_per_chunk = 5
    model_id = "llama-3-8b-instruct"  # Corrected format without vendor prefix
    
    # Map of user-friendly names to actual model IDs
    model_map = {
        "llama-3-8b": "llama-3-8b-instruct",
        "llama-3-70b": "llama-3-70b-instruct",
        "mistral-small": "mistral-small",
        "mistral-medium": "mistral-medium",
        "mistral-large": "mistral-large",
        "claude-instant": "claude-instant-1.2"
    }
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--max-chunks" and i + 1 < len(sys.argv):
            max_chunks = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--cards-per-chunk" and i + 1 < len(sys.argv):
            cards_per_chunk = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            model_input = sys.argv[i + 1]
            # Check if it's a short name or full name
            if model_input in model_map:
                model_id = model_map[model_input]
            elif "/" in model_input:
                # Remove vendor prefix if present
                model_id = model_input.split("/")[1]
            else:
                model_id = model_input
            i += 2
        else:
            i += 1
    
    # Create timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save run information
    with open(os.path.join(output_dir, "run_info.txt"), "w") as f:
        f.write(f"PDF File: {pdf_file}\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Cards per chunk: {cards_per_chunk}\n")
        if max_chunks:
            f.write(f"Max chunks: {max_chunks}\n")
        f.write(f"Started at: {timestamp}\n")
    
    print(f"\n===== GITHUB MODELS FLASHCARD GENERATION =====")
    print(f"PDF File: {pdf_file}")
    print(f"Model: {model_id}")
    print(f"Cards per chunk: {cards_per_chunk}")
    print(f"Output directory: {output_dir}")
    if max_chunks:
        print(f"Max chunks: {max_chunks}")
    print("=" * 48)
    
    # Step 1: Extract text from PDF
    chunks = extract_text_from_pdf(pdf_file)
    
    if max_chunks:
        print(f"Limiting to {max_chunks} chunks")
        chunks = chunks[:max_chunks]
    
    # Step 2: Set up GitHub Models client (using Azure AI Inference)
    print(f"Setting up Azure AI Inference client")
    endpoint = "https://models.inference.ai.azure.com"
    token = os.environ["GITHUB_TOKEN"]  # Should be automatically set in Codespace
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token)
    )
    
    # Try to list available models if possible
    try:
        print("Attempting to list available models...")
        # This is a placeholder - the SDK might not have this feature
        # We'll try a simple completion with a small prompt to see model info
        test_response = client.complete(
            model="llama-3-8b-instruct",  # Try this corrected format
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="List the available models.")
            ],
            temperature=0.7,
            max_tokens=10
        )
        print("Successfully connected to model API!")
    except Exception as e:
        print(f"Error listing models: {e}")
        # Try alternate model formats
        print("Will try alternating model formats during generation...")
    
    # Step 3: Generate flashcards
    all_flashcards = []
    
    # List of model formats to try if the primary one fails
    model_formats = [
        model_id,
        f"meta/{model_id}",
        model_id.replace("-instruct", ""),
        model_id.replace("llama-", "llama3-"),
    ]
    
    print(f"Generating flashcards...")
    print(f"Will try these model formats if needed: {model_formats}")
    
    for chunk in tqdm(chunks, desc="Generating flashcards"):
        print(f"Processing chunk {chunk.id + 1}/{len(chunks)} (pages {', '.join(str(p) for p in chunk.source_pages)})")
        
        # Try each model format until one works
        success = False
        for model_format in model_formats:
            try:
                print(f"Trying model format: {model_format}")
                flashcards = generate_flashcards_for_chunk(
                    chunk=chunk,
                    client=client,
                    model_id=model_format,
                    num_cards=cards_per_chunk
                )
                
                if flashcards:
                    all_flashcards.extend(flashcards)
                    print(f"Generated {len(flashcards)} flashcards using model: {model_format}")
                    success = True
                    # Remember the working model format for future chunks
                    model_formats = [model_format]
                    break
            except Exception as e:
                print(f"Error with model format '{model_format}': {e}")
        
        if not success:
            print(f"Failed to generate flashcards for chunk {chunk.id + 1} with any model format")
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    print(f"Generated a total of {len(all_flashcards)} flashcards")
    
    # Step 4: Export flashcards
    exported_files = export_flashcards(all_flashcards, output_dir)
    
    # Save completion information
    completion_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(output_dir, "run_info.txt"), "a") as f:
        f.write(f"Completed at: {completion_time}\n")
        f.write(f"Total chunks processed: {len(chunks)}\n")
        f.write(f"Total flashcards generated: {len(all_flashcards)}\n")
        for format_name, file_path in exported_files.items():
            f.write(f"Exported to {format_name}: {os.path.basename(file_path)}\n")
    
    print(f"\n===== PROCESSING COMPLETE =====")
    print(f"Total chunks processed: {len(chunks)}")
    print(f"Total flashcards generated: {len(all_flashcards)}")
    print(f"Output directory: {output_dir}")
    
    print("\nExported formats:")
    for format_name, file_path in exported_files.items():
        print(f"- {format_name.upper()}: {os.path.basename(file_path)}")
    
    print("\nNotes:")
    print(f"- Used model: {model_id}")
    print("- To use the Anki file, import it into Anki using File > Import")
    print("- Set the field separator to semicolon (;) when importing")
    print("=" * 40)

if __name__ == "__main__":
    main()
