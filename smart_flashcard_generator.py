#!/usr/bin/env python3
"""
Advanced Flashcard Generator for CS Textbooks with Smart Content Analysis
- Generates variable number of flashcards based on content density
- Differentiates between core concepts and supplementary examples
- Uses content analysis to prioritize important information
"""

import os
import sys
import json
import fitz  # PyMuPDF
import time
import re
from datetime import datetime
from tqdm import tqdm

# Azure AI Inference SDK imports
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Create output directory
OUTPUT_DIR = "smart_flashcards"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Available Models
AVAILABLE_MODELS = {
    "llama3-small": "Meta-Llama-3-8B-Instruct",
    "llama3-large": "Meta-Llama-3-70B-Instruct",
    "llama3.1-small": "Meta-Llama-3.1-8B-Instruct",
    "llama3.1-large": "Meta-Llama-3.1-70B-Instruct",
    "mistral-small": "Mistral-small",
    "mistral-large": "Mistral-large",
    "gpt4o-mini": "gpt-4o-mini",
    "gpt4o": "gpt-4o",
    "phi3-small": "Phi-3-small-8k-instruct"
}

class TextChunk:
    """Represents a chunk of text from the PDF."""
    def __init__(self, id, text, source_pages):
        self.id = id
        self.text = text
        self.source_pages = source_pages
        self.content_density = 0  # 0-10 score indicating density of important concepts
        self.key_concepts = []  # List of key concepts
        self.is_analyzed = False  # Whether chunk has been analyzed

class Flashcard:
    """Represents a flashcard with question and answer."""
    def __init__(self, question, answer, source_pages=None, importance=5, tags=None, concept=None):
        self.question = question
        self.answer = answer 
        self.source_pages = source_pages or []
        self.importance = importance  # 1-10 score indicating importance
        self.tags = tags or []
        self.concept = concept  # The key concept this card relates to

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

def analyze_chunk_content(chunk, client, model_name):
    """
    Analyze chunk to determine content density and extract key concepts.
    Returns the updated chunk with content_density and key_concepts set.
    """
    # Format the page source as a string
    page_sources = ", ".join(str(page) for page in chunk.source_pages)
    
    # Create system message
    system_message = SystemMessage(
        content="You are an expert computer science educator skilled at identifying important concepts in textbook content."
    )
    
    # Create user message
    user_message = UserMessage(
        content=f"""Analyze this computer science textbook content from page(s) {page_sources}:

{chunk.text}

Perform the following analysis:
1. Rate the density of important concepts in this content on a scale of 1-10 (where 10 means extremely dense with important concepts)
2. Identify key concepts that students should learn (focus on core knowledge, not trivial examples or anecdotes)
3. For each key concept, indicate whether it's:
   - CORE: Essential knowledge that must be learned
   - SUPPLEMENTARY: Helpful but not essential
   - EXAMPLE: Just an example that illustrates a concept

Respond in this exact JSON format:
{{
  "content_density": <number 1-10>,
  "key_concepts": [
    {{
      "concept": "<concept name>",
      "type": "<CORE, SUPPLEMENTARY, or EXAMPLE>",
      "importance": <number 1-10>
    }}
  ],
  "recommended_flashcard_count": <number of recommended flashcards based on content>
}}

Only output valid JSON that can be parsed with json.loads().
"""
    )
    
    try:
        # Call the client to analyze the content
        response = client.complete(
            messages=[system_message, user_message],
            model=model_name,
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=1500
        )
        
        # Extract the generated text
        content = response.choices[0].message.content
        
        # Parse the JSON response
        try:
            # Extract JSON from the response (in case there's extra text)
            json_match = re.search(r'({[\s\S]*})', content)
            if json_match:
                content = json_match.group(1)
            
            analysis = json.loads(content)
            
            # Update the chunk with analysis information
            chunk.content_density = analysis["content_density"]
            chunk.key_concepts = analysis["key_concepts"]
            chunk.recommended_flashcard_count = analysis.get("recommended_flashcard_count", 5)
            chunk.is_analyzed = True
            
            print(f"Analyzed chunk {chunk.id}:")
            print(f"  - Content density: {chunk.content_density}")
            print(f"  - Key concepts: {len(chunk.key_concepts)}")
            print(f"  - Recommended flashcards: {chunk.recommended_flashcard_count}")
            
            return chunk
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response for chunk {chunk.id}: {e}")
            print(f"Response content: {content}")
            # Set default values
            chunk.content_density = 5
            chunk.key_concepts = []
            chunk.recommended_flashcard_count = 5
            
            return chunk
    
    except Exception as e:
        print(f"Error analyzing chunk {chunk.id}: {e}")
        # Set default values
        chunk.content_density = 5
        chunk.key_concepts = []
        chunk.recommended_flashcard_count = 5
        
        return chunk

def generate_flashcards_for_chunk(chunk, client, model_name):
    """Generate flashcards for a text chunk using content analysis."""
    # If chunk hasn't been analyzed, use a default number of flashcards
    if not chunk.is_analyzed or not hasattr(chunk, 'recommended_flashcard_count'):
        num_cards = 5
    else:
        num_cards = chunk.recommended_flashcard_count
    
    # Format the page source as a string
    page_sources = ", ".join(str(page) for page in chunk.source_pages)
    
    # Format key concepts as a string for the prompt
    key_concepts_str = ""
    if hasattr(chunk, 'key_concepts') and chunk.key_concepts:
        key_concepts_str = "Key concepts in this content:\n"
        for i, concept in enumerate(chunk.key_concepts):
            importance = concept.get('importance', 5)
            concept_type = concept.get('type', 'CORE')
            key_concepts_str += f"{i+1}. {concept['concept']} ({concept_type}, Importance: {importance}/10)\n"
    
    # Create system message
    system_message = SystemMessage(
        content="""You are an expert computer science educator who creates high-quality flashcards from textbook content.
Focus on creating flashcards that test understanding of core concepts, not memorization of trivial details or examples.
"""
    )
    
    # Create user message
    user_message = UserMessage(
        content=f"""Text from CS textbook (from page {page_sources}):

{chunk.text}

{key_concepts_str}

Generate {num_cards} high-quality flashcards based on the text above, prioritizing CORE concepts over SUPPLEMENTARY ones or EXAMPLES.
Focus on concepts that students must understand to master the material.
Avoid creating flashcards on trivial details, implementation specifics (like how many GPUs were used), or anecdotal examples.

Format each flashcard as JSON objects in an array:
[
  {{
    "question": "Clear, specific question",
    "answer": "Concise, accurate answer",
    "concept": "The key concept this flashcard tests",
    "importance": <1-10 rating of importance>
  }}
]

Only output valid JSON that can be parsed with json.loads().
"""
    )
    
    try:
        # Call the client to generate flashcards
        response = client.complete(
            messages=[system_message, user_message],
            model=model_name,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract the generated text
        content = response.choices[0].message.content
        
        # Parse the JSON flashcards
        try:
            # Extract JSON from the response (in case there's extra text)
            json_match = re.search(r'(\[[\s\S]*\])', content)
            if json_match:
                content = json_match.group(1)
            
            flashcard_data = json.loads(content)
            
            # Convert to Flashcard objects
            flashcards = []
            for card_data in flashcard_data:
                flashcard = Flashcard(
                    question=card_data["question"],
                    answer=card_data["answer"],
                    source_pages=chunk.source_pages,
                    importance=card_data.get("importance", 5),
                    concept=card_data.get("concept", ""),
                    tags=[f"page-{'-'.join(map(str, chunk.source_pages))}", "cs", "textbook"]
                )
                flashcards.append(flashcard)
            
            return flashcards
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON flashcards for chunk {chunk.id}: {e}")
            print(f"Response content: {content}")
            return []
    
    except Exception as e:
        print(f"Error generating flashcards for chunk {chunk.id}: {e}")
        return []

def post_process_flashcards(all_flashcards, client, model_name, min_importance=3):
    """
    Post-process flashcards to remove duplicates, assess quality,
    and ensure comprehensive coverage of concepts.
    """
    if not all_flashcards:
        return []
    
    print(f"Post-processing {len(all_flashcards)} flashcards...")
    
    # Filter out low-importance flashcards
    filtered_cards = [card for card in all_flashcards if card.importance >= min_importance]
    print(f"Filtered out {len(all_flashcards) - len(filtered_cards)} low-importance flashcards")
    
    # Group by concept to detect duplicates
    concept_groups = {}
    for card in filtered_cards:
        concept = card.concept.lower() if card.concept else "uncategorized"
        if concept not in concept_groups:
            concept_groups[concept] = []
        concept_groups[concept].append(card)
    
    # For concepts with multiple cards, keep the highest importance cards
    deduplicated_cards = []
    for concept, cards in concept_groups.items():
        if len(cards) > 1:
            # Sort by importance (descending)
            sorted_cards = sorted(cards, key=lambda x: x.importance, reverse=True)
            # Keep the best card(s) for this concept
            deduplicated_cards.append(sorted_cards[0])
            # If there are significantly different angles/aspects, keep those too
            if len(sorted_cards) > 1 and sorted_cards[1].importance >= 7:
                deduplicated_cards.append(sorted_cards[1])
        else:
            deduplicated_cards.extend(cards)
    
    print(f"Removed {len(filtered_cards) - len(deduplicated_cards)} duplicate flashcards")
    
    # Sort by importance
    final_cards = sorted(deduplicated_cards, key=lambda x: x.importance, reverse=True)
    
    return final_cards

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
            "source_pages": card.source_pages,
            "importance": card.importance,
            "concept": card.concept,
            "tags": card.tags
        } for card in flashcards], f, indent=2)
    
    output_files["json"] = json_file
    
    # Export to CSV
    csv_file = os.path.join(output_dir, "flashcards.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("Question,Answer,Source Pages,Importance,Concept,Tags\n")
        for card in flashcards:
            pages = "-".join(str(p) for p in card.source_pages)
            tags = ",".join(card.tags)
            f.write(f"\"{card.question}\",\"{card.answer}\",\"{pages}\",\"{card.importance}\",\"{card.concept}\",\"{tags}\"\n")
    
    output_files["csv"] = csv_file
    
    # Export to Anki format
    anki_file = os.path.join(output_dir, "flashcards_anki.txt")
    with open(anki_file, "w", encoding="utf-8") as f:
        for card in flashcards:
            pages = "-".join(str(p) for p in card.source_pages)
            tags = " ".join(card.tags)
            f.write(f"{card.question}; {card.answer}; {tags} importance:{card.importance}\n")
    
    output_files["anki"] = anki_file
    
    return output_files

def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python smart_flashcard_generator.py <pdf_file> [--max-chunks N] [--model MODEL_NAME] [--min-importance N]")
        print("\nAvailable models:")
        for alias, model in AVAILABLE_MODELS.items():
            print(f"  {alias}: {model}")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    # Parse optional arguments
    max_chunks = None
    model_name = AVAILABLE_MODELS.get("gpt4o-mini")  # Default to GPT-4o mini
    min_importance = 3  # Default minimum importance threshold (1-10)
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--max-chunks" and i + 1 < len(sys.argv):
            max_chunks = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            model_input = sys.argv[i + 1]
            # Check if it's an alias or full name
            if model_input in AVAILABLE_MODELS:
                model_name = AVAILABLE_MODELS[model_input]
            else:
                # Assume it's a full model name
                model_name = model_input
            i += 2
        elif sys.argv[i] == "--min-importance" and i + 1 < len(sys.argv):
            min_importance = int(sys.argv[i + 1])
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
        f.write(f"Model: {model_name}\n")
        f.write(f"Min Importance: {min_importance}\n")
        if max_chunks:
            f.write(f"Max chunks: {max_chunks}\n")
        f.write(f"Started at: {timestamp}\n")
    
    print(f"\n===== SMART CS FLASHCARD GENERATION =====")
    print(f"PDF File: {pdf_file}")
    print(f"Model: {model_name}")
    print(f"Min Importance: {min_importance}")
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
    token = os.environ.get("GITHUB_TOKEN")  # Should be automatically set in Codespace
    
    if not token:
        print("ERROR: GITHUB_TOKEN environment variable not found.")
        print("This should be automatically set in GitHub Codespaces.")
        sys.exit(1)
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token)
    )
    
    # Step 3: Analyze content density for each chunk
    print(f"Analyzing content density and key concepts using model: {model_name}")
    
    analyzed_chunks = []
    for chunk in tqdm(chunks, desc="Analyzing chunks"):
        print(f"Analyzing chunk {chunk.id + 1}/{len(chunks)} (pages {', '.join(str(p) for p in chunk.source_pages)})")
        analyzed_chunk = analyze_chunk_content(chunk, client, model_name)
        analyzed_chunks.append(analyzed_chunk)
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    # Step 4: Generate flashcards based on content analysis
    all_flashcards = []
    
    print(f"Generating flashcards using model: {model_name}")
    
    for chunk in tqdm(analyzed_chunks, desc="Generating flashcards"):
        print(f"Generating flashcards for chunk {chunk.id + 1}/{len(chunks)} (density: {chunk.content_density}/10)")
        
        flashcards = generate_flashcards_for_chunk(
            chunk=chunk,
            client=client,
            model_name=model_name
        )
        
        if flashcards:
            all_flashcards.extend(flashcards)
            print(f"Generated {len(flashcards)} flashcards for chunk {chunk.id + 1}")
        else:
            print(f"No flashcards generated for chunk {chunk.id + 1}")
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    print(f"Generated a total of {len(all_flashcards)} flashcards")
    
    # Step 5: Post-process flashcards
    final_flashcards = post_process_flashcards(
        all_flashcards=all_flashcards,
        client=client,
        model_name=model_name,
        min_importance=min_importance
    )
    
    print(f"After post-processing: {len(final_flashcards)} flashcards")
    
    # Step 6: Export flashcards
    exported_files = export_flashcards(final_flashcards, output_dir)
    
    # Save completion information
    completion_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(output_dir, "run_info.txt"), "a") as f:
        f.write(f"Completed at: {completion_time}\n")
        f.write(f"Total chunks processed: {len(chunks)}\n")
        f.write(f"Total flashcards generated: {len(all_flashcards)}\n")
        f.write(f"Flashcards after post-processing: {len(final_flashcards)}\n")
        for format_name, file_path in exported_files.items():
            f.write(f"Exported to {format_name}: {os.path.basename(file_path)}\n")
    
    print(f"\n===== PROCESSING COMPLETE =====")
    print(f"Total chunks processed: {len(chunks)}")
    print(f"Total flashcards generated: {len(all_flashcards)}")
    print(f"Flashcards after post-processing: {len(final_flashcards)}")
    print(f"Output directory: {output_dir}")
    
    print("\nExported formats:")
    for format_name, file_path in exported_files.items():
        print(f"- {format_name.upper()}: {os.path.basename(file_path)}")
    
    print("\nNotes:")
    print(f"- Used model: {model_name}")
    print("- To use the Anki file, import it into Anki using File > Import")
    print("- Set the field separator to semicolon (;) when importing")
    print("=" * 40)

if __name__ == "__main__":
    main()
