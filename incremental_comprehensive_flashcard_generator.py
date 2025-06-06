#!/usr/bin/env python3
"""
Incremental Comprehensive Flashcard Generator for CS Textbooks

This script preserves all the sophisticated features of the comprehensive flashcard generator:
- Hierarchical concept mapping to track parent-child relationships
- Comprehensive coverage analysis to ensure all related components are covered
- Incremental processing with progress tracking between runs

The script can be stopped and restarted, continuing from where it left off.
"""

import os
import sys
import json
import fitz  # PyMuPDF
import time
import re
import argparse
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# Azure AI Inference SDK imports
import azure.ai.inference as inference

# Create output directories
OUTPUT_DIR = "comprehensive_flashcards"
PROGRESS_DIR = "progress"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROGRESS_DIR, exist_ok=True)

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

class ConceptNode:
    """Represents a concept with hierarchical relationships."""
    def __init__(self, name, parent=None, importance=5, concept_type="CORE"):
        self.name = name
        self.parent = parent
        self.children = []
        self.importance = importance
        self.type = concept_type  # CORE, SUPPLEMENTARY, or EXAMPLE
        self.covered = False  # Whether this concept has a flashcard
        self.source_pages = []  # Pages where this concept appears

    def add_child(self, child):
        """Add a child concept to this concept."""
        self.children.append(child)
        
    def to_dict(self):
        """Convert to a dictionary representation."""
        return {
            "name": self.name,
            "parent": self.parent.name if self.parent else None,
            "children": [child.name for child in self.children],
            "importance": self.importance,
            "type": self.type,
            "covered": self.covered,
            "source_pages": self.source_pages
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
        self.concept_hierarchy = []  # Hierarchical structure of concepts

class Flashcard:
    """Represents a flashcard with question and answer."""
    def __init__(self, question, answer, source_pages=None, importance=5, tags=None, concept=None, concept_path=None):
        self.question = question
        self.answer = answer 
        self.source_pages = source_pages or []
        self.importance = importance  # 1-10 score indicating importance
        self.tags = tags or []
        self.concept = concept  # The key concept this card relates to
        self.concept_path = concept_path or []  # Path from root concept to this concept

def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description='Generate comprehensive flashcards from a PDF file')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--model', default='gpt4o-mini', help='AI model to use')
    parser.add_argument('--max-requests', type=int, default=140, help='Maximum API requests per run')
    return parser.parse_args()

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

def analyze_chunk_with_concept_hierarchy(chunk, client, model_name):
    """
    Analyze chunk to determine content density and extract hierarchical concept organization.
    Identifies parent-child relationships between concepts.
    """
    # Format the page source as a string
    page_sources = ", ".join(str(page) for page in chunk.source_pages)
    
    # Create the prompt
    prompt = f"""You are an expert computer science educator skilled at identifying important concepts and their relationships in textbook content.

Analyze this computer science textbook content from page(s) {page_sources} with HIGHLY DETAILED FOCUS ON COMPONENT RELATIONSHIPS:

{chunk.text}

Perform the following analysis:
1. Rate the density of important concepts in this content on a scale of 1-10 (where 10 means extremely dense with important concepts)
2. Identify key concepts that students should learn
3. For each concept, determine:
   - Whether it's a CORE, SUPPLEMENTARY, or EXAMPLE concept
   - Its importance (1-10 scale)
   - Its parent concept (if any)
   - Its component parts or sub-concepts (if any)

CRITICAL: Identify ALL components or elements that make up larger concepts. For example:
- If "System Bus" is mentioned, identify ALL its components (data bus, address bus, control bus)
- If "CPU Registers" are discussed, list ALL individual registers mentioned
- If an algorithm has multiple steps or a protocol has multiple phases, identify EACH step/phase

Respond in this exact JSON format:
{{
  "content_density": <number 1-10>,
  "recommended_flashcard_count": <number of recommended flashcards>,
  "concept_hierarchy": [
    {{
      "concept": "<concept name>",
      "type": "<CORE, SUPPLEMENTARY, or EXAMPLE>",
      "importance": <number 1-10>,
      "parent": "<parent concept name or null>",
      "components": [
        "<component/child concept name>",
        ...
      ],
      "definition": "<brief definition of this concept>"
    }}
  ]
}}"""
    
    try:
        # Call the client to analyze the content
        response = client.chat_completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2  # Lower temperature for more consistent analysis
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
            chunk.recommended_flashcard_count = analysis.get("recommended_flashcard_count", 5)
            chunk.concept_hierarchy = analysis.get("concept_hierarchy", [])
            chunk.is_analyzed = True
            
            print(f"Analyzed chunk {chunk.id}:")
            print(f"  - Content density: {chunk.content_density}")
            print(f"  - Concepts identified: {len(chunk.concept_hierarchy)}")
            print(f"  - Recommended flashcards: {chunk.recommended_flashcard_count}")
            
            return chunk
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response for chunk {chunk.id}: {e}")
            print(f"Response content: {content}")
            # Set default values
            chunk.content_density = 5
            chunk.concept_hierarchy = []
            chunk.recommended_flashcard_count = 5
            
            return chunk
    
    except Exception as e:
        print(f"Error analyzing chunk {chunk.id}: {e}")
        # Set default values
        chunk.content_density = 5
        chunk.concept_hierarchy = []
        chunk.recommended_flashcard_count = 5
        
        return chunk

def build_concept_forest(analyzed_chunks):
    """
    Build a forest of concept trees from all analyzed chunks.
    Merges concepts that appear in multiple chunks.
    """
    # First, collect all concepts
    all_concepts = {}
    
    # Process each chunk to build the initial set of concepts
    for chunk in analyzed_chunks:
        if not hasattr(chunk, 'concept_hierarchy') or not chunk.concept_hierarchy:
            continue
            
        for concept_data in chunk.concept_hierarchy:
            concept_name = concept_data["concept"].strip()
            
            # Create or update the concept
            if concept_name not in all_concepts:
                all_concepts[concept_name] = {
                    "name": concept_name,
                    "type": concept_data.get("type", "CORE"),
                    "importance": concept_data.get("importance", 5),
                    "parents": [],
                    "components": concept_data.get("components", []),
                    "definition": concept_data.get("definition", ""),
                    "source_pages": chunk.source_pages.copy(),
                    "covered": False
                }
            else:
                # Update existing concept with new information
                all_concepts[concept_name]["importance"] = max(
                    all_concepts[concept_name]["importance"],
                    concept_data.get("importance", 5)
                )
                all_concepts[concept_name]["components"].extend(
                    [c for c in concept_data.get("components", []) 
                     if c not in all_concepts[concept_name]["components"]]
                )
                all_concepts[concept_name]["source_pages"].extend(
                    [p for p in chunk.source_pages if p not in all_concepts[concept_name]["source_pages"]]
                )
            
            # Add parent relationship
            parent = concept_data.get("parent")
            if parent and parent.strip() and parent.strip() != concept_name:
                if parent.strip() not in all_concepts[concept_name]["parents"]:
                    all_concepts[concept_name]["parents"].append(parent.strip())
    
    # Now, create ConceptNode objects and establish relationships
    concept_nodes = {}
    
    # First pass: create nodes
    for name, data in all_concepts.items():
        concept_nodes[name] = ConceptNode(
            name=name,
            importance=data["importance"],
            concept_type=data["type"]
        )
        concept_nodes[name].source_pages = data["source_pages"]
    
    # Second pass: establish parent-child relationships
    for name, data in all_concepts.items():
        node = concept_nodes[name]
        for parent_name in data["parents"]:
            if parent_name in concept_nodes:
                node.parent = concept_nodes[parent_name]
                concept_nodes[parent_name].add_child(node)
        
        # Add component relationships that aren't already established
        for component_name in data["components"]:
            if component_name in concept_nodes and component_name != name:
                if concept_nodes[component_name] not in node.children:
                    node.add_child(concept_nodes[component_name])
                    # Only set parent if it doesn't already have one
                    if concept_nodes[component_name].parent is None:
                        concept_nodes[component_name].parent = node
    
    # Find root concepts (those without parents)
    root_concepts = [node for node in concept_nodes.values() if node.parent is None]
    
    print(f"Built concept forest with {len(concept_nodes)} concepts and {len(root_concepts)} root concepts")
    
    return concept_nodes, root_concepts

def generate_comprehensive_flashcards(chunk, concept_nodes, client, model_name):
    """
    Generate flashcards for a chunk with a focus on comprehensive coverage.
    Uses the concept hierarchy to ensure all related concepts are covered.
    """
    # If chunk hasn't been analyzed, use a default number of flashcards
    if not chunk.is_analyzed or not hasattr(chunk, 'recommended_flashcard_count'):
        num_cards = 5
    else:
        num_cards = chunk.recommended_flashcard_count
    
    # Format the page source as a string
    page_sources = ", ".join(str(page) for page in chunk.source_pages)
    
    # Extract concepts relevant to this chunk
    chunk_concepts = []
    if hasattr(chunk, 'concept_hierarchy') and chunk.concept_hierarchy:
        for concept_data in chunk.concept_hierarchy:
            concept_name = concept_data["concept"].strip()
            if concept_name in concept_nodes:
                chunk_concepts.append(concept_nodes[concept_name])
                
                # Also add component concepts that should be covered
                for component_name in concept_data.get("components", []):
                    if component_name in concept_nodes and component_name != concept_name:
                        chunk_concepts.append(concept_nodes[component_name])
    
    # Create a concept report for the prompt
    concept_report = "Key concepts in this content:\n"
    for i, concept in enumerate(chunk_concepts):
        # For each concept, list its components/children
        components_str = ""
        if concept.children:
            components_str = "\n   Components: " + ", ".join(child.name for child in concept.children)
            
        parent_str = f" (part of {concept.parent.name})" if concept.parent else ""
            
        concept_report += f"{i+1}. {concept.name}{parent_str} ({concept.type}, Importance: {concept.importance}/10){components_str}\n"
    
    # Create the prompt
    prompt = f"""You are an expert computer science educator who creates high-quality flashcards from textbook content.
Focus on creating flashcards that test understanding of core concepts and ensure COMPREHENSIVE coverage of ALL related components.

Text from CS textbook (from page {page_sources}):

{chunk.text}

{concept_report}

Generate {num_cards} high-quality flashcards based on the text above with COMPREHENSIVE coverage of all related concepts and components:

CRITICAL: Make sure ALL components of a larger concept are covered. For example:
- If "System Bus" is mentioned, create cards for ALL components (data bus, address bus, control bus)
- If "CPU Registers" are discussed, include cards for ALL registers
- If an algorithm has multiple steps, ensure EACH key step gets coverage

Format each flashcard as JSON objects in an array:
[
  {{
    "question": "Clear, specific question",
    "answer": "Concise, accurate answer",
    "concept": "The key concept this flashcard tests",
    "related_concepts": ["List", "of", "related", "concepts"],
    "importance": <1-10 rating of importance>
  }}
]"""
    
    try:
        # Call the client to generate flashcards
        response = client.chat_completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
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
                # Determine the concept path (hierarchy)
                concept_name = card_data.get("concept", "").strip()
                concept_path = []
                
                if concept_name in concept_nodes:
                    # Build path from this concept up to the root
                    current = concept_nodes[concept_name]
                    while current:
                        concept_path.insert(0, current.name)
                        current = current.parent
                
                flashcard = Flashcard(
                    question=card_data["question"],
                    answer=card_data["answer"],
                    source_pages=chunk.source_pages,
                    importance=card_data.get("importance", 5),
                    concept=concept_name,
                    concept_path=concept_path,
                    tags=[f"page-{'-'.join(map(str, chunk.source_pages))}", "cs", "textbook"]
                )
                
                # Add related concept tags
                related_concepts = card_data.get("related_concepts", [])
                for related in related_concepts:
                    if related and related.strip() not in flashcard.tags:
                        flashcard.tags.append(related.strip())
                
                flashcards.append(flashcard)
            
            # Mark concepts as covered
            for card in flashcards:
                if card.concept in concept_nodes:
                    concept_nodes[card.concept].covered = True
            
            return flashcards
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON flashcards for chunk {chunk.id}: {e}")
            print(f"Response content: {content}")
            return []
    
    except Exception as e:
        print(f"Error generating flashcards for chunk {chunk.id}: {e}")
        return []

def create_missing_component_flashcards(concept_nodes, client, model_name):
    """
    Create flashcards for important components/concepts that are not yet covered.
    This ensures comprehensive coverage even for concepts spread across chunks.
    """
    # Identify uncovered but important concepts
    missing_concepts = []
    
    for name, node in concept_nodes.items():
        # If this is an important concept that isn't covered
        if not node.covered and node.importance >= 6 and node.type == "CORE":
            missing_concepts.append(node)
            continue
            
        # If this is a parent concept with uncovered important children
        if node.covered and node.children:
            uncovered_children = [child for child in node.children 
                                 if not child.covered and child.importance >= 4]
            if uncovered_children:
                missing_concepts.extend(uncovered_children)
    
    if not missing_concepts:
        print("No missing concept flashcards needed - comprehensive coverage achieved!")
        return []
    
    print(f"Creating flashcards for {len(missing_concepts)} uncovered important concepts")
    
    supplementary_flashcards = []
    
    # Process in batches to avoid too large prompts
    batch_size = 5
    for i in range(0, len(missing_concepts), batch_size):
        batch = missing_concepts[i:i+batch_size]
        
        # Build concept descriptions
        concept_descriptions = ""
        for j, concept in enumerate(batch):
            # Get parent description
            parent_info = f"Part of: {concept.parent.name}" if concept.parent else "Top-level concept"
            # Get children description
            children_info = f"Has components: {', '.join(child.name for child in concept.children)}" if concept.children else "No sub-components"
            # Source pages
            pages_info = f"From page(s): {', '.join(str(p) for p in concept.source_pages)}"
            
            concept_descriptions += f"{j+1}. {concept.name} - {parent_info}. {children_info}. {pages_info}\n"
        
        # Create the prompt
        prompt = f"""You are an expert computer science educator creating flashcards for important concepts.

Create flashcards for these important computer science concepts that need coverage:

{concept_descriptions}

For each concept, create a high-quality flashcard that:
1. Tests understanding of the core concept (not just memorization)
2. Provides a clear, concise answer
3. Relates the concept to its parent/larger system when relevant
4. Distinguishes it from related concepts when appropriate

Format each flashcard as JSON objects in an array:
[
  {{
    "question": "Clear, specific question about the concept",
    "answer": "Concise, accurate answer",
    "concept": "Exact concept name from the list",
    "importance": <1-10 rating of importance>
  }}
]"""
        
        try:
            # Call the client to generate supplementary flashcards
            response = client.chat_completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            # Extract the generated text
            content = response.choices[0].message.content
            
            # Parse the JSON flashcards
            try:
                # Extract JSON from the response
                json_match = re.search(r'(\[[\s\S]*\])', content)
                if json_match:
                    content = json_match.group(1)
                
                flashcard_data = json.loads(content)
                
                # Convert to Flashcard objects
                for card_data in flashcard_data:
                    concept_name = card_data.get("concept", "").strip()
                    
                    # Find the matching concept node
                    concept_node = None
                    for node in batch:
                        if node.name.lower() == concept_name.lower():
                            concept_node = node
                            break
                    
                    if not concept_node:
                        # Try a fuzzy match
                        for node in batch:
                            if node.name.lower() in concept_name.lower() or concept_name.lower() in node.name.lower():
                                concept_node = node
                                break
                    
                    # If we found a matching concept node
                    if concept_node:
                        # Build the concept path
                        concept_path = []
                        current = concept_node
                        while current:
                            concept_path.insert(0, current.name)
                            current = current.parent
                        
                        # Create the flashcard
                        flashcard = Flashcard(
                            question=card_data["question"],
                            answer=card_data["answer"],
                            source_pages=concept_node.source_pages,
                            importance=card_data.get("importance", concept_node.importance),
                            concept=concept_node.name,
                            concept_path=concept_path,
                            tags=["cs", "textbook", "supplementary"]
                        )
                        
                        # Add page tags
                        if concept_node.source_pages:
                            page_tag = f"page-{'-'.join(map(str, concept_node.source_pages))}"
                            flashcard.tags.append(page_tag)
                        
                        supplementary_flashcards.append(flashcard)
                        
                        # Mark the concept as covered
                        concept_node.covered = True
                
            except json.JSONDecodeError as e:
                print(f"Error parsing supplementary flashcards JSON: {e}")
                print(f"Response content: {content}")
                
        except Exception as e:
            print(f"Error generating supplementary flashcards: {e}")
    
    print(f"Generated {len(supplementary_flashcards)} supplementary flashcards for missing concepts")
    return supplementary_flashcards

def analyze_coverage(concept_nodes):
    """Analyze how well the concepts are covered by flashcards."""
    total_concepts = len(concept_nodes)
    covered_concepts = sum(1 for node in concept_nodes.values() if node.covered)
    
    # Analyze by importance
    high_importance = [node for node in concept_nodes.values() if node.importance >= 7]
    high_covered = sum(1 for node in high_importance if node.covered)
    
    # Analyze by type
    core_concepts = [node for node in concept_nodes.values() if node.type == "CORE"]
    core_covered = sum(1 for node in core_concepts if node.covered)
    
    # Create coverage report
    report = {
        "total_concepts": total_concepts,
        "covered_concepts": covered_concepts,
        "coverage_percentage": round(covered_concepts / total_concepts * 100, 1) if total_concepts > 0 else 0,
        "high_importance_total": len(high_importance),
        "high_importance_covered": high_covered,
        "high_importance_percentage": round(high_covered / len(high_importance) * 100, 1) if high_importance else 0,
        "core_total": len(core_concepts),
        "core_covered": core_covered,
        "core_percentage": round(core_covered / len(core_concepts) * 100, 1) if core_concepts else 0
    }
    
    print(f"\nCONCEPT COVERAGE ANALYSIS:")
    print(f"Overall: {report['covered_concepts']}/{report['total_concepts']} concepts covered ({report['coverage_percentage']}%)")
    print(f"High Importance: {report['high_importance_covered']}/{report['high_importance_total']} concepts covered ({report['high_importance_percentage']}%)")
    print(f"Core Concepts: {report['core_covered']}/{report['core_total']} concepts covered ({report['core_percentage']}%)")
    
    return report

def export_flashcards(flashcards, concept_coverage, output_dir):
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
            "concept_path": card.concept_path,
            "tags": card.tags
        } for card in flashcards], f, indent=2)
    
    output_files["json"] = json_file
    
    # Export to CSV
    csv_file = os.path.join(output_dir, "flashcards.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("Question,Answer,Source Pages,Importance,Concept,Concept Path,Tags\n")
        for card in flashcards:
            pages = "-".join(str(p) for p in card.source_pages)
            concept_path = " > ".join(card.concept_path)
            tags = ",".join(card.tags)
            f.write(f"\"{card.question}\",\"{card.answer}\",\"{pages}\",\"{card.importance}\",\"{card.concept}\",\"{concept_path}\",\"{tags}\"\n")
    
    output_files["csv"] = csv_file
    
    # Export to Anki format
    anki_file = os.path.join(output_dir, "flashcards_anki.txt")
    with open(anki_file, "w", encoding="utf-8") as f:
        for card in flashcards:
            pages = "-".join(str(p) for p in card.source_pages)
            tags = " ".join(card.tags)
            f.write(f"{card.question}; {card.answer}; {tags} importance:{card.importance}\n")
    
    output_files["anki"] = anki_file
    
    # Export to markdown
    md_file = os.path.join(output_dir, "flashcards.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Comprehensive CS Flashcards\n\n")
        for i, card in enumerate(flashcards, 1):
            f.write(f"## {i}. {card.concept}\n\n")
            f.write(f"**Question:** {card.question}\n\n")
            f.write(f"**Answer:** {card.answer}\n\n")
            f.write(f"**Importance:** {card.importance}/10\n\n")
            f.write(f"**Pages:** {', '.join(str(p) for p in card.source_pages)}\n\n")
            if card.concept_path:
                f.write(f"**Concept Path:** {' > '.join(card.concept_path)}\n\n")
            f.write("---\n\n")
    
    output_files["markdown"] = md_file
    
    # Export concept coverage analysis
    coverage_file = os.path.join(output_dir, "concept_coverage.json")
    with open(coverage_file, "w", encoding="utf-8") as f:
        json.dump(concept_coverage, f, indent=2)
    
    output_files["coverage"] = coverage_file
    
    return output_files

def load_progress():
    """Load progress data from previous runs."""
    progress_file = os.path.join(PROGRESS_DIR, "progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading progress file: {e}")
            return create_new_progress()
    else:
        return create_new_progress()

def create_new_progress():
    """Create a new progress tracking structure."""
    return {
        "chunks_processed": 0,
        "total_chunks": 0,
        "analyzed_chunks": [],
        "flashcards": [],
        "concept_nodes": {},
        "completed": False,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }

def save_progress(progress_data):
    """Save progress data for future runs."""
    # Update the last_updated timestamp
    progress_data["last_updated"] = datetime.now().isoformat()
    
    # Save to file
    progress_file = os.path.join(PROGRESS_DIR, "progress.json")
    with open(progress_file, "w", encoding="utf-8") as f:
        # Convert concept_nodes to serializable format
        if "concept_nodes_dict" not in progress_data and "concept_nodes" in progress_data:
            if progress_data["concept_nodes"] and not isinstance(progress_data["concept_nodes"], dict):
                progress_data["concept_nodes_dict"] = {
                    name: node.to_dict() for name, node in progress_data["concept_nodes"].items()
                }
            del progress_data["concept_nodes"]
        
        # Convert analyzed_chunks to serializable format
        if "analyzed_chunks" in progress_data:
            serializable_chunks = []
            for chunk in progress_data["analyzed_chunks"]:
                if isinstance(chunk, TextChunk):
                    serializable_chunks.append({
                        "id": chunk.id,
                        "text": chunk.text,
                        "source_pages": chunk.source_pages,
                        "content_density": chunk.content_density,
                        "key_concepts": chunk.key_concepts,
                        "is_analyzed": chunk.is_analyzed,
                        "concept_hierarchy": chunk.concept_hierarchy,
                        "recommended_flashcard_count": getattr(chunk, "recommended_flashcard_count", 5)
                    })
                else:
                    # It's already in dictionary form
                    serializable_chunks.append(chunk)
            progress_data["analyzed_chunks"] = serializable_chunks
        
        # Convert flashcards to serializable format
        if "flashcards" in progress_data:
            serializable_flashcards = []
            for card in progress_data["flashcards"]:
                if isinstance(card, Flashcard):
                    serializable_flashcards.append({
                        "question": card.question,
                        "answer": card.answer,
                        "source_pages": card.source_pages,
                        "importance": card.importance,
                        "concept": card.concept,
                        "concept_path": card.concept_path,
                        "tags": card.tags
                    })
                else:
                    # It's already in dictionary form
                    serializable_flashcards.append(card)
            progress_data["flashcards"] = serializable_flashcards
        
        # Save to file
        json.dump(progress_data, f, indent=2)

def reconstruct_objects(progress_data):
    """Reconstruct objects from serialized data."""
    # Reconstruct TextChunk objects
    if "analyzed_chunks" in progress_data:
        reconstructed_chunks = []
        for chunk_data in progress_data["analyzed_chunks"]:
            chunk = TextChunk(
                id=chunk_data["id"],
                text=chunk_data["text"],
                source_pages=chunk_data["source_pages"]
            )
            chunk.content_density = chunk_data.get("content_density", 0)
            chunk.key_concepts = chunk_data.get("key_concepts", [])
            chunk.is_analyzed = chunk_data.get("is_analyzed", False)
            chunk.concept_hierarchy = chunk_data.get("concept_hierarchy", [])
            if "recommended_flashcard_count" in chunk_data:
                chunk.recommended_flashcard_count = chunk_data["recommended_flashcard_count"]
            reconstructed_chunks.append(chunk)
        progress_data["analyzed_chunks"] = reconstructed_chunks
    
    # Reconstruct Flashcard objects
    if "flashcards" in progress_data:
        reconstructed_cards = []
        for card_data in progress_data["flashcards"]:
            card = Flashcard(
                question=card_data["question"],
                answer=card_data["answer"],
                source_pages=card_data.get("source_pages", []),
                importance=card_data.get("importance", 5),
                tags=card_data.get("tags", []),
                concept=card_data.get("concept", None),
                concept_path=card_data.get("concept_path", [])
            )
            reconstructed_cards.append(card)
        progress_data["flashcards"] = reconstructed_cards
    
    # Reconstruct ConceptNode objects if needed
    if "concept_nodes_dict" in progress_data:
        nodes = {}
        # First pass: create all nodes
        for name, data in progress_data["concept_nodes_dict"].items():
            nodes[name] = ConceptNode(
                name=name,
                importance=data.get("importance", 5),
                concept_type=data.get("type", "CORE")
            )
            nodes[name].covered = data.get("covered", False)
            nodes[name].source_pages = data.get("source_pages", [])
        
        # Second pass: establish relationships
        for name, data in progress_data["concept_nodes_dict"].items():
            if data.get("parent") and data["parent"] in nodes:
                nodes[name].parent = nodes[data["parent"]]
            
            for child_name in data.get("children", []):
                if child_name in nodes and nodes[child_name] not in nodes[name].children:
                    nodes[name].add_child(nodes[child_name])
        
        progress_data["concept_nodes"] = nodes
        del progress_data["concept_nodes_dict"]
    
    return progress_data

def main():
    args = setup_argparse()
    
    # Get the model name
    if args.model in AVAILABLE_MODELS:
        model_name = AVAILABLE_MODELS[args.model]
    else:
        model_name = args.model  # Use as-is if not in the alias list
    
    print(f"\n===== INCREMENTAL COMPREHENSIVE FLASHCARD GENERATION =====")
    print(f"PDF File: {args.pdf_path}")
    print(f"Model: {model_name}")
    print(f"Maximum API requests per run: {args.max_requests}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Progress directory: {PROGRESS_DIR}")
    print("=" * 60)
    
    # Load progress from previous runs
    progress = load_progress()
    progress = reconstruct_objects(progress)
    
    # Initialize Azure AI Inference client
    client = inference.Client()
    
    # If already completed, just export the results again
    if progress["completed"]:
        print("Flashcard generation was already completed in a previous run.")
        
        # Analyze coverage if we have concept nodes
        if "concept_nodes" in progress and progress["concept_nodes"]:
            coverage_report = analyze_coverage(progress["concept_nodes"])
        else:
            coverage_report = {"completed": True}
        
        # Export the flashcards
        export_flashcards(progress["flashcards"], coverage_report, OUTPUT_DIR)
        print("Regenerated output files.")
        os.environ["FLASHCARDS_COMPLETED"] = "true"
        return
    
    # Extract text from PDF if we're just starting
    chunks = []
    if progress["chunks_processed"] == 0 and progress["total_chunks"] == 0:
        print("Starting new flashcard generation process")
        chunks = extract_text_from_pdf(args.pdf_path)
        progress["total_chunks"] = len(chunks)
        save_progress(progress)
    else:
        # We need the chunks for reference, even if we've processed some already
        chunks = extract_text_from_pdf(args.pdf_path)
        print(f"Continuing from previous run: Processed {progress['chunks_processed']} of {progress['total_chunks']} chunks")
    
    # Track API requests for this run
    api_requests_made = 0
    
    # Process chunks that haven't been analyzed yet
    start_idx = progress["chunks_processed"]
    
    for i in range(start_idx, len(chunks)):
        if api_requests_made >= args.max_requests:
            print(f"Reached maximum API requests limit ({args.max_requests}) for this run.")
            break
        
        print(f"\nProcessing chunk {i+1} of {len(chunks)} (pages {', '.join(str(p) for p in chunks[i].source_pages)})")
        
        # Step 1: Analyze the chunk to identify concepts and their relationships
        print(f"Analyzing chunk with concept hierarchy...")
        analyzed_chunk = analyze_chunk_with_concept_hierarchy(chunks[i], client, model_name)
        api_requests_made += 1
        
        # If we've reached the request limit after analysis, save progress and exit
        if api_requests_made >= args.max_requests:
            progress["analyzed_chunks"].append(analyzed_chunk)
            progress["chunks_processed"] = i + 1
            save_progress(progress)
            print(f"Reached maximum API requests after analyzing chunk {i+1}. Saving progress and exiting.")
            os.environ["FLASHCARDS_COMPLETED"] = "false"
            return
        
        # Add the analyzed chunk to our progress
        progress["analyzed_chunks"].append(analyzed_chunk)
        
        # Step 2: Build the concept forest with what we have so far
        if i == 0 or i % 5 == 0 or i == len(chunks) - 1:  # Do this periodically to save processing time
            print("Building concept forest...")
            concept_nodes, root_concepts = build_concept_forest(progress["analyzed_chunks"])
            progress["concept_nodes"] = concept_nodes
        
        # Step 3: Generate flashcards for this chunk
        print(f"Generating flashcards for chunk {i+1}...")
        flashcards = generate_comprehensive_flashcards(
            chunk=analyzed_chunk,
            concept_nodes=progress["concept_nodes"],
            client=client,
            model_name=model_name
        )
        api_requests_made += 1
        
        if flashcards:
            progress["flashcards"].extend(flashcards)
            print(f"Generated {len(flashcards)} flashcards from chunk {i+1}")
        else:
            print(f"No flashcards generated from chunk {i+1}")
        
        # Update and save our progress
        progress["chunks_processed"] = i + 1
        save_progress(progress)
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    # Check if we've processed all chunks
    all_chunks_processed = progress["chunks_processed"] >= progress["total_chunks"]
    
    # If we've processed all chunks, create supplementary flashcards for any missing concepts
    if all_chunks_processed and api_requests_made < args.max_requests:
        print("\nAll chunks processed. Creating supplementary flashcards for any missing concepts...")
        
        # Ensure we have a complete concept forest
        concept_nodes, root_concepts = build_concept_forest(progress["analyzed_chunks"])
        progress["concept_nodes"] = concept_nodes
        
        # Generate flashcards for missing concepts
        supplementary_cards = create_missing_component_flashcards(
            concept_nodes=progress["concept_nodes"],
            client=client,
            model_name=model_name
        )
        api_requests_made += len(supplementary_cards) // 5 + 1  # Approximate API requests
        
        if supplementary_cards:
            progress["flashcards"].extend(supplementary_cards)
            print(f"Added {len(supplementary_cards)} supplementary flashcards for comprehensive coverage")
        
        # Mark as completed
        progress["completed"] = True
        os.environ["FLASHCARDS_COMPLETED"] = "true"
    else:
        os.environ["FLASHCARDS_COMPLETED"] = "false"
    
    # Analyze coverage
    coverage_report = analyze_coverage(progress["concept_nodes"])
    
    # Export the flashcards
    print(f"\nExporting {len(progress['flashcards'])} flashcards...")
    export_flashcards(progress["flashcards"], coverage_report, OUTPUT_DIR)
    
    # Save final progress
    save_progress(progress)
    
    # Final status
    if progress["completed"]:
        print("\n==== FLASHCARD GENERATION COMPLETE ====")
        print(f"Total chunks processed: {progress['chunks_processed']}/{progress['total_chunks']}")
        print(f"Total flashcards generated: {len(progress['flashcards'])}")
        print(f"Concept coverage: {coverage_report['coverage_percentage']}% overall, {coverage_report['core_percentage']}% of core concepts")
        print("Flashcards have been exported to multiple formats in the output directory")
        print("=" * 40)
    else:
        print("\n==== FLASHCARD GENERATION IN PROGRESS ====")
        print(f"Processed {progress['chunks_processed']}/{progress['total_chunks']} chunks in this run")
        print(f"Made {api_requests_made} API requests out of {args.max_requests} maximum")
        print(f"Total flashcards generated so far: {len(progress['flashcards'])}")
        print("Run the script again to continue processing")
        print("=" * 44)

if __name__ == "__main__":
    main()