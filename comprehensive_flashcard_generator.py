#!/usr/bin/env python3
"""
Comprehensive Flashcard Generator for CS Textbooks with Complete Topic Coverage
- Ensures all related components/concepts are covered with flashcards
- Uses hierarchical concept mapping to track parent-child relationships
- Performs concept-coverage analysis to fill in missing components
"""

import os
import sys
import json
import fitz  # PyMuPDF
import time
import re
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# Azure AI Inference SDK imports
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Add this custom JSON encoder class
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ConceptNode):
            return obj.to_dict()
        return super().default(obj)

# Create output directory
OUTPUT_DIR = "comprehensive_flashcards"
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

# Progress tracking
PROGRESS_DIR = "progress"
MAX_REQUESTS_PER_RUN = 140  # Adjust based on your needs (safe limit for 150/day)
os.makedirs(PROGRESS_DIR, exist_ok=True)

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
    
    # Create system message
    system_message = SystemMessage(
        content="You are an expert computer science educator skilled at identifying important concepts and their relationships in textbook content."
    )
    
    # Create user message
    user_message = UserMessage(
        content=f"""Analyze this computer science textbook content from page(s) {page_sources} with HIGHLY DETAILED FOCUS ON COMPONENT RELATIONSHIPS:

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
}}

Only output valid JSON that can be parsed with json.loads().
"""
    )
    
    try:
        # Call the client to analyze the content
        response = client.complete(
            messages=[system_message, user_message],
            model=model_name,
            temperature=0.2,  # Lower temperature for more consistent analysis
            max_tokens=2500
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
    
    # Create system message
    system_message = SystemMessage(
        content="""You are an expert computer science educator who creates high-quality flashcards from textbook content.
Focus on creating flashcards that test understanding of core concepts and ensure COMPREHENSIVE coverage of ALL related components.
"""
    )
    
    # Create user message
    user_message = UserMessage(
        content=f"""Text from CS textbook (from page {page_sources}):

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
            max_tokens=2500
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
        
        # Create system message
        system_message = SystemMessage(
            content="You are an expert computer science educator creating flashcards for important concepts."
        )
        
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
        
        # Create user message
        user_message = UserMessage(
            content=f"""Create flashcards for these important computer science concepts that need coverage:

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
]

Only output valid JSON that can be parsed with json.loads().
"""
        )
        
        try:
            # Call the client to generate supplementary flashcards
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
        "analyzed_chunks_processed": 0,
        "total_chunks": 0,
        "analyzed_chunks": [],
        "flashcards": [],
        "concept_nodes_dict": {},
        "completed": False,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "pdf_file": None,
        "output_dir": None,
        "model_name": None
    }

def save_progress(progress_data):
    """Save progress data for future runs."""
    # Update the last_updated timestamp
    progress_data["last_updated"] = datetime.now().isoformat()
    
    # Save to file
    progress_file = os.path.join(PROGRESS_DIR, "progress.json")
    with open(progress_file, "w", encoding="utf-8") as f:
        # Convert objects to serializable format
        serializable_progress = progress_data.copy()
        
        # Convert analyzed_chunks to serializable format
        if "analyzed_chunks" in serializable_progress:
            serializable_chunks = []
            for chunk in serializable_progress["analyzed_chunks"]:
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
            serializable_progress["analyzed_chunks"] = serializable_chunks
        
        # Convert flashcards to serializable format
        if "flashcards" in serializable_progress:
            serializable_flashcards = []
            for card in serializable_progress["flashcards"]:
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
            serializable_progress["flashcards"] = serializable_flashcards
        
        json.dump(serializable_progress, f, cls=CustomJSONEncoder, indent=2)

def reconstruct_objects(progress_data):
    """Reconstruct objects from serialized data."""
    # Reconstruct TextChunk objects
    if "analyzed_chunks" in progress_data:
        reconstructed_chunks = []
        for chunk_data in progress_data["analyzed_chunks"]:
            if isinstance(chunk_data, dict):
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
            else:
                # It's already a TextChunk
                reconstructed_chunks.append(chunk_data)
        progress_data["analyzed_chunks"] = reconstructed_chunks
    
    # Reconstruct Flashcard objects
    if "flashcards" in progress_data:
        reconstructed_cards = []
        for card_data in progress_data["flashcards"]:
            if isinstance(card_data, dict):
                card = Flashcard(
                    question=card_data["question"],
                    answer=card_data["answer"],
                    source_pages=card_data.get("source_pages", []),
                    importance=card_data.get("importance", 5),
                    tags=card_data.get("tags", []),
                    concept=card_data.get("concept", ""),
                    concept_path=card_data.get("concept_path", [])
                )
                reconstructed_cards.append(card)
            else:
                # It's already a Flashcard
                reconstructed_cards.append(card_data)
        progress_data["flashcards"] = reconstructed_cards
    
    # Reconstruct concept nodes
    if "concept_nodes_dict" in progress_data and progress_data["concept_nodes_dict"]:
        concept_nodes = {}
        
        # First pass: create all nodes
        for name, node_data in progress_data["concept_nodes_dict"].items():
            concept_nodes[name] = ConceptNode(
                name=name,
                importance=node_data.get("importance", 5),
                concept_type=node_data.get("type", "CORE")
            )
            concept_nodes[name].covered = node_data.get("covered", False)
            concept_nodes[name].source_pages = node_data.get("source_pages", [])
        
        # Second pass: rebuild relationships
        for name, node_data in progress_data["concept_nodes_dict"].items():
            if node_data.get("parent") and node_data["parent"] in concept_nodes:
                concept_nodes[name].parent = concept_nodes[node_data["parent"]]
            
            for child_name in node_data.get("children", []):
                if child_name in concept_nodes and child_name != name:
                    concept_nodes[name].add_child(concept_nodes[child_name])
        
        progress_data["concept_nodes"] = concept_nodes
    
    return progress_data

def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_flashcard_generator.py <pdf_file> [--max-chunks N] [--model MODEL_NAME] [--max-requests N]")
        print("\nAvailable models:")
        for alias, model in AVAILABLE_MODELS.items():
            print(f"  {alias}: {model}")
        sys.exit(1)
    
    # Parse command-line arguments
    pdf_file = sys.argv[1]
    max_chunks = None
    model_name = AVAILABLE_MODELS.get("gpt4o-mini")  # Default to GPT-4o mini
    max_requests = MAX_REQUESTS_PER_RUN
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--max-chunks" and i + 1 < len(sys.argv):
            max_chunks = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            model_input = sys.argv[i + 1]
            if model_input in AVAILABLE_MODELS:
                model_name = AVAILABLE_MODELS[model_input]
            else:
                model_name = model_input
            i += 2
        elif sys.argv[i] == "--max-requests" and i + 1 < len(sys.argv):
            max_requests = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    # Load progress from previous runs
    progress = load_progress()
    progress = reconstruct_objects(progress)
    
    # Initialize a flag to track completion status
    os.environ["FLASHCARDS_COMPLETED"] = "false"
    
    # Check if we're starting a new run or continuing an existing one
    if progress["completed"]:
        print("Flashcard generation was already completed in a previous run.")
        # Export the flashcards again if needed
        if "concept_nodes" in progress and progress["concept_nodes"]:
            coverage_report = analyze_coverage(progress["concept_nodes"])
        else:
            coverage_report = {"completed": True}
        
        if progress["output_dir"]:
            export_flashcards(progress["flashcards"], coverage_report, progress["output_dir"])
            print("Regenerated output files.")
        os.environ["FLASHCARDS_COMPLETED"] = "true"
        return
    
    # Set up the output directory
    if not progress["output_dir"]:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        progress["output_dir"] = output_dir
        progress["pdf_file"] = pdf_file
        progress["model_name"] = model_name
        
        # Save initial run information
        with open(os.path.join(output_dir, "run_info.txt"), "w") as f:
            f.write(f"PDF File: {pdf_file}\n")
            f.write(f"Model: {model_name}\n")
            if max_chunks:
                f.write(f"Max chunks: {max_chunks}\n")
            f.write(f"Started at: {timestamp}\n")
            f.write(f"Running incrementally with max {max_requests} API requests per run\n")
    else:
        output_dir = progress["output_dir"]
        # Use saved values if continuing a run
        if progress["pdf_file"]:
            pdf_file = progress["pdf_file"]
        if progress["model_name"]:
            model_name = progress["model_name"]
    
    # Track API requests for this run
    api_requests_made = 0
    
    print(f"\n===== INCREMENTAL COMPREHENSIVE CS FLASHCARD GENERATION =====")
    print(f"PDF File: {pdf_file}")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Maximum API requests per run: {max_requests}")
    
    # Set up GitHub Models client
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
    
    # Extract text from PDF if we're just starting
    chunks = []
    if progress["chunks_processed"] == 0 and progress["total_chunks"] == 0:
        print("Starting new flashcard generation process")
        chunks = extract_text_from_pdf(pdf_file)
        if max_chunks:
            print(f"Limiting to {max_chunks} chunks")
            chunks = chunks[:max_chunks]
        progress["total_chunks"] = len(chunks)
        save_progress(progress)
    else:
        # We need chunks for reference
        chunks = extract_text_from_pdf(pdf_file)
        if max_chunks:
            chunks = chunks[:max_chunks]
        print(f"Continuing from previous run: Processed {progress['chunks_processed']} of {progress['total_chunks']} chunks")
    
    # STEP 1: Process chunks that haven't been analyzed yet
    analyze_start_idx = len(progress["analyzed_chunks"])
    
    if analyze_start_idx < len(chunks):
        print(f"Analyzing content with hierarchical concept mapping using model: {model_name}")
        
        for i in range(analyze_start_idx, len(chunks)):
            if api_requests_made >= max_requests:
                print(f"Reached maximum API requests limit ({max_requests}) for analysis.")
                break
            
            print(f"DEBUG: Starting processing of chunk {i}")
            print(f"Analyzing chunk {i+1}/{len(chunks)} (pages {', '.join(str(p) for p in chunks[i].source_pages)})")
            analyzed_chunk = analyze_chunk_with_concept_hierarchy(chunks[i], client, model_name)
            progress["analyzed_chunks"].append(analyzed_chunk)
            api_requests_made += 1
            
            # Save progress after each chunk
            save_progress(progress)
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            print(f"DEBUG: Completed processing of chunk {i}")
    
    # STEP 2: Build the concept forest if we have analyzed chunks
    if progress["analyzed_chunks"]:
        if "concept_nodes" not in progress or not progress["concept_nodes"]:
            print("Building concept forest...")
            concept_nodes, root_concepts = build_concept_forest(progress["analyzed_chunks"])
            progress["concept_nodes"] = concept_nodes
            
            # Export concept hierarchy for reference
            concept_hierarchy_file = os.path.join(output_dir, "concept_hierarchy.json")
            with open(concept_hierarchy_file, "w", encoding="utf-8") as f:
                hierarchy_data = {
                    "concepts": {name: node.to_dict() for name, node in concept_nodes.items()},
                    "roots": [root.name for root in root_concepts]
                }
                json.dump(hierarchy_data, f, indent=2)
            
            # Save progress after building concept forest
            save_progress(progress)
        else:
            concept_nodes = progress["concept_nodes"]
    else:
        concept_nodes = {}
    
    # STEP 3: Generate flashcards for analyzed chunks
    if progress["analyzed_chunks"]:
        print(f"Generating comprehensive flashcards using model: {model_name}")
        
        for i in range(progress["chunks_processed"], len(progress["analyzed_chunks"])):
            if api_requests_made >= max_requests:
                print(f"Reached maximum API requests limit ({max_requests}) for flashcard generation.")
                break
            
            chunk = progress["analyzed_chunks"][i]
            print(f"Generating flashcards for chunk {chunk.id + 1}/{len(chunks)} (pages {', '.join(str(p) for p in chunk.source_pages)})")
            
            flashcards = generate_comprehensive_flashcards(
                chunk=chunk,
                concept_nodes=concept_nodes,
                client=client,
                model_name=model_name
            )
            api_requests_made += 1
            
            if flashcards:
                progress["flashcards"].extend(flashcards)
                print(f"Generated {len(flashcards)} flashcards for chunk {chunk.id + 1}")
            else:
                print(f"No flashcards generated for chunk {chunk.id + 1}")
            
            # Update and save progress
            progress["chunks_processed"] = i + 1
            save_progress(progress)
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
    
    # Check if we've processed all chunks
    all_chunks_processed = progress["chunks_processed"] >= progress["total_chunks"] and len(progress["analyzed_chunks"]) >= progress["total_chunks"]
    
    # STEP 4: Generate supplementary flashcards for missing concepts if we've processed all chunks
    if all_chunks_processed and api_requests_made < max_requests and concept_nodes:
        print("All chunks processed. Creating supplementary flashcards for missing concepts...")
        
        supplementary_flashcards = create_missing_component_flashcards(
            concept_nodes=concept_nodes,
            client=client,
            model_name=model_name
        )
        api_requests_made += 1  # Approximation
        
        if supplementary_flashcards:
            progress["flashcards"].extend(supplementary_flashcards)
            print(f"Added {len(supplementary_flashcards)} supplementary flashcards for missing concepts")
            save_progress(progress)
    
    # STEP 5: Complete if all chunks are processed and supplementary flashcards are done
    if all_chunks_processed:
        # Analyze coverage
        coverage_report = analyze_coverage(concept_nodes)
        
        # Export the flashcards
        print(f"Exporting {len(progress['flashcards'])} total flashcards")
        exported_files = export_flashcards(progress['flashcards'], coverage_report, output_dir)
        
        # Mark as completed
        progress["completed"] = True
        
        # Save completion information
        completion_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(os.path.join(output_dir, "run_info.txt"), "a") as f:
            f.write(f"Completed at: {completion_time}\n")
            f.write(f"Total chunks processed: {progress['total_chunks']}\n")
            f.write(f"Total concepts identified: {len(concept_nodes)}\n")
            f.write(f"Core concepts: {coverage_report['core_total']}\n")
            f.write(f"Core concept coverage: {coverage_report['core_percentage']}%\n")
            f.write(f"Total flashcards generated: {len(progress['flashcards'])}\n")
            for format_name, file_path in exported_files.items():
                f.write(f"Exported to {format_name}: {os.path.basename(file_path)}\n")
        
        print(f"\n===== PROCESSING COMPLETE =====")
        print(f"Total chunks processed: {progress['total_chunks']}")
        print(f"Total concepts identified: {len(concept_nodes)}")
        print(f"Total flashcards generated: {len(progress['flashcards'])}")
        
        print("\nExported formats:")
        for format_name, file_path in exported_files.items():
            print(f"- {format_name.upper()}: {os.path.basename(file_path)}")
        
        print("\nConcept Coverage:")
        print(f"- Overall: {coverage_report['coverage_percentage']}% of all concepts")
        print(f"- Core Concepts: {coverage_report['core_percentage']}% of core concepts")
        print(f"- High Importance: {coverage_report['high_importance_percentage']}% of high-importance concepts")
        
        os.environ["FLASHCARDS_COMPLETED"] = "true"
    else:
        # Progress report for incomplete runs
        print(f"\n===== PROGRESS UPDATE =====")
        print(f"Processed {len(progress['analyzed_chunks'])}/{progress['total_chunks']} chunks (analysis)")
        print(f"Processed {progress['chunks_processed']}/{progress['total_chunks']} chunks (flashcard generation)")
        print(f"Generated {len(progress['flashcards'])} flashcards so far")
        print(f"Made {api_requests_made} API requests in this run")
        print("Run the script again to continue processing")
    
    # Save final progress state
    save_progress(progress)
    
    print("\nNotes:")
    print(f"- Used model: {model_name}")
    print("- Progress saved and can be resumed in next run")
    print("=" * 45)
    print("Status update message")
    sys.stdout.flush()  # Force output to be displayed immediately

if __name__ == "__main__":
    main()
