#!/usr/bin/env python3
"""
Simple Flashcard Generator using GitHub Models samples directly.
"""

import os
import json
import requests
import sys
import re
from tqdm import tqdm
from datetime import datetime

# Directory for output
OUTPUT_DIR = "flashcards_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_text_chunks(pdf_path, max_chunks=None):
    """
    Load pre-extracted text chunks from your PDF.
    For simplicity, we'll use the first page as a sample.
    """
    # This is a placeholder - in a real implementation, 
    # we'd use your existing PDF extraction code
    chunks = [
        "This is a sample text chunk from the CS textbook.",
        "Computer networks allow devices to communicate and share resources.",
        "Algorithms are step-by-step procedures for solving computational problems.",
        "Data structures organize data to enable efficient access and modification.",
        "Object-oriented programming is a paradigm based on objects and classes.",
        "Operating systems manage computer hardware and software resources.",
        "Recursion is a method where the solution depends on solutions to smaller instances.",
        "Machine learning algorithms can learn from and make predictions on data.",
        "Cryptography is used to secure communication in the presence of adversaries.",
        "Databases store and organize data for easy access, management and updates."
    ]
    
    # If max_chunks is specified, limit the number of chunks
    if max_chunks:
        chunks = chunks[:max_chunks]
        
    return chunks

def call_github_model(text_chunk, model_id="meta/llama-3-8b-instruct"):
    """
    Call GitHub Models using the samples approach.
    """
    print(f"Calling GitHub Models API with model: {model_id}")
    
    # Instructions for generating flashcards
    system_prompt = """You are an expert educator creating high-quality flashcards from textbook content.
Generate 5 flashcards in question-answer format covering the most important concepts."""

    user_prompt = f"""Here is a section from a computer science textbook:

{text_chunk}

Create 5 high-quality flashcards in this exact format:
Q: [clear, specific question]
A: [concise, complete answer]

Focus on the most important concepts, make questions clear, and ensure answers are accurate and concise."""

    # Format messages based on GitHub Models sample code
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Check different possible API endpoints based on GitHub Codespace samples
    possible_endpoints = [
        "https://api.github.com/github-models/chat/completions",
        "https://api.github.com/models/chat/completions",
        "https://api.githubcopilot.com/chat/completions"
    ]
    
    headers = {
        "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    data = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # Try each endpoint
    for endpoint in possible_endpoints:
        try:
            print(f"Trying endpoint: {endpoint}")
            response = requests.post(
                endpoint,
                headers=headers,
                json=data,
                timeout=30
            )
            
            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error response: {response.text}")
        except Exception as e:
            print(f"Error with endpoint {endpoint}: {e}")
    
    # If all endpoints fail, return None
    return None

def parse_flashcards(response_text):
    """
    Parse flashcards from model response.
    """
    if not response_text:
        return []
    
    # Extract content from the response
    try:
        content = response_text["choices"][0]["message"]["content"]
    except:
        print("Unexpected response format:")
        print(response_text)
        return []
    
    # Split by Q: to get individual cards
    cards = []
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
        
        # Add to cards list
        cards.append({
            "question": question,
            "answer": answer
        })
    
    return cards

def execute_direct_api_call():
    """
    Execute a direct call to the GitHub Models API using cURL sample
    """
    try:
        import subprocess
        
        # Create a temporary file with the cURL command
        with open('temp_curl.sh', 'w') as f:
            f.write('''#!/bin/bash
curl https://api.github.com/models/meta/llama-3-8b-instruct/completions \\
  -X POST \\
  -H "Accept: application/vnd.github+json" \\
  -H "Authorization: Bearer $GITHUB_TOKEN" \\
  -H "X-GitHub-Api-Version: 2022-11-28" \\
  -d '{"prompt":"Create 5 computer science flashcards about algorithms","max_tokens":1000,"temperature":0.7}'
''')
        
        # Make it executable
        os.chmod('temp_curl.sh', 0o755)
        
        # Execute it
        result = subprocess.run(['./temp_curl.sh'], capture_output=True, text=True)
        
        print("cURL output:")
        print(result.stdout)
        
        # Clean up
        os.remove('temp_curl.sh')
        
        return result.stdout
    except Exception as e:
        print(f"Error with cURL approach: {e}")
        return None

def main():
    # Check for GitHub Model samples first
    if os.path.exists("samples"):
        print("Found GitHub Models samples directory!")
        sample_dirs = os.listdir("samples")
        print(f"Available sample directories: {', '.join(sample_dirs)}")
        
        if "python" in sample_dirs:
            print("\nPython samples found! Check these files to see how to properly call the API:")
            for file in os.listdir("samples/python"):
                print(f"- samples/python/{file}")
            
            print("\nYou should review these sample files to correctly access GitHub Models.")
            print("They will show the correct API endpoints and parameters to use.\n")
    
    print("\nAttempting multiple approaches to generate flashcards...")
    
    # Try direct cURL approach
    print("\n=== Trying direct cURL approach ===")
    curl_result = execute_direct_api_call()
    
    # Try our Python approach
    print("\n=== Trying Python approach with various endpoints ===")
    
    # Test with a single chunk
    test_chunk = "Computer science is the study of computation, automation, and information. Computer science spans theoretical disciplines to practical disciplines."
    
    # Try with different models
    models = [
        "meta/llama-3-8b-instruct",
        "mistralai/mistral-small",
        "anthropic/claude-instant-1.2",
        "openai/gpt-3.5-turbo"
    ]
    
    # Try each model
    successful_model = None
    for model in models:
        print(f"\nTrying model: {model}")
        response = call_github_model(test_chunk, model_id=model)
        
        if response:
            print("Success!")
            successful_model = model
            break
        else:
            print(f"Failed to use model: {model}")
    
    # Output final message
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"{OUTPUT_DIR}/results_{timestamp}.txt", "w") as f:
        f.write("GitHub Models API Testing Results\n")
        f.write("================================\n\n")
        
        if successful_model:
            f.write(f"Successfully accessed GitHub Models API using model: {successful_model}\n")
        else:
            f.write("Could not access any GitHub Models API endpoints.\n")
            f.write("To debug, check the samples directory and follow the examples there.\n")
            
        f.write("\nNext steps:\n")
        f.write("1. Check the 'samples' directory for working examples\n")
        f.write("2. Use the correct API endpoints and parameters from those samples\n")
        f.write("3. Try running the samples directly to verify GitHub Models access\n")
    
    print(f"\nResults saved to {OUTPUT_DIR}/results_{timestamp}.txt")
    print("\nNEXT STEPS:")
    print("1. Check the 'samples' directory for working examples")
    print("2. Use the correct API endpoints from those samples")
    print("3. Try running a sample code directly to verify GitHub Models access")

if __name__ == "__main__":
    main()
