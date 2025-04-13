#!/usr/bin/env python3
"""
Examine GitHub Models samples and create a working flashcard generator.
"""

import os
import sys
import glob
import json
from datetime import datetime

# Create output directory
OUTPUT_DIR = "flashcards_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def examine_sample(sample_path):
    """Examine a sample Python file and extract key information."""
    print(f"Examining sample: {sample_path}")
    
    with open(sample_path, 'r') as f:
        content = f.read()
    
    # Look for API endpoints
    endpoints = []
    for line in content.split('\n'):
        if 'http' in line and ('api' in line.lower() or 'endpoint' in line.lower()):
            endpoints.append(line.strip())
    
    # Look for import statements to understand dependencies
    imports = []
    for line in content.split('\n'):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            imports.append(line.strip())
    
    # Look for authentication methods
    auth_methods = []
    auth_keywords = ['key', 'token', 'auth', 'credential', 'secret']
    for line in content.split('\n'):
        if any(keyword in line.lower() for keyword in auth_keywords):
            auth_methods.append(line.strip())
    
    return {
        'endpoints': endpoints,
        'imports': imports,
        'auth_methods': auth_methods
    }

def find_samples():
    """Find all Python sample files."""
    samples = []
    
    if os.path.exists('samples/python'):
        # Check if sample directories exist
        sample_dirs = [d for d in os.listdir('samples/python') 
                      if os.path.isdir(os.path.join('samples/python', d))]
        
        for d in sample_dirs:
            dir_path = os.path.join('samples/python', d)
            py_files = glob.glob(f"{dir_path}/*.py")
            samples.extend(py_files)
    
    if os.path.exists('cookbooks/python'):
        # Check cookbook files
        cookbook_files = glob.glob('cookbooks/python/*.py')
        samples.extend(cookbook_files)
    
    return samples

def display_sample_info(sample_info, sample_path):
    """Display information extracted from a sample file."""
    print(f"\n=== Sample: {sample_path} ===")
    
    print("\nImports:")
    for imp in sample_info['imports']:
        print(f"  {imp}")
    
    print("\nEndpoints:")
    for endpoint in sample_info['endpoints']:
        print(f"  {endpoint}")
    
    print("\nAuthentication Methods:")
    for auth in sample_info['auth_methods']:
        print(f"  {auth}")
    
    print("\n" + "-"*50)

def extract_sample_code(sample_path):
    """Extract a self-contained example from a sample file."""
    with open(sample_path, 'r') as f:
        content = f.read()
    
    # Look for complete examples (often in main or __main__ blocks)
    if '__main__' in content:
        return content
    
    # Otherwise return the whole file
    return content

def create_working_example():
    """Create a working example based on the samples."""
    samples = find_samples()
    
    if not samples:
        print("No samples found. Please check the samples directory structure.")
        return
    
    print(f"Found {len(samples)} sample files.")
    
    # Analyze each sample
    all_sample_info = {}
    for sample in samples:
        try:
            sample_info = examine_sample(sample)
            all_sample_info[sample] = sample_info
            display_sample_info(sample_info, sample)
        except Exception as e:
            print(f"Error examining {sample}: {e}")
    
    # Find the most promising sample for flashcard generation
    best_sample = None
    for sample, info in all_sample_info.items():
        # Look for samples that mention completion, chat, or generation
        sample_content = extract_sample_code(sample)
        if ('completion' in sample_content.lower() or 
            'chat' in sample_content.lower() or 
            'generate' in sample_content.lower()):
            best_sample = sample
            break
    
    if not best_sample:
        best_sample = next(iter(all_sample_info.keys()))
    
    print(f"\nUsing {best_sample} as a base for our flashcard generator.")
    
    # Create a modified version for flashcards
    sample_code = extract_sample_code(best_sample)
    
    # Write the sample code to a file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sample_file = f"{OUTPUT_DIR}/sample_code_{timestamp}.py"
    with open(sample_file, 'w') as f:
        f.write(sample_code)
    
    print(f"Sample code saved to {sample_file}")
    print("\nTo create flashcards:")
    print(f"1. Review the sample code in {sample_file}")
    print("2. Update it to generate flashcards instead of its original purpose")
    print("3. Run it to test GitHub Models access")

def main():
    print("GitHub Models Sample Explorer")
    print("==========================\n")
    
    create_working_example()
    
    # Create adaptable flashcard generator using the samples as reference
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    adaptor_file = f"{OUTPUT_DIR}/flashcard_generator_{timestamp}.py"
    
    with open(adaptor_file, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Flashcard Generator using GitHub Models.
Based on the sample code provided in the GitHub Models Codespace.
"""

import os
import sys
import json
from datetime import datetime

# IMPORTANT: Update these imports to match what's in the sample code
# Example: 
# from azure.ai.inference import InferenceClient
# from azure.core.credentials import KeyCredential

# Replace with your textbook text
def get_textbook_content():
    # This is where you would load your textbook content
    return [
        "Computer science is the study of computation, automation, and information.",
        "Algorithms are step-by-step procedures for solving problems.",
        "Data structures are specialized formats for organizing and storing data.",
        "Operating systems manage computer hardware and software resources.",
        "Machine learning is a field of AI that enables systems to learn from data."
    ]

def generate_flashcards():
    """
    Generate flashcards using the GitHub Models API.
    Update this function based on the sample code.
    """
    # REPLACE THIS CODE with the appropriate implementation from the samples
    
    # Example structure (needs to be adapted):
    # client = InferenceClient(endpoint="...", credential=KeyCredential("..."))
    # response = client.complete(
    #     model="...",
    #     prompt="Create 5 flashcards about this topic: ...",
    #     temperature=0.7
    # )
    
    # For now, we'll just create placeholder flashcards
    return [
        {"question": "What is computer science?", "answer": "The study of computation, automation, and information."},
        {"question": "What are algorithms?", "answer": "Step-by-step procedures for solving problems."}
    ]

def save_flashcards(flashcards):
    """Save flashcards to file."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = f"flashcards_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(flashcards, f, indent=2)
    
    print(f"Saved {len(flashcards)} flashcards to {output_file}")

def main():
    print("Flashcard Generator using GitHub Models")
    print("=====================================\n")
    
    print("Getting textbook content...")
    content = get_textbook_content()
    
    print(f"Generating flashcards for {len(content)} text chunks...")
    flashcards = generate_flashcards()
    
    print(f"Generated {len(flashcards)} flashcards")
    save_flashcards(flashcards)
    
    print("\nTo create a complete solution:")
    print("1. Update this script with the proper API calls from the samples")
    print("2. Integrate your PDF extraction code")
    print("3. Add flashcard parsing and formatting")

if __name__ == "__main__":
    main()
''')
    
    print(f"\nCreated adaptable flashcard generator: {adaptor_file}")
    print("Modify this file based on the sample code to create a working solution.")

if __name__ == "__main__":
    main()
