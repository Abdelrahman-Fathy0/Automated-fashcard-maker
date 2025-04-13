#!/usr/bin/env python3
"""
Download required NLTK data packages.
"""

import nltk
import os
import sys
from tqdm import tqdm

def download_nltk_data():
    """Download required NLTK data packages."""
    resources = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4'
    ]
    
    print("Downloading NLTK resources...")
    
    for resource in tqdm(resources, desc="Downloading NLTK resources"):
        try:
            nltk.download(resource, quiet=True)
            print(f"✓ Successfully downloaded {resource}")
        except Exception as e:
            print(f"✗ Error downloading {resource}: {e}")
    
    # Set environment variable to use downloaded data
    os.environ['NLTK_DATA'] = os.path.expanduser('~/nltk_data')
    
    print("\nNLTK data download complete!")

if __name__ == "__main__":
    download_nltk_data()
