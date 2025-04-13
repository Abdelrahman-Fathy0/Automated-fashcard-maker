#!/usr/bin/env python3
"""
Download a test PDF for flashcard generation.
"""

import os
import requests
from tqdm import tqdm

def download_file(url, output_path):
    """Download a file with progress bar."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(output_path, 'wb') as file, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)


def main():
    """Download a test PDF."""
    # Example computer science textbook (public domain)
    pdf_url = "https://open.umn.edu/rails/active_storage/disk/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdDRm9VZEhKcFkyVXViM0puQ2daRVpYWmxiZz09IiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2tleSJ9fQ==--7be0d93301279773c15c1ec0b7b6eb6fc62c27f5/ComputerNetworks.pdf"
    output_path = "pdf_to_flashcards/test_pdfs/ComputerNetworks.pdf"
    
    print(f"Downloading test PDF to {output_path}...")
    download_file(pdf_url, output_path)
    print(f"PDF downloaded successfully!")
    print(f"You can now generate flashcards with:")
    print(f"python flashcard_generator.py {output_path} --max-chunks 5")


if __name__ == "__main__":
    main()
