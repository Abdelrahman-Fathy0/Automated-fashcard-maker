# GitHub Models Flashcard Generator

This tool generates flashcards from PDF textbooks using GitHub Models (via Azure AI Inference SDK).

## Usage

```bash
python github_flashcard_generator.py pdf_to_flashcards/test_pdfs/CS-TEXTBOOK.pdf [options]
Options:
--max-chunks N: Limit processing to N chunks (for testing)
--cards-per-chunk N: Number of flashcards to generate per chunk (default: 5)
--model MODEL_ID: Model ID to use (default: meta/llama-3-8b-instruct)
Available Models:
meta/llama-3-8b-instruct (default)
meta/llama-3-70b-instruct (higher quality)
mistralai/mistral-small
mistralai/mistral-medium
mistralai/mistral-large
Outputs
The tool creates an output directory with:

JSON, CSV, and Anki-compatible flashcard files
Run information
How It Works
Extracts text from the PDF in manageable chunks
Connects to GitHub Models via the Azure AI Inference SDK
Generates flashcards for each text chunk
Combines and exports all flashcards
Requirements
azure-ai-inference (installed automatically)
PyMuPDF (pip install PyMuPDF)
tqdm (pip install tqdm) EOL
echo "Created a working flashcard generator using the Azure AI Inference SDK" echo "Run it with: python github_flashcard_generator.py pdf_to_flashcards/test_pdfs/CS-TEXTBOOK.pdf --max-chunks 5"




