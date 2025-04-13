#!/usr/bin/env python3
"""
Test script to find working model names for Azure AI Inference.
"""

import os
import sys
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

def test_model(client, model_name):
    """Test if a model name works with the client."""
    try:
        print(f"Testing model: {model_name}")
        response = client.complete(
            model=model_name,
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="Say hello and tell me what model you are.")
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        # Print the response
        content = response.choices[0].message.content
        print(f"\n=== SUCCESS! Model {model_name} works ===")
        print(f"Response: {content}\n")
        return True
    except Exception as e:
        print(f"Error with model '{model_name}': {e}")
        return False

def main():
    # Set up the client
    endpoint = "https://models.inference.ai.azure.com"
    token = os.environ["GITHUB_TOKEN"]
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token)
    )
    
    # Test different model name formats
    model_options = [
        # Basic formats
        "llama-3-8b-instruct",
        "mistral-small",
        "claude-instant-1.2",
        
        # With vendor prefixes
        "meta/llama-3-8b-instruct",
        "mistralai/mistral-small",
        "anthropic/claude-instant-1.2",
        
        # Alternative formats
        "llama3-8b-instruct",
        "llama-3-8b",
        "llama3-8b",
        
        # GPT formats
        "gpt-35-turbo",
        "gpt-4",
        "openai/gpt-3.5-turbo",
        
        # Check for sample names
        "gpt-4-32k",
        "llama-2-13b",
        "llama-2-70b",
    ]
    
    # Track working models
    working_models = []
    
    for model in model_options:
        if test_model(client, model):
            working_models.append(model)
    
    # Summary
    print("\n=== SUMMARY ===")
    if working_models:
        print("Working models:")
        for model in working_models:
            print(f"- {model}")
        
        print("\nTo generate flashcards, use:")
        print(f"python fixed_github_flashcard.py pdf_to_flashcards/test_pdfs/CS-TEXTBOOK.pdf --model {working_models[0]} --max-chunks 3")
    else:
        print("No working models found.")
        print("Try examining the sample files more carefully to identify the exact model names.")

if __name__ == "__main__":
    main()
