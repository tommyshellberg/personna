#!/usr/bin/env python3
"""Test Ollama embedding model.

Usage:
    python scripts/test_embedding.py
"""

import ollama


def main():
    print("Testing nomic-embed-text embedding model...")

    try:
        response = ollama.embed(
            model="nomic-embed-text",
            input="This is a test sentence for embedding."
        )

        embedding = response["embeddings"][0]
        print(f"  Success!")
        print(f"  Vector dimensions: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")

    except Exception as e:
        print(f"  Failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
