#!/usr/bin/env python3
"""Test persona generation for a single user"""

import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from src.persona_generator import PersonaGenerator

load_dotenv()

def test_persona():
    # Load config
    with open('config/settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)
    
    # Find a user with comments
    output_dir = Path("data/output")
    comment_files = list(output_dir.glob("*.md"))
    comment_files = [f for f in comment_files if not f.name.endswith('_persona.md')]
    
    if not comment_files:
        print("✗ No comment files found. Run the comment fetcher first.")
        return
    
    # Use first user for testing
    test_file = comment_files[0]
    username = test_file.stem
    
    print(f"Testing persona generation for u/{username}...")
    print(f"Using comment file: {test_file}")
    
    try:
        persona_generator = PersonaGenerator(settings)
        persona = persona_generator.generate_persona(test_file)
        
        print("✓ Persona generated!")
        print(f"Length: {len(persona)} characters")
        print("\nFirst 500 characters:")
        print(persona[:500] + "..." if len(persona) > 500 else persona)
        
        # Save test persona
        test_output = output_dir / f"{username}_persona_test.md"
        with open(test_output, 'w', encoding='utf-8') as f:
            f.write(persona)
        
        print(f"\n✓ Test persona saved to: {test_output}")
        
    except Exception as e:
        print(f"✗ Error generating persona: {e}")

if __name__ == "__main__":
    test_persona()