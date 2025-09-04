#!/usr/bin/env python3
"""Simple Ollama test without extra dependencies"""

import subprocess
import json
import yaml
from pathlib import Path


def test_ollama():
    """Test if Ollama is working with configured model"""
    
    # Load model from config
    try:
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        with open(config_path, 'r') as f:
            settings = yaml.safe_load(f)
        model_name = settings['ollama']['model']
        base_url = settings['ollama']['base_url']
    except Exception as e:
        print(f"✗ Could not load config: {e}")
        print("Falling back to default: qwen3:8b")
        model_name = "qwen3:8b"
        base_url = "http://localhost:11434"
    try:
        # Test if service is running
        result = subprocess.run(
            ["curl", "-s", f"{base_url}/api/tags"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print("✗ Ollama not running. Start with: ollama serve")
            return False

        models = json.loads(result.stdout)
        available = [m['name'] for m in models['models']]
        print(f"✓ Ollama running. Models: {available}")

        # Test generation with configured model
        if model_name not in available:
            print(f"✗ {model_name} not found. Try: ollama pull {model_name}")
            return False

        print(f"✓ Using model: {model_name}")

        # Quick test generation
        payload = {
            "model": model_name,
            "prompt": "Say 'Hello' in one word:",
            "stream": False
        }

        result = subprocess.run(
            ["curl", "-s", "-X", "POST", f"{base_url}/api/generate",
             "-d", json.dumps(payload)],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            response = json.loads(result.stdout)
            print(f"✓ Generation works: '{response['response'].strip()}'")
            return True
        else:
            print(f"✗ Generation failed")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    test_ollama()
