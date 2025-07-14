#!/usr/bin/env python3
"""
Simple test script to verify Ollama is running and accessible.
"""

import requests
import json

def test_ollama_connection(url="http://localhost:11434"):
    """Test if Ollama is running and accessible."""
    try:
        # Test basic connectivity
        response = requests.get(f"{url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ Ollama is running!")
            print(f"Available models: {[model['name'] for model in models['models']]}")
            return True
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama. Make sure it's running on http://localhost:11434")
        print("   Start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error testing Ollama: {e}")
        return False

def test_ollama_query(url="http://localhost:11434", model="llama3.2"):
    """Test a simple query to Ollama."""
    try:
        prompt = "Say hello and confirm you are working properly."
        
        response = requests.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()["response"]
            print("✅ Ollama query test successful!")
            print(f"Response: {result}")
            return True
        else:
            print(f"❌ Query failed with status {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error querying Ollama: {e}")
        return False

def main():
    print("Testing Ollama connection...")
    
    if test_ollama_connection():
        print("\nTesting Ollama query...")
        test_ollama_query()
    else:
        print("\nPlease ensure Ollama is running:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Start the service: ollama serve")
        print("3. Pull a model: ollama pull llama3.2")

if __name__ == "__main__":
    main()
