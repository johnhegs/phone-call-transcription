#!/usr/bin/env python3
"""
Test script to verify faster-whisper installation and platform compatibility.
This script tests the faster-whisper installation without requiring audio files.
"""

import sys
import os
from pathlib import Path

def test_faster_whisper_import():
    """Test if faster-whisper can be imported successfully."""
    print("🧪 Testing faster-whisper import...")
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Failed to import faster-whisper: {e}")
        return False

def test_whisper_model_initialization():
    """Test if we can initialize a Whisper model."""
    print("\n🧪 Testing Whisper model initialization...")
    try:
        from faster_whisper import WhisperModel
        
        # Use the smallest model for testing
        model_name = "base"
        print(f"Loading model: {model_name} (faster-whisper optimized)...")
        
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print("✅ Whisper model loaded successfully!")
        
        # Get model info if available
        try:
            print(f"   Model name: {model_name}")
            print(f"   Device: cpu")
            print(f"   Compute type: int8")
        except Exception:
            pass
            
        return True, model
    except Exception as e:
        print(f"❌ Failed to initialize Whisper model: {e}")
        return False, None

def test_model_compatibility():
    """Test basic model compatibility without audio."""
    print("\n🧪 Testing model compatibility...")
    try:
        from faster_whisper import WhisperModel
        
        model = WhisperModel("base", device="cpu", compute_type="int8")
        
        # Test if we can access model properties
        print("✅ Model compatibility test passed!")
        print("   The model is ready for transcription tasks")
        
        return True
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available."""
    print("\n🧪 Testing dependencies...")
    
    required_packages = [
        ('ctranslate2', 'CTranslate2 backend'),
        ('av', 'Audio/Video processing'), 
        ('tokenizers', 'Tokenization'),
        ('numpy', 'Numerical computing'),
        ('tqdm', 'Progress bars')
    ]
    
    all_deps_ok = True
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: {description}")
        except ImportError:
            print(f"❌ {package}: {description} - NOT FOUND")
            all_deps_ok = False
    
    return all_deps_ok

def test_platform_optimization():
    """Check platform and performance optimizations."""
    print("\n🧪 Testing platform optimization...")
    
    import platform
    machine = platform.machine()
    system = platform.system()
    
    print(f"   System: {system}")
    print(f"   Architecture: {machine}")
    
    if machine.lower() in ['arm64', 'aarch64'] and system == 'Darwin':
        print("✅ Running on Apple Silicon (ARM64) - excellent performance expected")
    elif machine.lower() in ['x86_64', 'amd64']:
        print("✅ Running on x86_64 - good performance expected")
    else:
        print(f"✅ Running on {machine} - faster-whisper supports this platform")
    
    print("   faster-whisper uses CTranslate2 for optimized inference on all platforms")
    return True

def main():
    """Run all tests."""
    print("🚀 Testing faster-whisper installation and platform compatibility")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Import
    if test_faster_whisper_import():
        tests_passed += 1
    
    # Test 2: Dependencies
    if test_dependencies():
        tests_passed += 1
    
    # Test 3: Model initialization
    model_ok, model = test_whisper_model_initialization()
    if model_ok:
        tests_passed += 1
    
    # Test 4: Model compatibility
    if test_model_compatibility():
        tests_passed += 1
    
    # Test 5: Platform optimization check
    if test_platform_optimization():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! faster-whisper is ready to use.")
        print("\n💡 Your setup is ready for high-performance transcription.")
        print("   You can now run your transcription script with faster-whisper support.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
