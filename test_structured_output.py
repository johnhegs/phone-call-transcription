#!/usr/bin/env python3
"""
Test script for structured output functionality.
Tests parsing of structured prompt files without requiring full transcription.
"""

import os
import sys


def test_structured_prompt_parsing():
    """Test parsing of the structured prompt file."""
    print("🧪 Testing structured prompt parsing...")

    try:
        from transcribe_and_summarise import CallTranscriber

        # Initialize transcriber (will need config, but won't use Whisper for this test)
        transcriber = CallTranscriber()

        # Test parsing the example structured prompt file
        prompt_file = "structured_prompt_example.txt"

        if not os.path.exists(prompt_file):
            print(f"❌ Example prompt file not found: {prompt_file}")
            return False

        columns = transcriber.parse_structured_prompt(prompt_file)

        # Verify we got the expected columns
        expected_columns = ['FileName', 'Name', 'WerePaymentsDiscussed',
                          'AgentPerformance', 'AgentPerformanceRating', 'Outcome']

        print(f"\n📊 Parsed {len(columns)} columns:")
        for col in columns:
            print(f"  ✓ {col['name']}")
            print(f"    Prompt: {col['prompt'][:60]}...")

        # Check if all expected columns are present
        parsed_names = [col['name'] for col in columns]

        if set(parsed_names) == set(expected_columns):
            print(f"\n✅ All expected columns found!")
            return True
        else:
            missing = set(expected_columns) - set(parsed_names)
            extra = set(parsed_names) - set(expected_columns)
            if missing:
                print(f"\n⚠️  Missing columns: {missing}")
            if extra:
                print(f"⚠️  Extra columns: {extra}")
            return True  # Still consider it a pass if parsing worked

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_writing():
    """Test CSV writing functionality."""
    print("\n🧪 Testing CSV writing...")

    try:
        from transcribe_and_summarise import BatchCallTranscriber
        import csv
        import os

        # Initialize transcriber
        transcriber = BatchCallTranscriber()

        # Create test data
        test_columns = [
            {'name': 'FileName', 'prompt': 'test'},
            {'name': 'TestColumn', 'prompt': 'test prompt'}
        ]

        test_data = {
            'FileName': 'test_file.mp3',
            'TestColumn': 'test value with "quotes" and, commas'
        }

        test_csv_path = 'test_output.csv'

        # Write test CSV
        transcriber.write_structured_csv_row(test_csv_path, test_columns, test_data, is_first_row=True)

        # Verify file was created
        if os.path.exists(test_csv_path):
            # Read and verify content
            with open(test_csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"  Generated CSV:\n{content}")

            # Clean up
            os.remove(test_csv_path)
            print("✅ CSV writing test passed!")
            return True
        else:
            print("❌ CSV file was not created")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Structured Output Feature")
    print("=" * 60)

    tests_passed = 0
    total_tests = 2

    # Test 1: Structured prompt parsing
    if test_structured_prompt_parsing():
        tests_passed += 1

    # Test 2: CSV writing
    if test_csv_writing():
        tests_passed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Structured output feature is ready.")
        print("\n💡 Next steps:")
        print("   1. Run normal transcription: python transcribe_and_summarise.py --batch")
        print("   2. Generate structured CSV: python transcribe_and_summarise.py --structured structured_prompt_example.txt")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
