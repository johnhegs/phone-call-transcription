#!/usr/bin/env python3
"""
Transcribe stereo phone call recording and summarize using local Ollama instance.
Left and right stereo channels represent different speakers.
Creates chronologically ordered, human-readable transcripts.
"""

import os
import json
import requests
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import speech_recognition as sr
import whisper
from pathlib import Path
from datetime import datetime, timedelta
import re
import configparser
import argparse
import glob
import time
from typing import List, Tuple
import numpy as np


class CallTranscriber:
    def __init__(self, config_file="config.txt", prompt_file="prompt_template.txt"):
        """Initialize the transcriber with configurable settings."""
        # Load configuration
        self.config = self.load_config(config_file)
        self.prompt_template = self.load_prompt_template(prompt_file)
        
        # Initialize components with config values
        self.ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
        self.ollama_model = self.config.get('ollama_model', 'llama3.1:latest')
        self.whisper_model_name = self.config.get('whisper_model', 'base')
        self.segment_merge_threshold = float(self.config.get('segment_merge_threshold', '2.0'))
        self.left_speaker_name = self.config.get('left_speaker_name', 'Speaker 1')
        self.right_speaker_name = self.config.get('right_speaker_name', 'Speaker 2')
        self.force_language = self.config.get('force_language', 'auto').lower()
        
        self.recognizer = sr.Recognizer()
        
        # Load Whisper model
        print(f"Loading Whisper model: {self.whisper_model_name}...")
        self.whisper_model = whisper.load_model(self.whisper_model_name)
        
        print(f"Configuration loaded:")
        print(f"  - Ollama Model: {self.ollama_model}")
        print(f"  - Whisper Model: {self.whisper_model_name}")
        print(f"  - Speaker Labels: {self.left_speaker_name}, {self.right_speaker_name}")
    
    def load_config(self, config_file):
        """Load configuration from file."""
        config = {}
        
        if not os.path.exists(config_file):
            print(f"Warning: Config file '{config_file}' not found. Using defaults.")
            return config
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            config[key.strip().lower()] = value.strip()
            
            print(f"✅ Configuration loaded from {config_file}")
            
        except Exception as e:
            print(f"Warning: Error loading config file: {e}")
        
        return config
    
    def load_prompt_template(self, prompt_file):
        """Load prompt template from file."""
        if not os.path.exists(prompt_file):
            print(f"Warning: Prompt file '{prompt_file}' not found. Using default prompt.")
            return self.get_default_prompt()
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                template = f.read().strip()
            
            print(f"✅ Prompt template loaded from {prompt_file}")
            return template
            
        except Exception as e:
            print(f"Warning: Error loading prompt file: {e}. Using default.")
            return self.get_default_prompt()
    
    def get_default_prompt(self):
        """Return default prompt if template file is not available."""
        return """Please analyse the following phone call transcript and provide a comprehensive summary:

{TRANSCRIPT}

Please provide a clear, structured summary:"""
    
    def save_prompt_template(self, prompt_file="prompt_template.txt"):
        """Save current prompt template to file for user editing."""
        try:
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(self.prompt_template)
            print(f"✅ Prompt template saved to {prompt_file}")
        except Exception as e:
            print(f"Error saving prompt template: {e}")
        
    def separate_stereo_channels(self, audio_file_path):
        """
        Separate stereo MP3 into left and right channel WAV files.
        
        Args:
            audio_file_path (str): Path to the stereo MP3 file
            
        Returns:
            tuple: Paths to left and right channel files
        """
        print(f"Loading audio file: {audio_file_path}")
        audio = AudioSegment.from_mp3(audio_file_path)
        
        # Split into mono channels
        channels = audio.split_to_mono()
        left_channel = channels[0]  # Left channel (Speaker 1)
        right_channel = channels[1]  # Right channel (Speaker 2)
        
        # Export as WAV files for better transcription compatibility
        left_path = "left_channel_speaker1.wav"
        right_path = "right_channel_speaker2.wav"
        
        print("Exporting separated channels...")
        left_channel.export(left_path, format="wav")
        right_channel.export(right_path, format="wav")
        
        return left_path, right_path
    
    def transcribe_with_whisper_timing(self, audio_file_path, speaker_label, speaker_id):
        """
        Transcribe audio using OpenAI Whisper with timing information.
        
        Args:
            audio_file_path (str): Path to audio file
            speaker_label (str): Label for the speaker
            speaker_id (str): Unique speaker identifier ("left" or "right")
            
        Returns:
            list: List of segments with timing and text
        """
        print(f"Transcribing {speaker_label} using Whisper with timing...")
        
        try:
            # Suppress progress bars and warnings during transcription
            import warnings
            import logging
            
            # Temporarily suppress warnings and progress bars
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Disable tqdm progress bars for this transcription
                import os
                old_tqdm_disable = os.environ.get('TQDM_DISABLE', '')
                os.environ['TQDM_DISABLE'] = '1'
                
                try:
                    # Transcribe with word-level timestamps
                    transcribe_kwargs = {
                        'word_timestamps': True,
                        'verbose': False,
                        'no_speech_threshold': 0.6,
                        'condition_on_previous_text': False
                    }
                    
                    # Add language parameter if not auto-detection
                    if self.force_language and self.force_language != 'auto':
                        transcribe_kwargs['language'] = self.force_language
                        print(f"Forcing language to: {self.force_language}")
                    
                    result = self.whisper_model.transcribe(audio_file_path, **transcribe_kwargs)
                finally:
                    # Restore original TQDM_DISABLE setting
                    if old_tqdm_disable:
                        os.environ['TQDM_DISABLE'] = old_tqdm_disable
                    elif 'TQDM_DISABLE' in os.environ:
                        del os.environ['TQDM_DISABLE']
            
            segments = []
            for segment in result.get('segments', []):
                if segment['text'].strip():
                    segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'].strip(),
                        'speaker': speaker_label,
                        'speaker_id': speaker_id
                    })
            
            return segments
                
        except Exception as e:
            print(f"Whisper transcription failed for {speaker_label}: {e}")
            # Fallback to basic transcription
            result = self.whisper_model.transcribe(audio_file_path)
            text = result.get('text', '').strip()
            if text:
                return [{
                    'start': 0,
                    'end': 0,
                    'text': text,
                    'speaker': speaker_label,
                    'speaker_id': speaker_id
                }]
            return []
    
    def transcribe_with_speech_recognition(self, audio_file_path, speaker_label):
        """
        Fallback transcription using SpeechRecognition library.
        
        Args:
            audio_file_path (str): Path to audio file
            speaker_label (str): Label for the speaker
            
        Returns:
            str: Transcribed text with speaker label
        """
        print(f"Transcribing {speaker_label} using Google Speech Recognition...")
        
        try:
            with sr.AudioFile(audio_file_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.record(source)
                
            # Use Google's speech recognition
            text = self.recognizer.recognize_google(audio)
            return f"{speaker_label}: {text}"
            
        except sr.UnknownValueError:
            return f"{speaker_label}: [Speech not recognized]"
        except sr.RequestError as e:
            return f"{speaker_label}: [Error: {e}]"
    
    def merge_segments_chronologically(self, left_segments, right_segments):
        """
        Merge transcription segments from both channels in chronological order.
        
        Args:
            left_segments (list): Segments from left channel
            right_segments (list): Segments from right channel
            
        Returns:
            list: Chronologically ordered segments
        """
        all_segments = left_segments + right_segments
        
        # Sort by start time
        all_segments.sort(key=lambda x: x['start'])
        
        # Merge consecutive segments from the same speaker
        merged_segments = []
        current_segment = None
        
        for segment in all_segments:
            if (current_segment is None or 
                current_segment['speaker_id'] != segment['speaker_id'] or
                segment['start'] - current_segment['end'] > self.segment_merge_threshold):
                
                if current_segment:
                    merged_segments.append(current_segment)
                
                current_segment = segment.copy()
            else:
                # Merge with current segment
                current_segment['text'] += " " + segment['text']
                current_segment['end'] = segment['end']
        
        if current_segment:
            merged_segments.append(current_segment)
        
        return merged_segments
    
    def format_time(self, seconds):
        """
        Format seconds into MM:SS format.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted time string
        """
        if seconds == 0:
            return "00:00"
        
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def create_chronological_transcript(self, segments):
        """
        Create a human-readable chronological transcript.
        
        Args:
            segments (list): Chronologically ordered segments
            
        Returns:
            str: Formatted transcript
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        transcript = f"""{'='*80}
PHONE CALL TRANSCRIPT
{'='*80}
Generated: {timestamp}
Total Duration: {self.format_time(segments[-1]['end']) if segments else '00:00'}
Speakers: 2 (Left Channel = Speaker 1, Right Channel = Speaker 2)
{'='*80}
\n"""
        
        for i, segment in enumerate(segments):
            # Format speaker name using configurable names
            speaker_name = self.left_speaker_name if segment['speaker_id'] == 'left' else self.right_speaker_name
            time_stamp = self.format_time(segment['start'])
            
            # Clean up the text
            text = self.clean_text(segment['text'])
            
            # Add segment to transcript
            transcript += f"[{time_stamp}] {speaker_name}: {text}\n\n"
        
        transcript += f"{'='*80}\nEND OF TRANSCRIPT\n{'='*80}"
        
        return transcript
    
    def clean_text(self, text):
        """
        Clean and format transcribed text for better readability.
        
        Args:
            text (str): Raw transcribed text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Capitalize first letter of sentences
        sentences = re.split(r'[.!?]+', text)
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                cleaned_sentences.append(sentence)
        
        # Rejoin sentences
        if cleaned_sentences:
            text = '. '.join(cleaned_sentences)
            if not text.endswith('.') and not text.endswith('!') and not text.endswith('?'):
                text += '.'
        
        return text
    
    def query_ollama(self, prompt, model="llama3.1:latest"):
        """
        Send a query to local Ollama instance.
        
        Args:
            prompt (str): The prompt to send to the model
            model (str): Model name to use (default: llama3.2)
            
        Returns:
            str: Response from the model
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Make sure it's running on http://localhost:11434"
        except requests.exceptions.Timeout:
            return "Error: Request to Ollama timed out"
        except Exception as e:
            return f"Error querying Ollama: {e}"
    
    def summarize_call(self, transcript):
        """
        Generate a summary of the call using Ollama with configurable prompt.
        
        Args:
            transcript (str): The full call transcript
            
        Returns:
            str: Summary of the call
        """
        # Use the configurable prompt template
        prompt = self.prompt_template.replace('{TRANSCRIPT}', transcript)
        
        print("Generating summary using Ollama...")
        print(f"Using model: {self.ollama_model}")
        return self.query_ollama(prompt, self.ollama_model)
    
    def save_results(self, transcript, summary):
        """
        Save transcript and summary to files.
        
        Args:
            transcript (str): Full transcript
            summary (str): Generated summary
        """
        # Save transcript
        with open("call_transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)
        
        # Save summary
        with open("call_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        
        print("Results saved to:")
        print("- call_transcript.txt")
        print("- call_summary.txt")
    
    def create_conversation_analysis(self, segments, output_path=None):
        """
        Create additional conversation analysis and statistics.
        
        Args:
            segments (list): Chronologically ordered segments
            output_path (str): Optional path to save analysis file
        """
        if not segments:
            return
        
        # Calculate speaking time for each speaker
        speaker1_time = sum(s['end'] - s['start'] for s in segments if s['speaker_id'] == 'left')
        speaker2_time = sum(s['end'] - s['start'] for s in segments if s['speaker_id'] == 'right')
        total_time = segments[-1]['end'] if segments else 0
        
        # Count speaking turns
        speaker1_turns = len([s for s in segments if s['speaker_id'] == 'left'])
        speaker2_turns = len([s for s in segments if s['speaker_id'] == 'right'])
        
        # Create analysis report
        analysis = f"""{'='*80}
CONVERSATION ANALYSIS
{'='*80}
Call Duration: {self.format_time(total_time)}

Speaking Time:
- {self.left_speaker_name}: {self.format_time(speaker1_time)} ({speaker1_time/total_time*100:.1f}%)
- {self.right_speaker_name}: {self.format_time(speaker2_time)} ({speaker2_time/total_time*100:.1f}%)

Speaking Turns:
- {self.left_speaker_name}: {speaker1_turns} turns
- {self.right_speaker_name}: {speaker2_turns} turns

Average Turn Length:
- {self.left_speaker_name}: {self.format_time(speaker1_time/speaker1_turns) if speaker1_turns > 0 else '00:00'}
- {self.right_speaker_name}: {self.format_time(speaker2_time/speaker2_turns) if speaker2_turns > 0 else '00:00'}
{'='*80}"""
        
        # Save analysis to specified path or default location
        analysis_file = output_path or "conversation_analysis.txt"
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis)
        
        print("\n" + analysis)
        print(f"\nConversation analysis saved to: {analysis_file}")
    
    def cleanup_temp_files(self):
        """Remove temporary audio files."""
        temp_files = ["left_channel_speaker1.wav", "right_channel_speaker2.wav"]
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Cleaned up: {file}")
    
    def process_single_file(self, audio_file_path: str, transcript_output: str, summary_output: str, analysis_output: str):
        """
        Process a single audio file with specified output paths.
        
        Args:
            audio_file_path (str): Path to the audio file
            transcript_output (str): Path for transcript output
            summary_output (str): Path for summary output
            analysis_output (str): Path for analysis output
        """
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(audio_file_path)}")
            print(f"{'='*60}")
            
            # Step 1: Separate stereo channels
            left_path, right_path = self.separate_stereo_channels(audio_file_path)
            
            # Step 2: Transcribe each channel with timing information
            left_segments = self.transcribe_with_whisper_timing(
                left_path, "Speaker 1 (Left Channel)", "left"
            )
            right_segments = self.transcribe_with_whisper_timing(
                right_path, "Speaker 2 (Right Channel)", "right"
            )
            
            # Step 3: Merge segments chronologically
            print("Merging segments chronologically...")
            merged_segments = self.merge_segments_chronologically(left_segments, right_segments)
            
            # Step 4: Create chronological transcript
            print("Creating formatted transcript...")
            chronological_transcript = self.create_chronological_transcript(merged_segments)
            
            # Step 5: Generate summary
            summary = self.summarize_call(chronological_transcript)
            
            # Step 6: Save results to specified paths
            with open(transcript_output, "w", encoding="utf-8") as f:
                f.write(chronological_transcript)
            print(f"Transcript saved to: {transcript_output}")
            
            with open(summary_output, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"Summary saved to: {summary_output}")
            
            # Step 7: Create and save conversation analysis
            self.create_conversation_analysis(merged_segments, analysis_output)
            
            return True
        except Exception as e:
            print(f"Error processing {audio_file_path}: {e}")
            return False
        finally:
            # Clean up temporary files
            self.cleanup_temp_files()
    


class BatchCallTranscriber(CallTranscriber):
    """
    Extends CallTranscriber to handle batch processing of multiple audio files.
    """
    
    def __init__(self, config_file="config.txt", prompt_file="prompt_template.txt"):
        super().__init__(config_file, prompt_file)
        
        # Load batch processing configuration
        self.input_folder = self.config.get('input_folder', 'calls_to_process')
        self.transcript_folder = self.config.get('output_transcript_folder', 'calls_transcribed')
        self.summary_folder = self.config.get('output_summary_folder', 'calls_summary')
        self.analysis_folder = self.config.get('output_analysis_folder', 'calls_analysis')
        self.skip_existing = self.config.get('skip_existing_files', 'true').lower() == 'true'
        
        print(f"Batch processing configuration:")
        print(f"  - Input folder: {self.input_folder}")
        print(f"  - Transcript folder: {self.transcript_folder}")
        print(f"  - Summary folder: {self.summary_folder}")
        print(f"  - Analysis folder: {self.analysis_folder}")
        print(f"  - Skip existing files: {self.skip_existing}")
    
    def ensure_output_folders(self) -> None:
        """
        Create output folders if they don't exist.
        """
        folders = [self.transcript_folder, self.summary_folder, self.analysis_folder]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            print(f"✅ Ensured folder exists: {folder}")
    
    def find_audio_files(self, input_folder: str = None) -> List[str]:
        """
        Find all .mp3 files in the input folder.
        
        Args:
            input_folder (str): Override default input folder
        
        Returns:
            List[str]: List of .mp3 file paths
        """
        folder = input_folder or self.input_folder
        
        if not os.path.exists(folder):
            print(f"Input folder '{folder}' does not exist.")
            return []
        
        mp3_files = glob.glob(os.path.join(folder, "*.mp3"))
        mp3_files.sort()  # Process files in alphabetical order
        
        print(f"Found {len(mp3_files)} .mp3 files in '{folder}'")
        return mp3_files
    
    def get_output_paths(self, audio_file_path: str) -> Tuple[str, str, str]:
        """
        Generate output file paths for transcript, summary, and analysis.
        
        Args:
            audio_file_path (str): Path to the input audio file
        
        Returns:
            Tuple[str, str, str]: (transcript_path, summary_path, analysis_path)
        """
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        
        transcript_path = os.path.join(self.transcript_folder, f"{base_name}.txt")
        summary_path = os.path.join(self.summary_folder, f"{base_name}.txt")
        analysis_path = os.path.join(self.analysis_folder, f"{base_name}.txt")
        
        return transcript_path, summary_path, analysis_path
    
    def should_process_file(self, audio_file_path: str) -> bool:
        """
        Check if file should be processed based on existing outputs.
        
        Args:
            audio_file_path (str): Path to the input audio file
        
        Returns:
            bool: True if file should be processed
        """
        if not self.skip_existing:
            return True
        
        transcript_path, summary_path, analysis_path = self.get_output_paths(audio_file_path)
        
        # Check if all output files already exist
        all_exist = (os.path.exists(transcript_path) and 
                    os.path.exists(summary_path) and 
                    os.path.exists(analysis_path))
        
        if all_exist:
            print(f"⏭️  Skipping {os.path.basename(audio_file_path)} (outputs already exist)")
            return False
        
        return True
    
    def process_batch(self, input_folder: str = None) -> None:
        """
        Process all .mp3 files in the input folder.
        
        Args:
            input_folder (str): Override default input folder
        """
        print(f"\n{'='*80}")
        print("STARTING BATCH PROCESSING")
        print(f"{'='*80}")
        
        # Ensure output folders exist
        self.ensure_output_folders()
        
        # Find audio files
        audio_files = self.find_audio_files(input_folder)
        
        if not audio_files:
            print("No .mp3 files found to process.")
            return
        
        # Filter files that need processing
        files_to_process = [f for f in audio_files if self.should_process_file(f)]
        
        if not files_to_process:
            print("All files have already been processed (use --force to reprocess).")
            return
        
        print(f"\nProcessing {len(files_to_process)} files...")
        
        # Track processing statistics
        start_time = time.time()
        successful = 0
        failed = 0
        failed_files = []
        
        for i, audio_file in enumerate(files_to_process, 1):
            print(f"\n{'='*80}")
            print(f"Processing file {i}/{len(files_to_process)}: {os.path.basename(audio_file)}")
            print(f"{'='*80}")
            
            # Get output paths
            transcript_path, summary_path, analysis_path = self.get_output_paths(audio_file)
            
            # Process the file
            file_start_time = time.time()
            success = self.process_single_file(
                audio_file, transcript_path, summary_path, analysis_path
            )
            file_duration = time.time() - file_start_time
            
            if success:
                successful += 1
                print(f"✅ Completed in {file_duration:.1f} seconds")
            else:
                failed += 1
                failed_files.append(os.path.basename(audio_file))
                print(f"❌ Failed after {file_duration:.1f} seconds")
            
            # Show progress
            remaining = len(files_to_process) - i
            if remaining > 0:
                avg_time = (time.time() - start_time) / i
                estimated_remaining = avg_time * remaining
                print(f"📊 Progress: {i}/{len(files_to_process)} | Estimated time remaining: {estimated_remaining/60:.1f} minutes")
        
        # Final summary
        total_duration = time.time() - start_time
        self.print_batch_summary(successful, failed, failed_files, total_duration)
    
    def print_batch_summary(self, successful: int, failed: int, failed_files: List[str], duration: float) -> None:
        """
        Print a summary of the batch processing results.
        
        Args:
            successful (int): Number of successfully processed files
            failed (int): Number of failed files
            failed_files (List[str]): List of failed file names
            duration (float): Total processing duration in seconds
        """
        print(f"\n{'='*80}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Successfully processed: {successful} files")
        print(f"❌ Failed: {failed} files")
        print(f"⏱️  Total time: {duration/60:.1f} minutes ({duration/3600:.1f} hours)")
        
        if successful > 0:
            print(f"📈 Average time per file: {duration/successful:.1f} seconds")
        
        if failed_files:
            print(f"\n❌ Failed files:")
            for file in failed_files:
                print(f"   - {file}")
        
        print(f"\n📁 Output folders:")
        print(f"   - Transcripts: {self.transcript_folder}")
        print(f"   - Summaries: {self.summary_folder}")
        print(f"   - Analysis: {self.analysis_folder}")
        
        print(f"\n{'='*80}")


def main():
    """Main function with CLI argument parsing for single and batch processing."""
    parser = argparse.ArgumentParser(
        description="Transcribe and summarise audio files using Whisper and Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file (default behavior)
  python transcribe_and_summarize.py
  python transcribe_and_summarize.py --file activitycall.mp3
  
  # Process all files in batch mode
  python transcribe_and_summarize.py --batch
  
  # Force English language for all transcriptions
  python transcribe_and_summarize.py --batch --language english
  
  # Process files from custom input folder
  python transcribe_and_summarize.py --batch --input-folder custom_calls
  
  # Force reprocessing of all files (skip existing file check)
  python transcribe_and_summarize.py --batch --force
  
  # Process single file with forced language
  python transcribe_and_summarize.py --file call.mp3 --language english
"""
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--batch", 
        action="store_true",
        help="Process all .mp3 files in the input folder (batch mode)"
    )
    mode_group.add_argument(
        "--file", 
        type=str,
        default="activitycall.mp3",
        help="Process a specific audio file (single mode) - default: activitycall.mp3"
    )
    
    # Batch mode options
    parser.add_argument(
        "--input-folder",
        type=str,
        help="Input folder for batch processing (overrides config setting)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of files even if outputs already exist"
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        default="config.txt",
        help="Configuration file path - default: config.txt"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prompt_template.txt",
        help="Prompt template file path - default: prompt_template.txt"
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Force transcription language (overrides config) - e.g., 'english', 'spanish', 'auto'"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("AUDIO TRANSCRIPTION AND SUMMARISATION")
    print(f"{'='*80}")
    
    try:
        if args.batch:
            # Batch processing mode
            print("Mode: Batch Processing")
            
            transcriber = BatchCallTranscriber(args.config, args.prompt)
            
            # Override language setting if provided via command line
            if args.language:
                transcriber.force_language = args.language.lower()
                print(f"Language override: {args.language}")
            
            # Override skip_existing if --force is used
            if args.force:
                transcriber.skip_existing = False
                print("Force mode: Will reprocess existing files")
            
            transcriber.process_batch(args.input_folder)
            
        else:
            # Single file processing mode
            print(f"Mode: Single File Processing")
            print(f"File: {args.file}")
            
            # Check if audio file exists - try multiple locations
            file_path = args.file
            if not os.path.exists(file_path):
                # If running in Docker, also check calls_to_process directory
                alternative_path = os.path.join("calls_to_process", args.file)
                if os.path.exists(alternative_path):
                    file_path = alternative_path
                    print(f"Found file in calls_to_process directory: {file_path}")
                else:
                    print(f"Error: Audio file '{args.file}' not found!")
                    print("\nAvailable .mp3 files in current directory:")
                    mp3_files = glob.glob("*.mp3")
                    if mp3_files:
                        for file in sorted(mp3_files):
                            print(f"  - {file}")
                    else:
                        print("  None found")
                    
                    # Also check calls_to_process directory
                    calls_dir = "calls_to_process"
                    if os.path.exists(calls_dir):
                        print(f"\nAvailable .mp3 files in {calls_dir} directory:")
                        calls_mp3_files = glob.glob(os.path.join(calls_dir, "*.mp3"))
                        if calls_mp3_files:
                            for file in sorted(calls_mp3_files):
                                print(f"  - {os.path.basename(file)}")
                        else:
                            print("  None found")
                    return 1
            
            # Initialize transcriber
            transcriber = CallTranscriber(args.config, args.prompt)
            
            # Override language setting if provided via command line
            if args.language:
                transcriber.force_language = args.language.lower()
                print(f"Language override: {args.language}")
            
            # Generate output paths for single file processing
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            transcript_path = os.path.join("calls_transcribed", f"{base_name}.txt")
            summary_path = os.path.join("calls_summary", f"{base_name}.txt")
            analysis_path = os.path.join("calls_analysis", f"{base_name}.txt")
            
            # Ensure output directories exist
            os.makedirs("calls_transcribed", exist_ok=True)
            os.makedirs("calls_summary", exist_ok=True)
            os.makedirs("calls_analysis", exist_ok=True)
            
            # Process the call
            print("\nStarting call transcription and summarisation...")
            transcriber.process_single_file(file_path, transcript_path, summary_path, analysis_path)
        
        print("\n✅ Process completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Process failed with error: {e}")
        return 1


if __name__ == "__main__":
    main()
