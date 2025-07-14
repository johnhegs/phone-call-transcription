#!/usr/bin/env python3
"""
Configuration helper for the transcription application.
Allows non-technical users to easily modify prompts and settings.
"""

import os
import sys

def display_current_prompt():
    """Display the current prompt template."""
    prompt_file = "prompt_template.txt"
    
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("="*80)
        print("CURRENT PROMPT TEMPLATE")
        print("="*80)
        print(content)
        print("="*80)
    else:
        print("❌ Prompt template file not found!")

def display_current_config():
    """Display the current configuration."""
    config_file = "config.txt"
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("="*80)
        print("CURRENT CONFIGURATION")
        print("="*80)
        print(content)
        print("="*80)
    else:
        print("❌ Configuration file not found!")

def edit_prompt_interactive():
    """Interactively edit the prompt template."""
    prompt_file = "prompt_template.txt"
    
    print("="*80)
    print("EDIT PROMPT TEMPLATE")
    print("="*80)
    print("Instructions:")
    print("- Keep {TRANSCRIPT} in your prompt - this will be replaced with the actual transcript")
    print("- Use clear, specific instructions for the AI")
    print("- You can add or modify analysis categories as needed")
    print("- Press Enter twice when finished")
    print("-" * 40)
    
    lines = []
    print("Enter your prompt template (press Enter twice to finish):")
    
    while True:
        line = input()
        if line == "" and len(lines) > 0 and lines[-1] == "":
            break
        lines.append(line)
    
    # Remove the final empty line
    if lines and lines[-1] == "":
        lines = lines[:-1]
    
    new_prompt = "\n".join(lines)
    
    if new_prompt.strip():
        # Backup existing file
        if os.path.exists(prompt_file):
            os.rename(prompt_file, f"{prompt_file}.backup")
            print(f"✅ Previous prompt backed up to {prompt_file}.backup")
        
        # Save new prompt
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(new_prompt)
        
        print(f"✅ New prompt saved to {prompt_file}")
        
        # Validate that {TRANSCRIPT} is present
        if "{TRANSCRIPT}" not in new_prompt:
            print("⚠️  Warning: Your prompt doesn't contain {TRANSCRIPT}")
            print("   The AI won't receive the actual transcript without this placeholder!")
    else:
        print("❌ Empty prompt not saved")

def edit_config_interactive():
    """Interactively edit configuration settings."""
    config_file = "config.txt"
    
    print("="*80)
    print("EDIT CONFIGURATION")
    print("="*80)
    
    # Default config values
    config_options = {
        'ollama_model': {
            'current': 'llama3.1:latest',
            'description': 'AI model to use for analysis',
            'examples': ['llama3.1:latest', 'llama2:latest', 'mistral:latest']
        },
        'whisper_model': {
            'current': 'base',
            'description': 'Whisper model for transcription',
            'examples': ['tiny', 'base', 'small', 'medium', 'large']
        },
        'left_speaker_name': {
            'current': 'Speaker 1',
            'description': 'Name for left channel speaker',
            'examples': ['Customer', 'Agent', 'John', 'Support Rep']
        },
        'right_speaker_name': {
            'current': 'Speaker 2',
            'description': 'Name for right channel speaker',
            'examples': ['Agent', 'Customer', 'Mary', 'Client']
        },
        'segment_merge_threshold': {
            'current': '2.0',
            'description': 'Seconds gap before treating as new speaker turn',
            'examples': ['1.0', '2.0', '3.0', '5.0']
        }
    }
    
    # Load current config if exists
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip().lower()
                    if key in config_options:
                        config_options[key]['current'] = value.strip()
    
    new_config = []
    new_config.append("# Transcription and AI Analysis Configuration")
    new_config.append("# Edit these settings to customize the application behavior")
    new_config.append("")
    
    for key, info in config_options.items():
        print(f"\n{info['description']}")
        print(f"Current value: {info['current']}")
        print(f"Examples: {', '.join(info['examples'])}")
        
        new_value = input(f"New value (press Enter to keep current): ").strip()
        
        if new_value:
            final_value = new_value
        else:
            final_value = info['current']
        
        new_config.append(f"{key.upper()}={final_value}")
    
    # Backup existing file
    if os.path.exists(config_file):
        os.rename(config_file, f"{config_file}.backup")
        print(f"\n✅ Previous config backed up to {config_file}.backup")
    
    # Save new config
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_config))
    
    print(f"✅ New configuration saved to {config_file}")

def create_example_prompts():
    """Create example prompt templates for different use cases."""
    
    examples = {
        "customer_service": """Please analyze this customer service call transcript and provide:

1. **Customer Issue**: What problem or request did the customer have?
2. **Resolution Provided**: How was the issue addressed?
3. **Customer Satisfaction**: Was the customer satisfied with the resolution?
4. **Agent Performance**: How well did the agent handle the call?
5. **Follow-up Required**: Are there any actions needed after this call?
6. **Call Quality Score**: Rate the call quality from 1-10 with reasoning

Call Transcript:
{TRANSCRIPT}

Provide a detailed analysis:""",

        "sales_call": """Analyze this sales call transcript and provide:

1. **Prospect Needs**: What needs or pain points were identified?
2. **Products Discussed**: What solutions were presented?
3. **Objections Raised**: What concerns did the prospect express?
4. **Next Steps**: What follow-up actions were agreed upon?
5. **Sales Opportunity**: Rate the likelihood of closing (High/Medium/Low)
6. **Key Decision Makers**: Who are the important people mentioned?

Call Transcript:
{TRANSCRIPT}

Summary:""",

        "medical_consultation": """Analyze this medical consultation transcript:

1. **Chief Complaint**: Primary reason for the consultation
2. **Symptoms Discussed**: Key symptoms and concerns raised
3. **Assessment**: Medical professional's evaluation
4. **Treatment Plan**: Recommended treatments or interventions
5. **Follow-up Instructions**: Next steps for the patient
6. **Prescriptions**: Any medications mentioned

Call Transcript:
{TRANSCRIPT}

Medical Summary:""",

        "meeting_notes": """Create meeting notes from this call transcript:

1. **Attendees**: Who participated in the call
2. **Key Discussions**: Main topics covered
3. **Decisions Made**: Important decisions reached
4. **Action Items**: Tasks assigned with owners
5. **Next Meeting**: When is the follow-up scheduled
6. **Open Issues**: Unresolved matters requiring attention

Call Transcript:
{TRANSCRIPT}

Meeting Summary:"""
    }
    
    print("="*80)
    print("EXAMPLE PROMPT TEMPLATES")
    print("="*80)
    
    for name, template in examples.items():
        filename = f"prompt_example_{name}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(template)
        print(f"✅ Created: {filename}")
    
    print(f"\nTo use an example:")
    print(f"1. Copy the content from an example file")
    print(f"2. Paste it into prompt_template.txt")
    print(f"3. Or use option 2 in this menu to edit interactively")

def main():
    """Main configuration menu."""
    while True:
        print("\n" + "="*80)
        print("TRANSCRIPTION APP CONFIGURATION")
        print("="*80)
        print("1. View current prompt template")
        print("2. Edit prompt template")
        print("3. View current configuration")
        print("4. Edit configuration settings")
        print("5. Create example prompt templates")
        print("6. Exit")
        print("-" * 40)
        
        choice = input("Choose an option (1-6): ").strip()
        
        if choice == "1":
            display_current_prompt()
        elif choice == "2":
            edit_prompt_interactive()
        elif choice == "3":
            display_current_config()
        elif choice == "4":
            edit_config_interactive()
        elif choice == "5":
            create_example_prompts()
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
