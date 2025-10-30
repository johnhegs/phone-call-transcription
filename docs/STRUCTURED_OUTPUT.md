# Structured Output Feature

This document describes the structured CSV output feature that allows you to extract specific information from transcripts into a structured, Excel-compatible CSV file.

## Overview

The structured output mode processes existing transcripts and generates a single CSV file in the `calls_structured_output/` folder with one row per analyzed call. Each column in the CSV is defined by a customizable prompt that tells the AI what information to extract.

## Key Features

- **Customizable columns**: Define exactly what information to extract from each call
- **Excel-compatible**: Generates quote-delimited CSV files that work perfectly with Excel
- **Batch processing**: Analyzes all existing transcripts in one run
- **Timestamped output**: Each run generates a new file with date/time stamp
- **No re-transcription**: Uses existing transcripts, so it's fast and efficient

## Quick Start

### 1. Generate Transcripts First

Before using structured output, you need to have transcripts:

```bash
# Native
python transcribe_and_summarise.py --batch

# Docker
./docker-run.sh batch
```

### 2. Create a Structured Prompt File

Create a text file with your column definitions using this format:

```
#ColumnName
<prompt definition for this column>

#AnotherColumn
<prompt definition for this column>
```

See `structured_prompt_example.txt` for a complete example.

### 3. Run Structured Analysis

```bash
# Native
python transcribe_and_summarise.py --structured structured_prompt_example.txt

# Docker
docker-compose run --rm transcribe python transcribe_and_summarise.py --structured structured_prompt_example.txt
```

The output will be a timestamped CSV file like: `calls_structured_output/structured_output_20241030_143022.csv`

## Structured Prompt File Format

### Basic Syntax

Each column definition starts with `#ColumnName` followed by the prompt on the next lines:

```
#ColumnName
Prompt definition goes here.
It can span multiple lines.

#NextColumn
Another prompt definition.
```

### Special Column: FileName

If you include a column named `FileName`, it will automatically be populated with the transcript filename (without extension). No AI analysis is needed for this column.

Example:
```
#FileName
The filename of the transcript being analysed.
```

### Example Prompt File

Here's a complete example for customer service analysis:

```
#FileName
The filename of the transcript being analysed.

#CustomerName
If the customer mentioned their first name, include it here, otherwise leave this column blank

#IssueCategory
Categorize the main issue into one of: Technical, Billing, Product Question, Complaint, or Other

#WasIssueResolved
Did the agent resolve the customer's issue? Answer YES or NO

#AgentPerformance
Give a 4 word summary of the agent's performance

#AgentRating
Give a number for the Agent's performance out of 10. 1 is awful, 10 is excellent

#CustomerSatisfaction
On balance, was the outcome of the call a happy customer? YES or NO

#FollowUpRequired
Does this call require any follow-up action? YES or NO

#KeyQuote
Extract one key quote from the customer that summarizes their main concern (include quotes)
```

## Best Practices for Prompts

### 1. Be Specific

❌ Bad: "What was discussed?"
✅ Good: "Summarize the main issue discussed in 10 words or less"

### 2. Request Specific Formats

For columns that should have standardized values:

```
#Priority
Rate the urgency as: LOW, MEDIUM, or HIGH
```

### 3. Use YES/NO for Binary Questions

```
#RequiresFollowUp
Does this call require follow-up action? Answer YES or NO
```

### 4. Set Constraints

```
#Summary
Provide a 3-sentence summary of the call outcome
```

### 5. Request Numeric Ratings

```
#SatisfactionScore
Rate customer satisfaction from 1-10, where 1 is very dissatisfied and 10 is very satisfied
```

## Output Format

The generated CSV file will:

- Have all fields enclosed in quotes (Excel-compatible)
- Properly escape quotes within field values (doubles them)
- Include a header row with column names
- Have one row per analyzed transcript
- Be saved in `calls_structured_output/` folder with timestamp: `structured_output_YYYYMMDD_HHMMSS.csv`

Example output:

```csv
"FileName","CustomerName","IssueCategory","AgentRating"
"call_001","John","Technical","8"
"call_002","","Billing","9"
"call_003","Sarah","Complaint","6"
```

## Usage Examples

### Basic Usage

```bash
python transcribe_and_summarise.py --structured my_prompt.txt
```

### With Custom Config

```bash
python transcribe_and_summarise.py --structured my_prompt.txt --config custom_config.txt
```

### Testing the Feature

Run the test script to verify everything works:

```bash
python test_structured_output.py
```

## Performance Considerations

- **Processing time**: Each column requires a separate AI query, so more columns = longer processing time
- **Recommended**: Keep columns under 10 for reasonable performance
- **Parallel processing**: Not currently supported, but may be added in future versions

### Estimated Processing Times

For a batch of 10 transcripts with 6 columns:
- Small calls (5 min): ~5-10 minutes total
- Medium calls (15 min): ~10-20 minutes total
- Large calls (30+ min): ~20-40 minutes total

## Common Use Cases

### Customer Service Analysis

Extract customer names, issue categories, resolution status, and satisfaction scores.

### Sales Call Analysis

Identify prospect needs, objections raised, products discussed, and likelihood to close.

### Medical Consultation

Extract chief complaints, symptoms, diagnoses, and treatment plans.

### Meeting Notes

Capture attendees, decisions made, action items, and next steps.

## Troubleshooting

### "No transcript files found"

Make sure you've run normal transcription first:
```bash
python transcribe_and_summarise.py --batch
```

### "Failed to parse structured prompt file"

Check your prompt file format:
- Column names must start with `#`
- No empty column names
- Make sure file encoding is UTF-8

### Inconsistent AI Responses

If the AI isn't following your instructions:
- Make the prompt more specific
- Add examples in the prompt
- Request specific formats (YES/NO, numbers, categories)

### Ollama Connection Error

Make sure Ollama is running:
```bash
# Check if running
ollama list

# Start if needed
ollama serve
```

## Advanced Tips

### Extract Multiple Pieces of Information

You can use multiple columns to break down complex information:

```
#PaymentAmount
If a payment amount was discussed, extract only the number, otherwise leave blank

#PaymentCurrency
If a payment was discussed, what currency? (USD, EUR, GBP, etc.)

#PaymentDueDate
When is the payment due? Format as YYYY-MM-DD or "Not mentioned"
```

### Create Summary Categories

```
#CallType
Categorize this call as one of: New Customer, Support Request, Complaint, Sales Inquiry, Follow-up
```

### Extract Sentiment

```
#CustomerTone
How would you describe the customer's tone? Choose one: Frustrated, Neutral, Happy, Angry, Confused
```

## Integration with Excel

### Opening in Excel

The generated CSV files are fully compatible with Microsoft Excel:

1. Double-click the CSV file, or
2. Open Excel → File → Open → Select the CSV file

Excel will automatically recognize the quote-delimited format.

### Creating Pivot Tables

Once imported, you can use Excel's pivot table features to:
- Count calls by category
- Calculate average agent ratings
- Analyze trends over time
- Create charts and visualizations

### Filtering and Sorting

Use Excel's built-in filtering to:
- Find all calls with specific issues
- Filter by agent performance rating
- Show only unresolved calls
- Search for specific customer names

## Future Enhancements

Potential improvements under consideration:

- Parallel column processing for faster analysis
- Custom output folder configuration
- Support for JSON output format
- Incremental processing (only new transcripts)
- Validation rules for column values
- Custom AI model selection per column
