# LangChain Meeting Transcript Processor

A CLI application that processes meeting transcripts using LangChain and a locally running GLM4.7 model to extract structured information (speaker, text, and intent).

## Features

- Processes meeting transcripts from JSON input
- Extracts speaker segments with intent classification
- Uses LangChain with structured output for reliable schema compliance
- Supports local GLM4.7 model via OpenAI-compatible API

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your GLM4.7 model is running locally with an OpenAI-compatible API endpoint (default: `http://localhost:8000/v1`)

## Configuration

Create a `.env` file (optional) to customize settings:

```env
GLM_API_URL=http://localhost:8000/v1
GLM_API_KEY=your-api-key-if-needed
MODEL_NAME=glm4-7
TEMPERATURE=0.7
MAX_TOKENS=2000
```

If no `.env` file is provided, defaults will be used.

## Usage

### Basic Usage

Process a transcript file. Output is saved to `output_structured.txt` by default:

```bash
python src/transcript_processor.py example_input.json
```

### Custom Output File

Specify a different output filename:

```bash
python src/transcript_processor.py example_input.json custom_output.json
```

### Logging and streaming

Progress and model output go to **stderr** so JSON can be piped from stdout. You’ll see:

- Step-by-step messages (loading input, calling the model, parsing, writing output)
- If the model API supports it, tokens streamed to stderr as they are generated

To save only the JSON and hide logs:  
`python src/transcript_processor.py example_input.json 2>/dev/null`

## Input Format

The input JSON file must contain a `transcript_raw` field:

```json
{
  "transcript_raw": "Speaker 1: Hello everyone. Speaker 2: Let's start the meeting."
}
```

## Output Format

The output is a JSON array of speaker segments. Each segment includes:

| Field | Description |
|-------|-------------|
| `speaker` | Speaker name |
| `text` | Exact text spoken |
| `intent` | One of: suggestion, commitment, information, question, decision, action_item, agreement, clarification, other |
| `reason` | Short explanation for the intent label |
| `resolved_context` | What earlier topic this refers to (if applicable); empty string if not |
| `context_unclear` | `true` if reference or meaning cannot be resolved from context |

Example:

```json
[
  {
    "speaker": "Unknown",
    "text": "We should send the proposal today.",
    "intent": "suggestion",
    "reason": "Proposes a concrete next step.",
    "resolved_context": "",
    "context_unclear": false
  },
  {
    "speaker": "John",
    "text": "I'll do that.",
    "intent": "commitment",
    "reason": "Explicit promise to perform the suggested action.",
    "resolved_context": "Sending the proposal",
    "context_unclear": false
  }
]
```

## Intent Categories

- `suggestion` - Proposals or recommendations
- `commitment` - Promises or agreements to act (not short acknowledgements)
- `information` - Facts or updates shared
- `question` - Questions asked
- `decision` - Decisions made (distinct from agreement)
- `action_item` - Tasks assigned
- `agreement` - Agreement with a prior point
- `clarification` - Seeking or giving clarification
- `other` - Other conversational role

## Project Structure

```
agent-ai/
├── src/
│   ├── __init__.py
│   └── transcript_processor.py    # Main processing logic
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── example_input.json              # Example input file
```

## Requirements

- Python 3.8+
- LangChain 0.1.0+
- A locally running GLM4.7 model with OpenAI-compatible API
