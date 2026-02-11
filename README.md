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

Process a transcript file and output to stdout:

```bash
python src/transcript_processor.py example_input.json
```

### Save Output to File

```bash
python src/transcript_processor.py example_input.json output.json
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

The output is a JSON array of speaker segments:

```json
[
  {
    "speaker": "Unknown",
    "text": "We should send the proposal today.",
    "intent": "suggestion"
  },
  {
    "speaker": "John",
    "text": "I'll do that.",
    "intent": "commitment"
  }
]
```

## Intent Categories

- `suggestion` - Proposals or recommendations
- `commitment` - Promises or agreements to act
- `information` - Facts or updates shared
- `question` - Questions asked
- `decision` - Decisions made
- `action_item` - Tasks assigned

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
