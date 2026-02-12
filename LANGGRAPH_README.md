# LangGraph Action Item Extraction System

This is a new LangGraph-based system for extracting action items from meeting transcripts. It's designed to handle large transcripts reliably by processing them in chunks with cross-chunk context resolution.

## Architecture

The system uses a multi-node LangGraph workflow:

```
Transcript → Segmenter → Loop:
    RelevanceGate → (if YES) → LocalExtractor → EvidenceNormalizer → ContextResolver
→ GlobalDeduplicator → ActionFinalizer → Final JSON
```

## Components

### 1. Segmenter Node
- **Role**: Structural chunking (NO AI)
- **Logic**: Splits transcript by speaker turns, groups into 8-15 turns per chunk
- **Goal**: Preserve conversational integrity

### 2. Relevance Gate Node
- **Role**: LLM filter for work-relevant content
- **Model**: GLM4.7Flash
- **Output**: YES/NO for each chunk

### 3. Local Extractor Node
- **Role**: Extract evidence from current chunk
- **Output**: Candidate segments with action details (NOT final truth)

### 4. Evidence Normalizer Node
- **Role**: Structure cleaning (no heavy reasoning)
- **Tasks**:
  - Standardize verbs ("take care of" → "fix")
  - Trim ASR noise
  - Remove duplicates within chunk
  - Add span IDs

### 5. Context Resolver Node (Core Intelligence)
- **Role**: Cross-chunk reasoning
- **Tasks**:
  - Reference Completion: Link "I'll do that" to earlier topics
  - Ownership Linking: Connect vague mentions with commitments
  - Deadline Linking: Update deadlines from later context
  - Topic Tracking: Maintain active topic memory

### 6. Global Deduplicator Node
- **Role**: Stop loops + repetition
- **Rules**: Two actions are duplicates if:
  - Same speaker
  - Similar verb
  - Similar object
  - Same meeting window

### 7. Action Finalizer Node
- **Role**: Enforce output schema
- **Tasks**:
  - Fill nulls
  - Normalize verbs
  - Drop low-confidence actions (< 0.3)
  - Sort chronologically

## Files

- `src/langgraph_models.py` - Data models (Segment, Action, ActionDetails)
- `src/langgraph_state.py` - GraphState TypedDict definition
- `src/langgraph_nodes.py` - All node implementations
- `src/langgraph_workflow.py` - Main graph workflow definition
- `src/langgraph_main.py` - Entry point script
- `run_langgraph.py` - Simple runner script

## Usage

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the LangGraph system
python run_langgraph.py example_input.json output_langgraph.json

# Or run directly
python -m src.langgraph_main example_input.json output_langgraph.json
```

## Input Format

Input JSON file should contain:
```json
{
  "transcript_raw": "Speaker1: text here\n\nSpeaker2: more text..."
}
```

## Output Format

Output JSON contains an array of Action objects:
```json
[
  {
    "description": "Clear description of the action",
    "assignee": "Speaker name",
    "deadline": "Timeline mentioned",
    "speaker": "Who mentioned it",
    "verb": "Normalized verb (e.g., 'fix', 'send')",
    "object_text": null,
    "confidence": 0.7,
    "source_spans": ["span_id1", "span_id2"],
    "meeting_window": [0, 2]
  }
]
```

## State Management

The `GraphState` maintains:
- `chunks`: List of transcript chunks
- `chunk_index`: Current chunk being processed
- `candidate_segments`: Segments from current chunk
- `unresolved_references`: Segments with unresolved references
- `active_topics`: Topic tracking dictionary
- `merged_actions`: Accumulated actions
- `emitted_text_spans`: Anti-loop memory (set of processed spans)

## Differences from Original System

- **Chunked Processing**: Handles large transcripts by processing in chunks
- **Cross-Chunk Context**: Context Resolver maintains memory across chunks
- **Relevance Filtering**: Only processes work-relevant chunks
- **Deduplication**: Global deduplication prevents duplicate actions
- **Memory Management**: Tracks unresolved references and active topics

## Configuration

Uses the same `config.py` and `.env` settings as the original system:
- `GLM_API_URL`: API endpoint
- `MODEL_NAME`: Model name (default: "glm4-7")
- `TEMPERATURE`, `MAX_TOKENS`, etc.: Model parameters
