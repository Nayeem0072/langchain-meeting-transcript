"""Main transcript processor using LangChain with structured output."""
import json
import logging
import sys
from pathlib import Path
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Step-by-step logging to stderr (stdout reserved for JSON output)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class StreamingStderrCallbackHandler(BaseCallbackHandler):
    """Streams LLM tokens to stderr so stdout stays clean for JSON output."""

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        logger.info("LLM call started (model is generating...)")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        sys.stderr.write(token)
        sys.stderr.flush()

    def on_llm_end(self, response, **kwargs) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()
        logger.info("LLM call finished.")


class SpeakerSegment(BaseModel):
    """Represents a segment of speech with speaker, text, intent, and context metadata."""
    speaker: str = Field(description="Speaker name")
    text: str = Field(description="Exact text spoken (must be exact substring from transcript)")
    intent: Literal[
        "suggestion",
        "commitment",
        "information",
        "question",
        "decision",
        "action_item",
        "agreement",
        "clarification",
    ] = Field(description="Conversational role / intent of the segment")
    reason: str = Field(description="Short explanation for the intent label")
    resolved_context: str = Field(
        default="",
        description="What earlier topic this refers to, if applicable; empty string if not applicable",
    )
    context_unclear: bool = Field(
        default=False,
        description="True if reference or meaning cannot be resolved from context",
    )


class TranscriptSegments(BaseModel):
    """Container for a list of speaker segments (required by LangChain structured output)."""
    segments: List[SpeakerSegment] = Field(description="List of speaker segments in transcript order")


class TranscriptProcessor:
    """Processes meeting transcripts and extracts structured information."""
    
    def __init__(self):
        """Initialize the processor with LangChain model."""
        logger.info("Initializing model: %s at %s", config.MODEL_NAME, config.GLM_API_URL)
        callbacks = [StreamingStderrCallbackHandler()]
        # Initialize ChatOpenAI with custom base_url for local GLM4.7 model
        self.llm = ChatOpenAI(
            base_url=config.GLM_API_URL,
            api_key=config.GLM_API_KEY or "not-needed",
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            streaming=True,
            callbacks=callbacks,
        )
        
        # Bind structured output to the model (must be a single Pydantic model, not List[...])
        self.structured_llm = self.llm.with_structured_output(TranscriptSegments)
        logger.info("Model ready.")
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI system that extracts ONLY WORK-RELEVANT operational content from meeting transcripts.

Your task is analytical, NOT generative.

You are NOT writing a summary.
You are NOT continuing patterns.
You are NOT predicting what comes next.

You must ONLY extract segments that EXPLICITLY EXIST in the transcript text.

━━━━━━━━━━━━━━━━━━
CRITICAL ANTI-HALLUCINATION RULES
━━━━━━━━━━━━━━━━━━

• NEVER invent dialogue
• NEVER repeat a segment unless it appears again verbatim in the transcript
• If the same segment text appears multiple times in output but not in transcript → you are hallucinating
• Each output segment must map to a real, unique position in the transcript
• STOP extraction when transcript content ends — do NOT continue pattern

If unsure whether a segment exists → DO NOT OUTPUT IT.

━━━━━━━━━━━━━━━━━━
CONTEXT INTERPRETATION RULE
━━━━━━━━━━━━━━━━━━

Segments must be interpreted using surrounding conversation context, not in isolation.
However, context may ONLY come from the provided transcript.

Do NOT infer missing meetings, systems, tickets, or processes.

━━━━━━━━━━━━━━━━━━
STEP 1 — SEGMENTATION
━━━━━━━━━━━━━━━━━━

Break transcript into speaker turns exactly as written.

━━━━━━━━━━━━━━━━━━
STEP 2 — HARD WORK FILTER
━━━━━━━━━━━━━━━━━━

Only keep segments related to:

• tasks
• decisions
• plans
• timelines
• ownership
• risks
• project/product/technical/business discussion

DISCARD completely:

• small talk
• jokes
• greetings
• filler words
• ASR noise
• unclear fragments
• social talk

Do NOT output discarded segments.

━━━━━━━━━━━━━━━━━━
STEP 3 — INTENT CLASSIFICATION
━━━━━━━━━━━━━━━━━━

For each remaining segment classify ONE:

suggestion | commitment | information | question | decision | action_item | agreement | clarification

━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━

Return STRICT JSON.

Each segment must include:

{{
  "speaker": "",
  "text": "",              ← EXACT substring from transcript
  "intent": "",
  "reason": "",
  "resolved_context": "",
  "context_unclear": false
}}

━━━━━━━━━━━━━━━━━━
LOOP PREVENTION DIRECTIVE
━━━━━━━━━━━━━━━━━━

Before producing each segment, internally verify:

1. Does this exact text appear in transcript?
2. Has this exact text already been output?
3. Am I continuing a pattern instead of analyzing new transcript content?

If any answer is YES → DO NOT OUTPUT.

When no more valid segments remain → STOP.

Do not pad the list.
Do not repeat structures.
Do not continue patterns.
"""),
            ("human", "Analyze the following meeting transcript:\n\n{transcript}"),
        ])
    
    def process(self, transcript_raw: str) -> List[SpeakerSegment]:
        """
        Process a raw transcript and return structured speaker segments.
        
        Args:
            transcript_raw: The raw transcript text
            
        Returns:
            List of SpeakerSegment objects
        """
        # Create the chain
        chain = self.prompt | self.structured_llm
        
        logger.info("Calling LLM (streaming response below; this may take a while)...")
        # Invoke the chain
        result: TranscriptSegments = chain.invoke({"transcript": transcript_raw})
        logger.info("Parsing structured output...")
        segments = result.segments
        logger.info("Done. Extracted %d segment(s).", len(segments))
        return segments
    
    def process_from_file(self, input_file: str) -> List[SpeakerSegment]:
        """
        Load transcript from JSON file and process it.
        
        Args:
            input_file: Path to JSON file with transcript_raw field
            
        Returns:
            List of SpeakerSegment objects
        """
        try:
            logger.info("Loading input file: %s", input_file)
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "transcript_raw" not in data:
                raise ValueError("Input JSON must contain 'transcript_raw' field")
            
            transcript = data["transcript_raw"]
            logger.info("Input loaded (%d characters). Starting transcript analysis.", len(transcript))
            return self.process(transcript)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in input file: {e}")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python transcript_processor.py <input.json> [output.txt]", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output_structured.txt"
    
    try:
        logger.info("Step 1/2: Initializing processor and loading input.")
        processor = TranscriptProcessor()
        segments = processor.process_from_file(input_file)
        
        logger.info("Step 2/2: Writing output.")
        # Convert to dict for JSON serialization
        result = [segment.model_dump() for segment in segments]
        
        # Output results
        output_json = json.dumps(result, indent=2, ensure_ascii=False)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_json)
        logger.info("Results saved to %s", output_file)
        logger.info("All done.")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
