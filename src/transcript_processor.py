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
    """Represents a segment of speech with speaker, text, and intent."""
    speaker: str = Field(description="The name of the speaker, or 'Unknown' if not identified")
    text: str = Field(description="The text content of what the speaker said")
    intent: Literal[
        "suggestion",
        "commitment",
        "information",
        "question",
        "decision",
        "action_item"
    ] = Field(description="The intent or purpose of this speech segment")


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
            ("system", """You are an expert at analyzing meeting transcripts. 
Your task is to break down the transcript into individual speaker segments and classify each segment's intent.

For each segment, identify:
1. The speaker name (use "Unknown" if the speaker cannot be identified)
2. The exact text spoken
3. The intent category:
   - "suggestion": Proposals, recommendations, or ideas
   - "commitment": Promises, agreements to act, or confirmations
   - "information": Facts, updates, or information shared
   - "question": Questions asked
   - "decision": Decisions made or conclusions reached
   - "action_item": Tasks assigned or action items identified

Return a JSON object with a "segments" key containing a list of all speaker segments in the order they appear in the transcript."""),
            ("human", "Analyze the following meeting transcript:\n\n{transcript}")
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
        print("Usage: python transcript_processor.py <input.json> [output.json]", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        logger.info("Step 1/2: Initializing processor and loading input.")
        processor = TranscriptProcessor()
        segments = processor.process_from_file(input_file)
        
        logger.info("Step 2/2: Writing output.")
        # Convert to dict for JSON serialization
        result = [segment.model_dump() for segment in segments]
        
        # Output results
        output_json = json.dumps(result, indent=2, ensure_ascii=False)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_json)
            logger.info("Results saved to %s", output_file)
        else:
            print(output_json)
        logger.info("All done.")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
