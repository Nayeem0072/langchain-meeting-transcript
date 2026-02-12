"""Main transcript processor using LangChain with structured output."""
import json
import logging
import sys
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator
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

# Enable debug logging for httpx and openai
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)


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


class ActionDetails(BaseModel):
    """Action details for action_item intents."""
    description: str | None = Field(default=None, description="Description of the action")
    assignee: str | None = Field(default=None, description="Who is assigned to do the action")
    deadline: str | None = Field(default=None, description="Deadline or timeline mentioned")
    confidence: float | None = Field(default=None, description="Confidence score (0.0-1.0) for action resolution")
    
    @model_validator(mode='before')
    @classmethod
    def convert_confidence(cls, data):
        """Convert string confidence values to floats."""
        if isinstance(data, dict) and "confidence" in data:
            conf = data["confidence"]
            if isinstance(conf, str):
                # Convert common string values to floats
                conf_lower = conf.lower()
                if conf_lower in ["high", "very high"]:
                    data["confidence"] = 0.9
                elif conf_lower in ["medium", "moderate"]:
                    data["confidence"] = 0.6
                elif conf_lower in ["low"]:
                    data["confidence"] = 0.3
                else:
                    # Try to parse as float, or set to None
                    try:
                        data["confidence"] = float(conf)
                    except (ValueError, TypeError):
                        data["confidence"] = None
        return data


class SpeakerSegment(BaseModel):
    """Represents a segment of speech with speaker, text, intent, and context metadata."""
    speaker: str = Field(description="Speaker name")
    text: str = Field(description="Exact text spoken (must be exact substring from transcript)")
    intent: Literal[
        "suggestion",
        "information",
        "question",
        "decision",
        "action_item",
        "agreement",
        "clarification",
    ] = Field(description="Conversational role / intent of the segment")
    resolved_context: str = Field(
        default="",
        description="What earlier topic this refers to, if applicable; empty string if not applicable",
    )
    context_unclear: bool = Field(
        default=False,
        description="True if reference or meaning cannot be resolved from context",
    )
    action_details: Optional[ActionDetails] = Field(
        default_factory=lambda: ActionDetails(),
        description="Action details. Populated only for action_item intent, otherwise all fields are null.",
    )
    
    @model_validator(mode='before')
    @classmethod
    def handle_null_action_details(cls, data):
        """Convert null action_details to empty ActionDetails object."""
        if isinstance(data, dict) and data.get("action_details") is None:
            data["action_details"] = {"description": None, "assignee": None, "deadline": None, "confidence": None}
        return data


class TranscriptSegments(BaseModel):
    """Container for a list of speaker segments (required by LangChain structured output)."""
    segments: List[SpeakerSegment] = Field(description="List of speaker segments in transcript order")


class TranscriptProcessor:
    """Processes meeting transcripts and extracts structured information."""
    
    def __init__(self):
        """Initialize the processor with LangChain model."""
        logger.info("Initializing model: %s at %s", config.MODEL_NAME, config.GLM_API_URL)
        callbacks = [StreamingStderrCallbackHandler()]
        # Build extra_body for Ollama-specific parameters
        # These are passed directly to Ollama API request body
        extra_body = {
            "top_p": config.TOP_P,
            "repeat_penalty": config.REPEAT_PENALTY,
            "presence_penalty": config.PRESENCE_PENALTY,
        }
        # Initialize ChatOpenAI with custom base_url for local GLM4.7 model
        self.llm = ChatOpenAI(
            base_url=config.GLM_API_URL,
            api_key=config.GLM_API_KEY or "not-needed",
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            streaming=False,
            callbacks=callbacks,
            extra_body=extra_body,
        )
        
        # Bind structured output to the model (must be a single Pydantic model, not List[...])
        self.structured_llm = self.llm.with_structured_output(TranscriptSegments, method="json_mode")
        logger.info("Model ready.")
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meeting transcript analyzer. Extract only work-relevant segments and classify them.

KEEP segments about: tasks, decisions, plans, timelines, ownership, risks, technical/business discussion
DISCARD: greetings, small talk, jokes, filler, ASR noise

INTENT types: suggestion | information | question | decision | action_item | agreement | clarification

action_item — use this whenever someone:
  - assigns or self-assigns a task ("I'll send the report", "can you update the doc")
  - proposes a concrete next step with implied ownership ("let's send an email about payment")
  - acknowledges responsibility for a follow-up ("I'll check on that")
  When in doubt between suggestion and action_item, prefer action_item if a real task is implied.

For action_item segments, resolve full meaning from context using action_details.

Return JSON:
{{
  "segments": [
    {{
      "speaker": "",
      "text": "",
      "intent": "",
      "resolved_context": "",
      "context_unclear": false,
      "action_details": {{
        "description": null,
        "assignee": null,
        "deadline": null,
        "confidence": null
      }}
    }}
  ]
}}

action_details is populated only for action_item. All fields null for other intents.
text must be an exact substring from the transcript.
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
