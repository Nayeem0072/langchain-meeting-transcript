"""Main entry point for LangGraph action item extraction."""
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Handle both direct execution and module import
try:
    from .langgraph_workflow import extract_actions
except ImportError:
    # If relative import fails, try absolute
    from src.langgraph_workflow import extract_actions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main():
    """CLI entry point for LangGraph action extraction."""
    if len(sys.argv) < 2:
        print("Usage: python langgraph_main.py <input.json> [output.json]", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output_langgraph.json"
    
    try:
        logger.info("Loading input file: %s", input_file)
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "transcript_raw" not in data:
            raise ValueError("Input JSON must contain 'transcript_raw' field")
        
        transcript = data["transcript_raw"]
        logger.info("Input loaded (%d characters). Starting LangGraph extraction.", len(transcript))
        
        # Extract actions
        actions = extract_actions(transcript)
        
        # Write output
        logger.info("Writing output to: %s", output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(actions, f, indent=2, ensure_ascii=False)
        
        logger.info("Done. Extracted %d action(s).", len(actions))
        logger.info("Results saved to %s", output_file)
        
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_file)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in input file: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
