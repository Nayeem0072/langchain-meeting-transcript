"""Configuration settings for the transcript processor."""
import os
from dotenv import load_dotenv

load_dotenv()

# Model endpoint configuration
GLM_API_URL = os.getenv("GLM_API_URL", "http://localhost:8000/v1")
GLM_API_KEY = os.getenv("GLM_API_KEY", None)

# Model parameters
MODEL_NAME = os.getenv("MODEL_NAME", "glm4-7")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
TOP_P = float(os.getenv("TOP_P", "0.15"))
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.2"))
PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", "0.6"))
