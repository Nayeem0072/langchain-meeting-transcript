"""Configuration settings for the transcript processor."""
import os
from dotenv import load_dotenv

load_dotenv()

# Model endpoint configuration
GLM_API_URL = os.getenv("GLM_API_URL", "http://localhost:8000/v1")
GLM_API_KEY = os.getenv("GLM_API_KEY", None)

# Model parameters
MODEL_NAME = os.getenv("MODEL_NAME", "glm4-7")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
