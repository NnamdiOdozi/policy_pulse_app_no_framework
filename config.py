import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configurations
PINECONE_INDEX_NAME = "policypulse"
SONAR_MODEL_NAME = "sonar-3"
TOP_K_DEFAULT = 5 # So we can return the top 5 results from Pinecone DB 
RELEVANCE_THRESHOLD = 0.75

# API endpoints (non-sensitive)
SONAR_API_URL = "https://api.sonar.com/v1/chat/completions"

# Retrieve sensitive values from environment
def get_pinecone_api_key():
    return os.environ.get("PINECONE_API_KEY")

def get_sonar_api_key():
    return os.environ.get("SONAR_API_KEY")

def get_hf_token():
    return os.environ.get("HF_TOKEN")

def get_openai_api_key():
    return os.environ.get("OPENAI_API_KEY")