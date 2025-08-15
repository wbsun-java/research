import os

# The .env file is loaded by main.py using load_dotenv()
# So os.getenv will work here.

# --- API Key Mappings ---
API_KEYS = {
    "openai": os.getenv('OPENAI_API_KEY'),
    "groq": os.getenv('GROQ_API_KEY'),
    "google": os.getenv('GOOGLE_API_KEY'),
    "anthropic": os.getenv('ANTHROPIC_API_KEY'),
    "xai": os.getenv('XAI_API_KEY'),
}

API_KEY_ARG_NAMES = {
    "openai": "openai_api_key",
    "groq": "groq_api_key",
    "google": "google_api_key",
    "anthropic": "anthropic_api_key",
    "xai": "api_key",
}

# --- Define Available Tasks ---
TASKS = {
    "1": {"name": "Refine Text (Email, Article, etc.)", "id": "refine"},
    "2": {"name": "Translate: English -> Chinese", "id": "en_to_zh"},
    "3": {"name": "Translate: Chinese -> English", "id": "zh_to_en"},
}