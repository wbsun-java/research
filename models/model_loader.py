# Import necessary standard LangChain classes
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_xai.chat_models import ChatXAI

# --- MODEL_DEFINITIONS (with model_id_key) ---
MODEL_DEFINITIONS = {
    "openai": [
        {"key": "openai/gpt-4.1-mini", "model_name": "gpt-4.1-mini", "class": ChatOpenAI, "args": {"temperature": 0.7},
         "model_id_key": "model_name"},
        {"key": "openai/gpt-4.5-preview-2025-02-27", "model_name": "gpt-4.5-preview-2025-02-27", "class": ChatOpenAI,
         "args": {"temperature": 0.7}, "model_id_key": "model_name"},
        {"key": "openai/gpt-4o-mini", "model_name": "gpt-4o-mini", "class": ChatOpenAI,
         "args": {"temperature": 0.7}, "model_id_key": "model_name"},
        {"key": "openai/gpt-4.1", "model_name": "gpt-4.1", "class": ChatOpenAI,
         "args": {"temperature": 0.7}, "model_id_key": "model_name"},
    ],
    "groq": [
        {"key": "groq/meta-llama/llama-guard-4-12b", "model_name": "meta-llama/llama-guard-4-12b", "class": ChatGroq,
         "args": {"temperature": 0.7}, "model_id_key": "model_name"},
        {"key": "groq/meta-llama/llama-4-scout-17b-16e-instruct",
         "model_name": "meta-llama/llama-4-scout-17b-16e-instruct", "class": ChatGroq, "args": {"temperature": 0.7},
         "model_id_key": "model_name"},
        {"key": "groq/deepseek-r1-distill-llama-70b", "model_name": "deepseek-r1-distill-llama-70b", "class": ChatGroq,
         "args": {"temperature": 0.7}, "model_id_key": "model_name"},
        {"key": "groq/llama-4-maverick-17b", "model_name": "meta-llama/llama-4-maverick-17b-128e-instruct",
         "class": ChatGroq, "args": {"temperature": 0.7}, "model_id_key": "model_name"},
        {"key": "groq/llama-3.3-70b-versatile", "model_name": "llama-3.3-70b-versatile", "class": ChatGroq,
         "args": {"temperature": 0.7}, "model_id_key": "model_name"},
        {"key": "groq/qwen-qwq-32b", "model_name": "qwen-qwq-32b", "class": ChatGroq, "args": {"temperature": 0.7},
         "model_id_key": "model_name"},
    ],
    "google": [
        {"key": "google/gemini-2.5-flash-preview-04-17", "model_name": "gemini-2.5-flash-preview-04-17",
         "class": ChatGoogleGenerativeAI, "args": {"temperature": 0.7}, "model_id_key": "model"},
        {"key": "google/gemini-2.5-pro", "model_name": "gemini-2.5-pro",
         "class": ChatGoogleGenerativeAI, "args": {"temperature": 0.7}, "model_id_key": "model"},
        {"key": "google/gemma-3-12b-it", "model_name": "gemma-3-12b-it", "class": ChatGoogleGenerativeAI,
         "args": {"temperature": 0.7}, "model_id_key": "model"},
        {"key": "google/gemini-2.5-flash", "model_name": "gemini-2.5-flash", "class": ChatGoogleGenerativeAI,
         "args": {"temperature": 0.7}, "model_id_key": "model"},
    ],
    "anthropic": [
        {"key": "anthropic/claude-3-5-sonnet", "model_name": "claude-3-5-sonnet-20241022", "class": ChatAnthropic,
         "args": {"temperature": 0.7}, "model_id_key": "model_name"},
        {"key": "anthropic/claude-3.7-sonnet", "model_name": "claude-3-7-sonnet-20250219", "class": ChatAnthropic,
         "args": {"temperature": 0.7}, "model_id_key": "model_name"},
        {"key": "anthropic/Claude Sonnet 4", "model_name": "claude-sonnet-4-20250514", "class": ChatAnthropic,
         "args": {"temperature": 0.7}, "model_id_key": "model_name"},
        {"key": "anthropic/Claude Opus 4", "model_name": "claude-opus-4-20250514", "class": ChatAnthropic,
         "args": {"temperature": 0.7}, "model_id_key": "model_name"}
    ],
    "xai": [
        {"key": "xAI/Grok 3", "model_name": "grok-3", "class": ChatXAI,
         "args": {"temperature": 0.7}, "model_id_key": "model"},
        {"key": "xAI/grok 3 mini", "model_name": "grok-3-mini", "class": ChatXAI,
         "args": {"temperature": 0.7}, "model_id_key": "model"},
        {"key": "xAI/Grok 3 mini fast", "model_name": "grok-3-mini-fast", "class": ChatXAI,
         "args": {"temperature": 0.7}, "model_id_key": "model"},
        {"key": "xAI/Grok 4", "model_name": "grok-4-0407", "class": ChatXAI,
         "args": {"temperature": 0.7}, "model_id_key": "model"},
    ],
}


# --- initialize_models Function (MODIFIED) ---
def initialize_models(api_keys: dict, api_key_arg_names: dict):
    """
    Initializes all models defined in MODEL_DEFINITIONS based on available API keys.
    This is a pure function that receives configuration and returns data without
    side effects like printing.

    Args:
        api_keys (dict): A dictionary mapping provider names to their API keys.
        api_key_arg_names (dict): A dictionary mapping provider names to the
                                  constructor argument name for the API key.

    Returns:
        tuple: (initialized_models dict, initialization_errors dict)
    """
    initialized_models = {}
    initialization_errors = {}

    for provider, model_list in MODEL_DEFINITIONS.items():
        api_key = api_keys.get(provider)
        api_key_arg_name = api_key_arg_names.get(provider)

        if not api_key_arg_name:
            # Log errors for all models under this unconfigured provider
            for model_def in model_list:
                initialization_errors[model_def["key"]] = f"Provider '{provider}' not configured in API_KEY_ARG_NAMES."
            continue

        if not api_key:
            for model_def in model_list:
                initialization_errors[model_def["key"]] = f"{provider.capitalize()} API Key not found."
            continue

        for model_def in model_list:
            model_key = model_def["key"]
            model_identifier_value = model_def["model_name"]
            model_class = model_def["class"]
            constructor_id_param_name = model_def.get("model_id_key")

            if not constructor_id_param_name:
                error_msg = f"Configuration error: 'model_id_key' missing for {model_key} in MODEL_DEFINITIONS."
                initialization_errors[model_key] = error_msg
                continue

            model_args_from_def = model_def["args"].copy()
            model_args_from_def[api_key_arg_name] = api_key

            try:
                constructor_kwargs = {constructor_id_param_name: model_identifier_value}
                final_constructor_args = {**constructor_kwargs, **model_args_from_def}

                initialized_models[model_key] = model_class(**final_constructor_args)

            except Exception as e:
                error_msg = f"Failed to initialize {model_key}: {e}"
                initialization_errors[model_key] = error_msg

    return initialized_models, initialization_errors