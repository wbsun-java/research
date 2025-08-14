import sys
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Import Custom Model Loader ---
try:
    from models import model_loader  # Imports models/model_loader.py
except ImportError as e:
    print(f"FATAL ERROR: Could not import 'model_loader.py' from the 'models' directory: {e}")
    print("Make sure 'models/__init__.py' and 'models/model_loader.py' exist.")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred while importing 'model_loader.py': {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Import LangChain Components ---
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"FATAL ERROR: Could not import LangChain core components: {e}")
    print("Please ensure LangChain is installed correctly (e.g., pip install langchain-core).")
    sys.exit(1)

# --- Import Configurations and Utilities ---
try:
    from config import settings  # For API key loading logic
    from utils import input_helpers  # For get_multiline_input
    # We might not need specific task prompts from 'prompts' module anymore,
    # but keeping it for now in case settings.TASKS is referenced indirectly.
except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary modules (settings or utils): {e}")
    print("Ensure 'config/settings.py' and 'utils/input_helpers.py' exist with their __init__.py files.")
    sys.exit(1)

# --- New Prompt for Q&A Summarization ---
QA_SUMMARY_PROMPT_TEMPLATE = """
You are a helpful research assistant.
Please answer the following question based on your knowledge.
Provide only a concise summary of your answer. Do not include your thinking process, intermediate steps, or any conversational fluff.

Question:
{user_question}

Concise Summary of Answer:
"""


# --- Model Initialization Wrapper (largely unchanged) ---
def initialize_all_ai_models():
    """
    Initializes AI models by calling the main initializer function from model_loader.py.
    """
    print("Attempting to initialize AI models via models/model_loader.py...")
    initialized_models_dict = {}
    initialization_errors_dict = {}

    try:
        if hasattr(model_loader, 'initialize_models'):
            initialized_models_dict, initialization_errors_dict = model_loader.initialize_models()
        else:
            error_msg = "ERROR: 'models/model_loader.py' does not have a recognized model initialization function (e.g., 'initialize_models')."
            print(error_msg)
            initialization_errors_dict["model_loader.py_interface"] = error_msg
            return {}, initialization_errors_dict

        if not initialized_models_dict and not initialization_errors_dict:
            print("Warning: model_loader.py's initialization function returned no models and no errors. "
                  "Check model_loader.py implementation and API key availability in .env (loaded by config/settings.py).")
            initialization_errors_dict[
                "model_loader_empty_return"] = "No models or specific errors returned from model_loader.py."
        elif not initialized_models_dict:
            print("No models were successfully initialized by model_loader.py.")

    except AttributeError as e:
        error_msg = f"ERROR: 'models/model_loader.py' seems to be missing an expected attribute or function: {e}"
        print(error_msg)
        initialization_errors_dict["model_loader.py_interface_attr"] = error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while trying to load models from model_loader.py: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        initialization_errors_dict["model_loader.py_execution"] = error_msg

    return initialized_models_dict, initialization_errors_dict


# --- Model Execution for Q&A ---
def get_summarized_answer(model_key, question, models_dict):
    """
    Gets a summarized answer from the selected model for the given question.
    """
    print(f"\nAsking model '{model_key}' for a summarized answer...")
    model_instance = models_dict.get(model_key)

    if not model_instance:
        return f"Error: Model '{model_key}' not found in initialized models."

    try:
        output_parser = StrOutputParser()
        prompt_template = ChatPromptTemplate.from_template(QA_SUMMARY_PROMPT_TEMPLATE)
        chain = prompt_template | model_instance | output_parser

        result = chain.invoke({"user_question": question})
        return result

    except Exception as e:
        print(f"ERROR during model execution for {model_key}:")
        traceback.print_exc()
        return f"Error getting answer from model {model_key}: {e}"


# --- Main Application Loop for Research Q&A ---
def run_research_qa_loop():
    """Initializes models and runs the research Q&A loop."""

    initialized_models, init_errors = initialize_all_ai_models()

    if not initialized_models:
        print("\n--- FATAL ERROR ---")
        print("No AI models were successfully initialized.")
        if init_errors:
            print("The following errors occurred during initialization:")
            for model_name_key, error_msg in init_errors.items():
                print(f"- {model_name_key}: {error_msg}")
        else:
            print("No specific errors were reported, but initialization failed.")
        sys.exit(1)

    available_model_keys = sorted(list(initialized_models.keys()))

    print(f"\n--- Successfully Initialized Models ({len(available_model_keys)} available) ---")
    # This initial printing of models can be refactored into a helper if used multiple times
    current_provider_group_init = None
    model_display_number_init = 1
    for key_init in available_model_keys:
        provider_name_init = key_init.split('/')[0]
        if provider_name_init != current_provider_group_init:
            if current_provider_group_init is not None:
                print("---")
            current_provider_group_init = provider_name_init
        print(f"{model_display_number_init}. {key_init}")
        model_display_number_init += 1

    if init_errors:
        print("\n--- Initialization Warnings ---")
        print("Some models may have failed to initialize. They will not be available if not listed above.")
        for model_name_key, error_msg in init_errors.items():
            if model_name_key not in initialized_models:
                print(f"- {model_name_key}: {error_msg}")

    # Outer loop for model selection and Q&A sessions
    while True:
        selected_key = None
        # Model selection part
        while True:  # Model selection inner loop
            print("\n----------------------------------------")
            print("Select a model for your research questions:")
            current_provider_group_select = None
            model_select_number = 1
            for key_option in available_model_keys:
                provider_name_select = key_option.split('/')[0]
                if provider_name_select != current_provider_group_select:
                    if current_provider_group_select is not None:
                        print("---")
                    current_provider_group_select = provider_name_select
                print(f"{model_select_number}. {key_option}")
                model_select_number += 1
            print("0. Exit Program")

            choice = input("Enter model choice number: ").strip()
            try:
                choice_num = int(choice)
                if choice_num == 0:
                    print("Exiting program.")
                    sys.exit(0)  # Exit program directly
                if 1 <= choice_num <= len(available_model_keys):
                    selected_key = available_model_keys[choice_num - 1]
                    print(f"\n>>> You have selected model: {selected_key} <<<")
                    break  # Exit model selection inner loop, proceed to Q&A
                else:
                    print("Invalid model choice number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Q&A Loop with the selected model
        if selected_key:  # Proceed only if a model was selected
            while True:  # Q&A session loop for the current model
                print("\n----------------------------------------")
                # Simplified question prompt
                question = input_helpers.get_multiline_input(
                    f"Ask a question to {selected_key}:")

                if not question.strip():
                    print("Question cannot be empty.")
                    continue

                print(f"\nProcessing your question with {selected_key}...")
                summary_answer = get_summarized_answer(selected_key, question, initialized_models)

                print(f"\n--- Summarized Answer from {selected_key} ---")
                print(summary_answer)
                print("----------------------------------------")

                # Post-answer menu
                action_prompt = (
                    "Options:\n"
                    "1. Ask another question (to this model)\n"
                    "2. Change model\n"
                    "3. Exit program\n"
                    "Enter choice (1/2/3): "
                )
                user_action_valid = False
                while not user_action_valid:  # Loop for valid action choice
                    user_action = input(action_prompt).strip()
                    if user_action == '1':  # Ask another question
                        user_action_valid = True
                        # No break needed here, outer Q&A loop will continue
                    elif user_action == '2':  # Change model
                        print(f"Ending session with {selected_key}.")
                        user_action_valid = True
                        # This break will exit the Q&A session loop,
                        # and the outer model selection loop will restart.
                    elif user_action == '3':  # Exit program
                        print("Exiting program.")
                        sys.exit(0)
                    else:
                        print("Invalid choice. Please enter 1, 2, or 3.")

                if user_action == '2':  # If "Change model" was chosen
                    break  # Break from the Q&A session loop to go back to model selection
        else:
            # This case should ideally not be reached if model selection logic is sound
            print("Error: No model was selected. Returning to model selection.")
            # continue will restart the outer model selection loop


# --- Main Execution Guard ---
if __name__ == "__main__":
    try:
        run_research_qa_loop()
    except SystemExit:
        print("\nProgram terminated.")
    except Exception as e:
        print("\n--- UNHANDLED EXCEPTION IN MAIN LOOP ---")
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        print("\nResearch Q&A tool finished.")