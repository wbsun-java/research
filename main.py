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
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_core.tools import BaseTool
except ImportError as e:
    print(f"FATAL ERROR: Could not import LangChain components: {e}")
    print("Please ensure LangChain is installed correctly (e.g., pip install langchain-core langchain-community langchain).")
    sys.exit(1)

# --- Import Configurations and Utilities ---
try:
    from config import settings  # For API key loading logic
    from utils import input_helpers  # For get_multiline_input
    from prompts.agent_prompts import AGENT_PROMPT_TEMPLATE
except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary modules (settings or utils): {e}")
    print("Ensure 'config/settings.py', 'utils/input_helpers.py', and 'prompts/agent_prompts.py' exist with their __init__.py files.")
    sys.exit(1)


# --- Helper Functions ---
def _display_model_menu(model_keys: list):
    """Prints a formatted menu of available models, grouped by provider."""
    current_provider_group = None
    for i, key in enumerate(model_keys):
        provider_name = key.split('/')[0]
        if provider_name != current_provider_group:
            if current_provider_group is not None:
                print("---")
            current_provider_group = provider_name
            print(f"--- {provider_name.upper()} ---")
        print(f"{i + 1}. {key}")


# --- Custom DuckDuckGo Search Tool ---
class DuckDuckGoSearchResults(BaseTool):
    """Custom tool for DuckDuckGo search using the ddgs package."""
    
    name: str = "duckduckgo_results_json"
    description: str = "A tool that searches DuckDuckGo for results and returns them as a JSON array."
    
    def _run(self, query: str) -> str:
        """Execute the search and return results."""
        try:
            # The `ddg` function from older versions of `duckduckgo-search` is deprecated.
            # We now use the synchronous `DDGS` class.
            from ddgs.ddgs_sync import DDGS

            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=10)

            # Format results as JSON string
            import json
            return json.dumps(results) if results else "[]"
        except Exception as e:
            return f"Error performing search: {str(e)}"


# --- Model Initialization Wrapper ---
def initialize_all_ai_models():
    """
    Initializes AI models by calling the main initializer function from model_loader.py.
    This function now passes configuration directly, decoupling the loader.
    """
    print("\n--- Initializing Models ---")

    try:
        # Pass configuration directly to the decoupled model loader
        initialized_models, init_errors = model_loader.initialize_models(
            api_keys=settings.API_KEYS,
            api_key_arg_names=settings.API_KEY_ARG_NAMES
        )
        print("--- Model Initialization Complete ---")
        return initialized_models, init_errors
    except Exception as e:
        error_msg = f"An unexpected error occurred while trying to load models: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {}, {"model_loader.py_execution": error_msg}


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

    # --- Initialize Tools ---
    # This is done once, as the tool is independent of the model.
    search_tool = DuckDuckGoSearchResults(name="duckduckgo_results_json")
    tools = [search_tool]
    print("\n--- Tools Initialized ---")
    print("✅ DuckDuckGo Search is ready.")

    # --- Create Agent Prompt Template from imported string ---
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_PROMPT_TEMPLATE),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    available_model_keys = sorted(list(initialized_models.keys()))

    print(f"\n--- Successfully Initialized Models ({len(available_model_keys)} available) ---")
    _display_model_menu(available_model_keys)

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
            _display_model_menu(available_model_keys)
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

        # Agent Session Loop with the selected model
        if selected_key:
            # --- Create Agent and Executor for the selected model ---
            llm = initialized_models[selected_key]
            agent = create_tool_calling_agent(llm, tools, agent_prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            print(f"Agent created for model: {selected_key}. Ready for questions.")

            # --- Initialize Chat History for the new session ---
            chat_history: list[BaseMessage] = []

            # Q&A session loop for the current model
            while True:  # Q&A session loop for the current model
                print("\n----------------------------------------")
                # Simplified question prompt
                question = input_helpers.get_multiline_input(
                    f"Ask a question to {selected_key}:")

                if not question.strip():
                    print("Question cannot be empty.")
                    continue

                print(f"\nProcessing your question with agent ({selected_key})...")
                try:
                    response = agent_executor.invoke({
                        "input": question,
                        "chat_history": chat_history
                    })
                    summary_answer = response.get("output", "Agent did not return an answer.")
                    # Add interaction to history for conversational context
                    chat_history.extend([HumanMessage(content=question), response["output"]])
                except Exception as e:
                    summary_answer = f"An error occurred while running the agent: {e}"

                print(f"\n--- Answer from {selected_key} ---")
                print(summary_answer)
                print("----------------------------------------")

                # Post-answer menu
                action_prompt = (
                    "Options:\n"
                    "1. Ask a follow-up (continue this conversation)\n"
                    "2. Start a new topic (clears conversation history)\n"
                    "3. Change model\n"
                    "4. Exit program\n"
                    "Enter choice (1/2/3/4): "
                )
                user_action_valid = False
                while not user_action_valid:  # Loop for valid action choice
                    user_action = input(action_prompt).strip()
                    if user_action == '1':  # Ask a follow-up
                        user_action_valid = True
                        # No break needed here, outer Q&A loop will continue
                    elif user_action == '2':  # Start a new topic
                        print("\nClearing conversation history for a new topic.")
                        chat_history.clear()
                        user_action_valid = True
                        # No break needed, loop will continue with empty history
                    elif user_action == '3':  # Change model
                        print(f"Ending session with {selected_key}.")
                        user_action_valid = True
                        # This break will exit the Q&A session loop,
                        # and the outer model selection loop will restart.
                    elif user_action == '4':  # Exit program
                        print("Exiting program.")
                        sys.exit(0)
                    else:
                        print("Invalid choice. Please enter 1, 2, 3, or 4.")

                if user_action == '3':  # If "Change model" was chosen
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
