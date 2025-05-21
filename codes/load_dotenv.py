import os
from dotenv import load_dotenv
import sys

def load_env_config():
    """
    Load environment variables from .env file.
    This function should be imported and called at the beginning of each script
    that needs access to environment variables.
    
    Returns:
        bool: True if .env file was loaded successfully, False otherwise
    """
    # Get the project root directory (assumes this file is in the codes/ directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Path to the .env file
    env_path = os.path.join(project_root, '.env')
    
    # Load environment variables from .env file
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
        return True
    else:
        print(f"Warning: .env file not found at {env_path}", file=sys.stderr)
        return False

if __name__ == "__main__":
    # If run directly, just load the environment variables
    load_env_config()
    
    # Print all environment variables (for debugging)
    env_vars = [
        "OPENAI_API_KEY", 
        "GPT_VERSION",
        "PAPER_NAME",
        "PDF_PATH",
        "PDF_JSON_PATH", 
        "PDF_JSON_CLEANED_PATH",
        "PDF_LATEX_PATH",
        "OUTPUT_DIR", 
        "OUTPUT_REPO_DIR"
    ]
    
    # LiteLLM environment variables
    litellm_vars = [
        "AWS_REGION",
        "ANTHROPIC_MODEL",
        "DISABLE_PROMPT_CACHING",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE"
    ]
    
    for var in env_vars + litellm_vars:
        # Mask API keys for security
        if var in ["OPENAI_API_KEY"] and os.environ.get(var):
            print(f"{var}=***********{os.environ.get(var)[-4:]}")
        else:
            print(f"{var}={os.environ.get(var, 'Not set')}")