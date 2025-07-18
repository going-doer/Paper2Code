import os
import litellm
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer
from load_dotenv import load_env_config

def initialize_litellm():
    """
    Initialize LiteLLM with configuration from environment variables.
    Must be called after load_env_config().
    """
    # Load environment variables if not already loaded
    load_env_config()
    
    try:
        # Check for different provider configurations in environment
        if os.environ.get("AWS_REGION_NAME") or (os.environ.get("AWS_REGION") and os.environ.get("BEDROCK_MODEL")):
            # AWS Bedrock configuration detected
            try:
                import boto3
            except ImportError:
                print("WARNING: boto3 is required for AWS Bedrock but is not installed.")
                print("Please install it with 'pip install boto3'")
                return None
            
            # Set region from either standard variable or our custom one
            if os.environ.get("AWS_REGION") and not os.environ.get("AWS_REGION_NAME"):
                os.environ["AWS_REGION_NAME"] = os.environ.get("AWS_REGION")
            
            # Get model name
            model_name = os.environ.get("BEDROCK_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
            bedrock_model = f"bedrock/{model_name}"
            
            # Set AWS credentials paths if provided
            if os.environ.get("AWS_SHARED_CREDENTIALS_FILE"):
                os.environ["AWS_SHARED_CREDENTIALS_FILE"] = os.path.expanduser(os.environ.get("AWS_SHARED_CREDENTIALS_FILE"))
            
            if os.environ.get("AWS_CONFIG_FILE"):
                os.environ["AWS_CONFIG_FILE"] = os.path.expanduser(os.environ.get("AWS_CONFIG_FILE"))
            
            # Configure caching
            if os.environ.get("DISABLE_PROMPT_CACHING") == "1":
                litellm.cache = None
            
            print(f"LiteLLM configured to use AWS Bedrock with model: {bedrock_model}")
            return bedrock_model
            
        elif os.environ.get("OPENAI_API_KEY"):
            # OpenAI configuration detected
            model_name = os.environ.get("OPENAI_MODEL", "o3-mini")
            openai_model = f"openai/{model_name}"
            print(f"LiteLLM configured to use OpenAI with model: {openai_model}")
            return openai_model
            
        elif os.environ.get("ANTHROPIC_API_KEY"):
            # Anthropic configuration detected
            model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
            anthropic_model = f"anthropic/{model_name}"
            print(f"LiteLLM configured to use Anthropic with model: {anthropic_model}")
            return anthropic_model
            
        else:
            # No LiteLLM provider configuration detected
            print("Using model specified via command line args (no LiteLLM configuration detected)")
            return None
    except Exception as e:
        print(f"WARNING: Failed to initialize LiteLLM: {str(e)}")
        print("Falling back to vLLM implementation")
        return None

def get_llm_client(model_name: str, tp_size: int = 2, max_model_len: int = 128000, temperature: float = 1.0):
    """
    Get an LLM client based on the environment configuration.
    
    Args:
        model_name: The model to use (may be overridden by env vars)
        tp_size: tensor parallel size for vLLM
        max_model_len: maximum model length for vLLM
        temperature: sampling temperature
        
    Returns:
        A tuple of (client, tokenizer, is_litellm, sampling_params)
    """
    # Try to initialize LiteLLM first
    litellm_model = initialize_litellm()
    
    # If LiteLLM is configured, use it
    if litellm_model:
        tokenizer = None  # LiteLLM handles tokenization
        return litellm_model, tokenizer, True, {"temperature": temperature}
    
    # Otherwise use vLLM implementation
    try:
        from vllm import LLM, SamplingParams
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if "Qwen" in model_name:
            llm = LLM(model=model_name, 
                    tensor_parallel_size=tp_size, 
                    max_model_len=max_model_len,
                    gpu_memory_utilization=0.95,
                    trust_remote_code=True, enforce_eager=True, 
                    rope_scaling={"factor": 4.0, "original_max_position_embeddings": 32768, "type": "yarn"})
            sampling_params = SamplingParams(temperature=temperature, max_tokens=131072)
        
        elif "deepseek" in model_name:
            llm = LLM(model=model_name, 
                    tensor_parallel_size=tp_size, 
                    max_model_len=max_model_len,
                    gpu_memory_utilization=0.95,
                    trust_remote_code=True, enforce_eager=True)
            sampling_params = SamplingParams(temperature=temperature, max_tokens=128000, stop_token_ids=[tokenizer.eos_token_id])
        
        else:
            # Generic configuration for other models
            llm = LLM(model=model_name,
                    tensor_parallel_size=tp_size,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=0.95,
                    trust_remote_code=True)
            sampling_params = SamplingParams(temperature=temperature, max_tokens=128000)
        
        return llm, tokenizer, False, sampling_params
    except ImportError:
        raise ImportError("vLLM is not installed. Please install it with 'pip install vllm' to use this mode")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize vLLM: {str(e)}")

def run_inference(client, tokenizer, is_litellm, sampling_params, messages):
    """
    Run inference using either LiteLLM or vLLM.
    
    Args:
        client: The LLM client (either LiteLLM model name or vLLM instance)
        tokenizer: Tokenizer for vLLM (None for LiteLLM)
        is_litellm: Whether we're using LiteLLM
        sampling_params: Parameters for sampling
        messages: The messages to process
        
    Returns:
        Generated completion text
    """
    if is_litellm:
        # Use LiteLLM for inference with exponential backoff retry (prevents stopping with service bottlenecks)
        import time
        max_retries = 10
        retry_delay = 1  # start with 1 second
        
        for attempt in range(max_retries):
            try:
                response = litellm.completion(
                    model=client,
                    messages=messages,
                    temperature=sampling_params.get("temperature", 1.0)
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Double the wait time
                else:
                    print(f"All {max_retries} attempts failed")
                    raise RuntimeError(f"LiteLLM failed after {max_retries} attempts: {str(e)}")
    else:
        # Use vLLM for inference
        prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True)]
        outputs = client.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        completion = [output.outputs[0].text for output in outputs]
        return completion[0]