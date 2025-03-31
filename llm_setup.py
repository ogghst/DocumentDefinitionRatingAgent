import os
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from models import CheckResult

# Load environment variables
load_dotenv()

# Retrieve configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Validate required environment variables
if not OLLAMA_BASE_URL or not OLLAMA_MODEL:
    raise ValueError("Missing OLLAMA_BASE_URL and OLLAMA_MODEL environment variables. Cannot initialize LLM.")

# Initialize LLM components
def initialize_llm():
    """Initialize and return the LLM client"""
    
    try:

        if DEEPSEEK_API_KEY and DEEPSEEK_API_BASE:
        
            from langchain_deepseek import ChatDeepSeek
            print(f"Using DeepSeek LLM: base_url='{DEEPSEEK_API_BASE}'")
            llm = ChatDeepSeek(
                model="deepseek-chat",
                temperature=0.3,
                max_tokens=512,
                timeout=30,
                max_retries=2,
                streaming=True,
                base_url=DEEPSEEK_API_BASE,
                api_key=DEEPSEEK_API_KEY
            )

        elif OLLAMA_MODEL and OLLAMA_BASE_URL:
            print(f"Using Ollama LLM: model='{OLLAMA_MODEL}', base_url='{OLLAMA_BASE_URL}'")
            llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                max_tokens=512,
                temperature=0.3,
                max_predict=1024,
                # Add a reasonable timeout
                timeout=30,
                streaming=True
            )

        else:
            raise ValueError("Missing DEEPSEEK_API_BASE or OLLAMA_BASE_URL environment variables. Cannot initialize LLM.")
        
        # Test the connection
        print("Testing LLM connection with simple prompt...")
        test_response = llm.invoke("Hello!")
        print(f"LLM test response received: {str(test_response)[:50]}...")
        
        print("LLM client initialized successfully.")
        return llm
    except Exception as e:
        print(f"ERROR initializing LLM: {e}")
        raise

def initialize_embeddings():
    """Initialize and return the embeddings model"""
    print(f"Using Ollama Embeddings: model='{OLLAMA_MODEL}', base_url='{OLLAMA_BASE_URL}'")
    
    embeddings = OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    print("Ollama Embeddings initialized successfully.")
    return embeddings

# Initialize Pydantic output parser
analysis_parser = PydanticOutputParser(pydantic_object=CheckResult) 