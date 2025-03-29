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
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Validate required environment variables
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set.")
if not DEEPSEEK_API_BASE:
    raise ValueError("DEEPSEEK_API_BASE environment variable not set.")
if not OLLAMA_MODEL:
    raise ValueError("OLLAMA_MODEL environment variable not set (e.g., 'phi4:14b').")

# Initialize LLM components
def initialize_llm():
    """Initialize and return the LLM client"""
    print(f"Using Ollama LLM: model='{OLLAMA_MODEL}', base_url='{OLLAMA_BASE_URL}'")
    
    try:
        # Add a timeout to prevent hanging
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            max_tokens=1024,
            temperature=0.0,
            max_predict=1024,
            # Add a reasonable timeout
            timeout=30,
            streaming=True
        )
        
        # Test the connection
        print("Testing LLM connection with simple prompt...")
        test_response = llm.invoke("Hello!")
        print(f"LLM test response received: {str(test_response)[:50]}...")
        
        print("Ollama LLM client initialized successfully.")
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