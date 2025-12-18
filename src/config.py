import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Remove static definition to allow dynamic updates
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
    
    @staticmethod
    def get_openai_key():
        # Check environment variable dynamically (updated by main.py)
        return os.getenv("OPENAI_API_KEY")

    MODEL_NAME = "gpt-4o" 
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    RAG_BATCH_SIZE = 5 
    MEMORY_FILE = "data/problem_memory.json"
    CHROMA_PATH = "data/chroma_db"