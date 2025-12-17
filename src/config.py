import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-4o" # Multimodal model for Vision + Logic
    EMBEDDING_MODEL = "text-embedding-3-small"
    MEMORY_FILE = "data/problem_memory.json"
    CHROMA_PATH = "data/chroma_db"