import json
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.config import Config

# Initialize Embeddings
embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)

def initialize_vector_store():
    """
    Loads initial math knowledge base.
    """
    # Mock data for demonstration - in production, load from data/knowledge_base/
    texts = [
        "The derivative of x^n is nx^(n-1).",
        "Bayes theorem: P(A|B) = (P(B|A) * P(A)) / P(B)",
        "Integration by parts: ∫u dv = uv - ∫v du",
        "Quadratic Formula: x = (-b ± sqrt(b^2 - 4ac)) / 2a"
    ]
    docs = [Document(page_content=t) for t in texts]
    
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=Config.CHROMA_PATH
    )
    return vectorstore

def get_retriever():
    if os.path.exists(Config.CHROMA_PATH):
        vectorstore = Chroma(persist_directory=Config.CHROMA_PATH, embedding_function=embeddings)
    else:
        vectorstore = initialize_vector_store()
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def save_to_memory(problem, solution, verification_status):
    """
    Self-Learning: Saves correct solutions to JSON.
    """
    entry = {
        "problem": problem,
        "solution": solution,
        "verified": verification_status
    }
    
    data = []
    if os.path.exists(Config.MEMORY_FILE):
        with open(Config.MEMORY_FILE, 'r') as f:
            data = json.load(f)
    
    data.append(entry)
    
    with open(Config.MEMORY_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Optional: Add to VectorDB for future RAG retrieval (Self-correction loop)
    # Chroma.add_documents([Document(page_content=f"Problem: {problem}\nSolution: {solution}")])