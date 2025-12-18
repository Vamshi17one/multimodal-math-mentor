import json
import os
import shutil
import tempfile
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from src.config import Config

# Initialize Embeddings
embeddings = OpenAIEmbeddings(
    model=Config.EMBEDDING_MODEL,
    
    dimensions=1536 
)


SEED_KNOWLEDGE = [
    
    "Derivative Power Rule: d/dx(x^n) = nx^(n-1)",
    "Integration Power Rule: ∫x^n dx = (x^(n+1))/(n+1) + C, for n ≠ -1",
    "Logarithm Rule: ln(a*b) = ln(a) + ln(b)",
    "Euler's Identity: e^(i*pi) + 1 = 0",
    "Quadratic Formula: x = (-b ± sqrt(b^2 - 4ac)) / 2a",
    "Pythagorean Identity: sin^2(x) + cos^2(x) = 1",
    "Sum of arithmetic series: S_n = n/2 * (2a + (n-1)d)",
    
    
    "Mistake: (a+b)^2 ≠ a^2 + b^2. Correct: a^2 + 2ab + b^2",
    "Mistake: sqrt(x^2 + y^2) ≠ x + y. It cannot be simplified further.",
    "Pitfall: Dividing by zero is undefined. Always check denominators.",
    "Pitfall: Forgetting +C in indefinite integrals.",
    "Constraint: The domain of ln(x) is x > 0.",
    "Constraint: The range of sin(x) and cos(x) is [-1, 1] for real inputs.",
    
    
    "Template: To find local maxima/minima, take f'(x), set to 0, solve for x, then check f''(x).",
    "Template: For limits at infinity with polynomials, divide numerator and denominator by the highest power of x."
]

def initialize_vector_store():
    """
    Initializes ChromaDB. If empty, loads curated seed data.
    """
    
    if os.path.exists(Config.CHROMA_PATH) and os.listdir(Config.CHROMA_PATH):
        return Chroma(persist_directory=Config.CHROMA_PATH, embedding_function=embeddings)
    
    
    print("--- Initializing Vector Store with Seed Data ---")
    docs = [
        Document(page_content=text, metadata={"source": "Curated_Seed_Data"}) 
        for text in SEED_KNOWLEDGE
    ]
    
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=Config.CHROMA_PATH
    )
    return vectorstore

def get_retriever():
    vectorstore = initialize_vector_store()
    # Fetch k=5 to get a good mix of sources
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def process_and_index_files(uploaded_files: List):
    """
    Ingests user uploaded files (PDF/TXT) into the vector store.
    Strictly follows batch size of 5.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    documents = []
    
    # 1. Process Files into Documents
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            temp_path = os.path.join(temp_dir, file.name)
            
            
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            
            loader = None
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(temp_path)
            
            if loader:
                raw_docs = loader.load()
                
                for doc in raw_docs:
                    doc.metadata["source"] = file.name
                documents.extend(raw_docs)

    if not documents:
        return "No valid text extracted from files."

    
    chunked_docs = text_splitter.split_documents(documents)
    
    # 3. Batch Indexing (Batch Size = 5)
    vectorstore = initialize_vector_store()
    batch_size = Config.RAG_BATCH_SIZE
    total_chunks = len(chunked_docs)
    
    print(f"--- Indexing {total_chunks} chunks in batches of {batch_size} ---")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunked_docs[i : i + batch_size]
        
        vectorstore.add_documents(batch)
        print(f"Processed batch {i//batch_size + 1}")

    return f"Successfully indexed {len(uploaded_files)} files ({total_chunks} chunks)."

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
        try:
            with open(Config.MEMORY_FILE, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    
    data.append(entry)
    
    
    os.makedirs(os.path.dirname(Config.MEMORY_FILE), exist_ok=True)
    
    with open(Config.MEMORY_FILE, 'w') as f:
        json.dump(data, f, indent=2)