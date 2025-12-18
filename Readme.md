# 🧮 Reliable Multimodal Math Mentor

**Multimodal Math Mentor** is an advanced agentic AI application designed to act as a reliable math tutor. Unlike standard chatbots that often hallucinate on arithmetic, this system uses a **Graph-based Agentic Workflow (LangGraph)** to separate logic into specialized roles.

It combines **Python Code Execution** for precise calculations, **RAG (Retrieval-Augmented Generation)** for conceptual grounding, and **Multimodal Inputs** (Vision & Audio) to solve problems accurately.

---

## 🚀 Key Features

### 1. 🧠 Agentic State Machine (LangGraph)
The system is built on a Directed Acyclic Graph (DAG) where specialized agents collaborate:
*   **Parser Agent:** Structurally analyzes the input to determine if clarification is needed.
*   **Router Agent:** Intelligently routes problems:
    *   **Calculation Path:** Uses a **Python Solver** to write and execute code, ensuring arithmetic accuracy (e.g., integrals, matrix multiplication).
    *   **Conceptual Path:** Uses a **RAG Solver** to retrieve axioms and definitions from the vector database.
*   **Verifier Agent:** rigorous check stage that reviews the answer for domain errors (e.g., division by zero, unit mismatches) before showing it to the user.
*   **Explainer Agent:** Translates the technical output into a clear, educational Markdown explanation.

### 2. 👁️🎙️ Multimodal Interaction
*   **Visual Problem Solving:** Upload images of handwritten or printed math. The system uses **EasyOCR** with confidence checking to digitize the problem.
*   **Voice Mode:** Ask questions verbally. Uses **OpenAI GPT-4o-Transcribe** for high-fidelity speech-to-text.
*   **Interactive Verification:** Users can review and edit extracted text *before* the agents start solving.

### 3. 📚 Knowledge Base & RAG
*   **Dynamic Ingestion:** Upload PDF textbooks or TXT notes via the sidebar.
*   **Vector Search:** Powered by **ChromaDB** and **OpenAI Embeddings**.
*   **Seed Knowledge:** Comes pre-loaded with critical math rules (e.g., Derivative Power Rules, Euler's Identity).

### 4. ⚙️ Robust Configuration
*   **Dynamic API Key Management:** Securely input your OpenAI Key in the UI (Session-based).
*   **Memory:** "Helpful" solutions are saved to a local JSON file to reinforce learning over time.

---

## 📐 System Architecture

The following Mermaid diagram illustrates the exact `LangGraph` workflow defined in `src/graph.py`:

g```mermaid
graph TD
    Start[User Input] --> Processor[OCR / Audio Transcribe]
    Processor --> Parser[Parser Agent]
    Parser --> Router{Router Decision}
    
    Router -- "Calculation" --> Python[Python Code Solver]
    Router -- "Conceptual" --> RAG[RAG Solver]
    
    Python --> Verifier[Verifier Agent]
    RAG --> Verifier
    
    Verifier -- "Correct" --> Explainer[Explainer Agent]
    Verifier -- "Incorrect" --> End[Stop & Error]
    
    Explainer --> UI[Display Solution]
```

---

## 📂 Project Structure

```text
multimodal-math-mentor/
├── data/                  
│   ├── chroma_db/          # Persistent Vector Database
│   └── problem_memory.json # Saved verified solutions
├── src/                   
│   ├── agents.py           # Agent definitions (Parser, Router, Solvers, Verifier)
│   ├── config.py           # Configuration & Environment handling
│   ├── graph.py            # LangGraph StateGraph construction
│   ├── processors.py       # OCR (EasyOCR) & Audio (gpt-4o-transcribe/GPT-4o) logic
│   └── rag.py              # ChromaDB setup and Document Ingestion
├── main.py                 # Streamlit Frontend application
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
*   Python 3.10+
*   An [OpenAI API Key](https://platform.openai.com/) (Requires access to `gpt-4o`).

### 2. Clone Repository
```bash
git clone https://github.com/Vamshi17one/multimodal-math-mentor.git
cd multimodal-math-mentor
```

### 3. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
streamlit run main.py
```

---

## 📖 Usage Guide

1.  **Settings (Sidebar):**
    *   Enter your **OpenAI API Key** and click "Apply Key".
    *   (Optional) Upload reference PDFs and click "Index Documents" to train the RAG system.
2.  **Select Input Mode:**
    *   **Text:** Type the math problem directly.
    *   **Image:** Upload a `.jpg` or `.png`. Click "Extract Text" to run OCR.
    *   **Audio:** Upload an audio file. Click "Transcribe".
3.  **Verify:**
    *   The system displays the extracted text. Edit it if necessary to correct OCR errors.
4.  **Solve:**
    *   Click **"Confirm & Solve Problem"**.
    *   Watch the status indicators as agents process the request.
5.  **Review:**
    *   Expand **"🕵️ Agent Logic & Tools"** to see the Python code executed or documents retrieved.

---

## 📦 Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/) & [LangChain](https://www.langchain.com/)
*   **LLM:** OpenAI GPT-4o
*   **Vector DB:** ChromaDB
*   **OCR:** EasyOCR (with OpenCV/NumPy)
*   **Audio:** OpenAI gpt-4o-transcribe / GPT-4o-Audio

---

## ⚠️ Limitations

*   **Python Sandboxing:** The solver utilizes `exec()` to run generated code. While this provides powerful calculation capabilities, it is designed for local demonstration purposes.
*   **OCR Accuracy:** Handwriting recognition depends on image contrast and clarity. Complex matrix notation may sometimes require manual correction in the verification stage.