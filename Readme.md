# 🧮 Multimodal Math Mentor (Agentic RAG)

**Multimodal Math Mentor** is an advanced AI-powered tutoring application designed to solve JEE-level math problems. It leverages **LangGraph** for agentic workflows, **RAG (Retrieval-Augmented Generation)** for formula retrieval, and **Multimodal AI** (GPT-4o, Whisper) to process text, images, and audio inputs.

Built with [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/), and [OpenAI](https://openai.com/).

---

## 🚀 Features

*   **Multimodal Inputs:**
    *   📝 **Text:** Direct problem input.
    *   📷 **Vision:** Upload images of handwritten or printed math problems (OCR via GPT-4o).
    *   🎙️ **Audio:** Voice out questions using OpenAI Whisper integration.
*   **Agentic Workflow (LangGraph):**
    *   **Parser:** Structures raw input into a mathematical schema.
    *   **Retriever:** Fetches relevant math formulas and theorems from a vector database.
    *   **Solver:** Solves the problem step-by-step using retrieved context.
    *   **Verifier:** Critiques the solution for logical correctness before showing it to the user.
*   **Human-in-the-Loop (HITL):** Allows users to verify/edit parsed inputs before the agents start solving.
*   **Self-Learning Memory:** Saves verified correct solutions to local storage to improve future context.

---

## 📂 Project Structure

The project is organized as follows:

```text
multimodal-math-mentor/
├── data/                  # Stores vector DB and memory files
│   ├── chroma_db/         # ChromaDB persistence directory
│   └── problem_memory.json
├── src/                   # Core application logic
│   ├── agents.py          # LangChain Agents (Parser, Solver, Verifier)
│   ├── config.py          # Configuration and Environment variables
│   ├── graph.py           # LangGraph state machine definition
│   ├── processors.py      # Audio (Whisper) and Image (GPT-4o) processing
│   └── rag.py             # RAG logic and Vector Store management
├── main.py                # Streamlit Frontend entry point
├── requirements.txt       # Python dependencies
├── .env                   # API Keys (Not included in repo)
└── README.md              # Documentation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
*   Python 3.9 or higher.
*   An [OpenAI API Key](https://platform.openai.com/) (GPT-4o access recommended).

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/multimodal-math-mentor.git
cd multimodal-math-mentor
```

### 3. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Create a `requirements.txt` file (or use the one provided below) and install:

```bash
pip install -r requirements.txt
```

<details>
<summary><strong>See content of requirements.txt</strong></summary>

```text
streamlit
langchain
langchain-openai
langchain-community
langchain-core
langgraph
chromadb
openai
pydantic
python-dotenv
Pillow
```
</details>

### 5. Environment Configuration
Create a `.env` file in the root directory and add your OpenAI API Key:

```ini
OPENAI_API_KEY=sk-your-openai-api-key-here
```

---

## ▶️ How to Run

1.  Navigate to the project root directory.
2.  Run the Streamlit application:

```bash
streamlit run main.py
```

3.  The application will open in your browser at `http://localhost:8501`.

---

## 🧠 System Architecture (The Graph)

The core logic uses **LangGraph** to define a directed flow:

1.  **Start:** User provides input (Text/Image/Audio).
2.  **Processor:** Converts raw input to text (OCR/ASR).
3.  **Parser Node:** Structuring the text into JSON.
    *   *Conditional Edge:* If ambiguous, stop and ask User (HITL).
4.  **Retriever Node:** Queries ChromaDB for math formulas.
5.  **Solver Node:** Generates a solution using context.
6.  **Verifier Node:** Checks if the solution is mathematically sound.
    *   *Conditional Edge:* If incorrect, halt. If correct, proceed.
7.  **Explainer Node:** Formats the answer for the student.
8.  **End:** Display result.

---

