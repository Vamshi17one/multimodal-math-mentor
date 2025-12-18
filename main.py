import streamlit as st
import os
from PIL import Image
import io
import sys
import asyncio
from pathlib import Path

backend_path = Path(__file__).resolve().parents[3]
if str(backend_path) not in sys.path:
    sys.path.append(str(backend_path))

# Import Config to check key existence
from src.config import Config 
from src.processors import process_image, process_audio
from src.graph import build_graph
from src.rag import save_to_memory, process_and_index_files 

st.set_page_config(page_title="Multimodal Math Mentor", layout="wide")
st.title("🧮 Reliable Multimodal Math Mentor")

# --- Initialize Session State for API Key ---
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = os.getenv("OPENAI_API_KEY", "")

def apply_api_key():
    """Callback to set the API Key in environment"""
    if st.session_state.user_api_key:
        os.environ["OPENAI_API_KEY"] = st.session_state.user_api_key
        st.toast("API Key Applied Successfully!", icon="✅")
    else:
        st.toast("Please enter a key before applying.", icon="⚠️")

def clear_api_key():
    """Callback to clear API Key from environment and session"""
    st.session_state.user_api_key = ""
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    st.toast("API Key Cleared!", icon="🗑️")

with st.sidebar:
    st.header("⚙️ Settings")
    
    # --- UPDATED: API Key Management ---
    st.text_input(
        "OpenAI API Key", 
        type="password", 
        key="user_api_key",
        help="Enter your key and click Apply."
    )
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.button("Apply Key", on_click=apply_api_key, use_container_width=True)
    with col_k2:
        st.button("Clear Key", on_click=clear_api_key, use_container_width=True)

    input_mode = st.radio("Select Input Mode", ["Text", "Image", "Audio"])
    st.divider()
    st.header("📚 Knowledge Base")
    uploaded_kb_files = st.file_uploader("Upload Documents", type=["pdf", "txt"], accept_multiple_files=True)
    
    # Check for key before indexing
    if uploaded_kb_files and st.button("Index Documents"):
        if not Config.get_openai_key():
            st.error("Please apply an OpenAI API Key first.")
        else:
            with st.status("Ingesting Knowledge..."):
                result_msg = process_and_index_files(uploaded_kb_files)
                st.success(result_msg)

# Build graph (lazy loading of key happens inside agents)
app = build_graph()

# ... [Session State initialization] ...
if "messages" not in st.session_state:
    st.session_state.messages = []
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "final_result" not in st.session_state:
    st.session_state.final_result = None

async def run_agent_graph(initial_state, status_container):
    current_state = initial_state.copy()
    async for event in app.astream(initial_state):
        for node_name, state_update in event.items():
            status_container.write(f"✅ **{node_name.replace('_', ' ').capitalize()}** finished.")
            current_state.update(state_update)
            if "messages" in state_update:
                st.caption(state_update["messages"][-1])
    return current_state


# --- INPUT HANDLING SECTIONS ---
raw_input = ""

if input_mode == "Text":
    # --- UPDATED: Text Input with Action Button ---
    with st.form(key="text_input_form"):
        raw_input = st.text_area("Type your math problem here:", height=150)
        submit_col1, submit_col2 = st.columns([1, 4])
        with submit_col1:
            process_btn = st.form_submit_button("🚀 Analyze & Solve", type="primary")
    
    # If user types and clicks button, update state
    if process_btn and raw_input:
        st.session_state.extracted_text = raw_input
        # We don't rerun immediately to allow the Verify section below to render naturally

elif input_mode == "Image":
    uploaded_file = st.file_uploader("Upload Math Problem", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        if st.button("Extract Text (OCR)"):
            with st.spinner("Analyzing image..."):
                img_bytes = io.BytesIO()
                image.save(img_bytes, format=image.format)
                text = asyncio.run(process_image(img_bytes.getvalue()))
                st.session_state.extracted_text = text

elif input_mode == "Audio":
    audio_file = st.file_uploader("Upload Audio Question", type=["mp3", "wav", "m4a"])
    if audio_file and st.button("Transcribe"):
        # Check for key before Audio processing
        if not Config.get_openai_key():
            st.error("Please apply an OpenAI API Key in settings.")
        else:
            with st.spinner("Listening..."):
                text = asyncio.run(process_audio(audio_file))
                st.session_state.extracted_text = text


# --- PROCESSING SECTION ---
if st.session_state.extracted_text:
    st.divider()
    st.subheader("📝 Verify Input")
    
    # Allow user to edit extracted text before solving
    edited_text = st.text_area("Confirm Question:", value=st.session_state.extracted_text, height=100)
    
    solve_btn = st.button("Confirm & Solve Problem", type="primary")
    
    # Trigger graph only when Solve button is clicked
    if solve_btn:
        # Check for key before Solving
        if not Config.get_openai_key():
            st.error("Please apply an OpenAI API Key in settings.")
        else:
            initial_state = {
                "raw_input": edited_text,
                "input_type": input_mode.lower(),
                "messages": [],
                "retrieved_docs": [],
                "code_snippet": "",
                "code_output": ""
            }
            
            with st.status("Agents working...", expanded=True) as status:
                final_res = asyncio.run(run_agent_graph(initial_state, status))
                st.session_state.final_result = final_res
                status.update(label="Complete!", state="complete", expanded=False)

# ... [The rest of the display logic remains the same] ...
if st.session_state.final_result:
    res = st.session_state.final_result
    
    if res.get("parsed_problem", {}).get("needs_clarification"):
        st.error("⚠️ Ambiguous input. Please edit the text.")
    
    elif res.get("is_correct") is False:
        st.error("🛑 The Verifier blocked this solution.")
        st.write(f"**Critique:** {res.get('messages', ['Unknown Error'])[-1]}")
        st.info("Please reformulate your question or add more context.")
        
    else:
        st.divider()
        st.subheader("💡 Solution")
        st.markdown(res.get("explanation", "No explanation generated."))
        
        with st.expander("🕵️ Agent Logic & Tools"):
            cat = res.get("problem_category", "Unknown")
            st.info(f"**Router Decision:** {cat.capitalize()} Path")
            
            if cat == "calculation":
                st.write("**🐍 Python Code Executed:**")
                st.code(res.get("code_snippet", ""), language="python")
                st.write("**Output:**")
                st.code(res.get("code_output", ""))
            
            if cat == "conceptual" and res.get("retrieved_docs"):
                st.write("**📚 RAG Context:**")
                for doc in res["retrieved_docs"]:
                    st.text(doc[:200] + "...")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Helpful"):
                save_to_memory(res["parsed_problem"]["problem_text"], res["final_answer"], True)
                st.success("Saved!")
        with col2:
            st.button("❌ Incorrect")
            
    st.divider()
    if st.button("🔄 Clear Response", type="primary"):
        st.session_state.final_result = None
        st.rerun()