import streamlit as st
from PIL import Image
import io
import sys
import asyncio
import re
from pathlib import Path

backend_path = Path(__file__).resolve().parents[3]
if str(backend_path) not in sys.path:
    sys.path.append(str(backend_path))

from src.processors import process_image, process_audio
from src.graph import build_graph
from src.rag import save_to_memory, process_and_index_files 

st.set_page_config(page_title="Multimodal Math Mentor", layout="wide")

st.title("🧮 Reliable Multimodal Math Mentor")


with st.sidebar:
    st.header("⚙️ Settings")
    input_mode = st.radio("Select Input Mode", ["Text", "Image", "Audio"])
    
    st.divider()
    st.header("📚 Knowledge Base")
    st.info("Upload textbooks, formula sheets, or class notes to improve agent accuracy.")
    
    uploaded_kb_files = st.file_uploader(
        "Upload Documents (PDF/TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_kb_files:
        if st.button("Index Documents"):
            with st.status("Ingesting Knowledge...", expanded=True) as status:
                st.write("Chunking and Embedding (Batch Size: 5)...")
                result_msg = process_and_index_files(uploaded_kb_files)
                st.success(result_msg)
                status.update(label="Knowledge Base Updated!", state="complete", expanded=False)

app = build_graph()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "final_result" not in st.session_state:
    st.session_state.final_result = None


async def handle_image_processing(img_bytes):
    return await process_image(img_bytes)

async def handle_audio_processing(audio_file):
    return await process_audio(audio_file)

async def run_agent_graph(initial_state, status_container):
    current_state = initial_state.copy()
    
    async for event in app.astream(initial_state):
        for node_name, state_update in event.items():
            status_container.write(f"✅ **{node_name.capitalize()}** finished task.")
            current_state.update(state_update)
            
            if "messages" in state_update and state_update["messages"]:
                st.caption(state_update["messages"][-1])
    return current_state


raw_input = ""

if input_mode == "Text":
    raw_input = st.text_area("Type your math problem here:")
    if raw_input:
        st.session_state.extracted_text = raw_input

elif input_mode == "Image":
    uploaded_file = st.file_uploader("Upload Math Problem", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Extract Text (OCR)"):
            with st.spinner("Analyzing image..."):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_bytes = img_byte_arr.getvalue()
                text = asyncio.run(handle_image_processing(img_bytes))
                st.session_state.extracted_text = text

elif input_mode == "Audio":
    audio_file = st.file_uploader("Upload Audio Question", type=["mp3", "wav", "m4a"])
    if audio_file:
        if st.button("Transcribe Audio"):
            with st.spinner("Listening..."):
                text = asyncio.run(handle_audio_processing(audio_file))
                st.session_state.extracted_text = text


if st.session_state.extracted_text:
    st.subheader("📝 Verify Input")
    edited_text = st.text_area("Confirm or Edit Question:", value=st.session_state.extracted_text, height=100)
    
    if st.button("Solve Problem"):
        initial_state = {
            "raw_input": edited_text,
            "input_type": input_mode.lower(),
            "messages": []
        }
        
        with st.status("Agents working...", expanded=True) as status:
            final_res = asyncio.run(run_agent_graph(initial_state, status))
            st.session_state.final_result = final_res
            status.update(label="Complete!", state="complete", expanded=False)


if st.session_state.final_result:
    res = st.session_state.final_result
    has_answer = "final_answer" in res
    
    if res.get("parsed_problem", {}).get("needs_clarification"):
        st.error("⚠️ The parser found the problem ambiguous. Please edit the text above.")
        st.warning(f"Trace: {res.get('messages', [])}")

    elif has_answer:
        st.divider()
        st.subheader("💡 Solution")
        
        if res.get("is_correct") is False:
            st.warning("⚠️ The Verifier flagged this solution as potentially incorrect.")
        
        # --- NEW DISPLAY LOGIC ---
        explanation_text = res.get("explanation", res["final_answer"])
        
        # 1. Attempt to extract a block equation to show it prominently using st.latex
        # Looks for patterns like $$ ... $$ or \[ ... \]
        latex_blocks = re.findall(r'\$\$(.*?)\$\$', explanation_text, re.DOTALL)
        
        if latex_blocks:
            # Display the last found major equation as the "Final Answer" highlight
            st.caption("Final Result:")
            st.latex(latex_blocks[-1])
        
        # 2. Render the full text with Markdown (Streamlit converts $...$ to LaTeX automatically)
        st.markdown(explanation_text)
        # -------------------------

        
        with st.expander("📚 Referenced Sources (RAG)"):
            docs = res.get("retrieved_docs", [])
            if docs:
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(doc) 
            else:
                st.write("No specific documents retrieved. Used general knowledge.")

        st.divider()
        st.subheader("🧠 Teach the System")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Accurate"):
                save_to_memory(res["parsed_problem"]["problem_text"], res["final_answer"], True)
                st.success("Stored in memory!")
        
        with col2:
            if st.button("❌ Incorrect"):
                st.info("Feedback logged.")
    else:
        st.error("⚠️ An unexpected error occurred.")