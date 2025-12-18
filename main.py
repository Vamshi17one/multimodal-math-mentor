import streamlit as st
from PIL import Image
import io
import sys
import asyncio
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
    uploaded_kb_files = st.file_uploader("Upload Documents", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_kb_files and st.button("Index Documents"):
        with st.status("Ingesting Knowledge..."):
            result_msg = process_and_index_files(uploaded_kb_files)
            st.success(result_msg)

app = build_graph()

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


raw_input = ""
if input_mode == "Text":
    raw_input = st.text_area("Type your math problem here:")
    if raw_input: st.session_state.extracted_text = raw_input
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
        with st.spinner("Listening..."):
            text = asyncio.run(process_audio(audio_file))
            st.session_state.extracted_text = text


if st.session_state.extracted_text:
    st.subheader("📝 Verify Input")
    edited_text = st.text_area("Confirm Question:", value=st.session_state.extracted_text, height=100)
    
    if st.button("Solve Problem"):
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