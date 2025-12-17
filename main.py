import streamlit as st
from PIL import Image
import io
import sys
from pathlib import Path
backend_path = Path(__file__).resolve().parents[3]
if str(backend_path) not in sys.path:
    sys.path.append(str(backend_path))
from src.processors import process_image, process_audio
from src.graph import build_graph
from src.rag import save_to_memory

st.set_page_config(page_title="Multimodal Math Mentor", layout="wide")

st.title("🧮 Reliable Multimodal Math Mentor")
st.markdown("### Agentic RAG System for JEE Math")

# --- Sidebar: Input Mode ---
input_mode = st.sidebar.radio("Select Input Mode", ["Text", "Image", "Audio"])
app = build_graph()

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "final_result" not in st.session_state:
    st.session_state.final_result = None

# --- 1. Input Processing ---
raw_input = ""

if input_mode == "Text":
    raw_input = st.text_area("Type your math problem here:")
    if raw_input:
        st.session_state.extracted_text = raw_input

elif input_mode == "Image":
    uploaded_file = st.file_uploader("Upload Math Problem (JPG/PNG)", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Extract Text (OCR)"):
            with st.spinner("Analyzing image..."):
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_bytes = img_byte_arr.getvalue()
                
                text = process_image(img_bytes)
                st.session_state.extracted_text = text

elif input_mode == "Audio":
    audio_file = st.file_uploader("Upload Audio Question", type=["mp3", "wav", "m4a"])
    if audio_file:
        if st.button("Transcribe Audio"):
            with st.spinner("Listening..."):
                text = process_audio(audio_file)
                st.session_state.extracted_text = text

# --- 2. HITL: Parser Verification ---
if st.session_state.extracted_text:
    st.subheader("📝 Verify Input")
    edited_text = st.text_area("Confirm or Edit Question:", value=st.session_state.extracted_text, height=100)
    
    if st.button("Solve Problem"):
        # Run the Agent Graph
        initial_state = {
            "raw_input": edited_text,
            "input_type": input_mode.lower(),
            "messages": []
        }
        
        # --- FIX STARTS HERE: State Accumulation ---
        current_state = initial_state.copy()
        
        with st.status("Agents working...", expanded=True) as status:
            # Stream graph updates
            for event in app.stream(initial_state):
                for node_name, state_update in event.items():
                    status.write(f"✅ **{node_name.capitalize()}** finished task.")
                    
                    # Update the cumulative state with new info from this node
                    current_state.update(state_update)
                    
                    if "messages" in state_update:
                        st.caption(state_update["messages"][-1])
            
            # Save the fully accumulated state, not just the last event
            st.session_state.final_result = current_state
            status.update(label="Complete!", state="complete", expanded=False)
        # --- FIX ENDS HERE ---

# --- 3. Output & Feedback ---
if st.session_state.final_result:
    res = st.session_state.final_result
    
    # Check if we have an answer (even if unverified)
    has_answer = "final_answer" in res
    
    # 1. Parsing Failed
    if res.get("parsed_problem", {}).get("needs_clarification"):
        st.error("⚠️ The parser found the problem ambiguous. Please edit the text above.")
        st.warning(f"Trace: {res.get('messages', [])}")

    # 2. Answer exists, but verification might have failed
    elif has_answer:
        st.divider()
        st.subheader("💡 Solution")
        
        # Handle Verification Warnings
        if res.get("is_correct") is False:
            st.warning("⚠️ The Verifier flagged this solution as potentially incorrect. Please review the logic.")
            # Note: The 'explainer' node is skipped on verification failure, 
            # so we fallback to 'final_answer' raw text.
            st.markdown(res.get("explanation", res["final_answer"]))
        else:
            # Success path
            st.markdown(res.get("explanation", res["final_answer"]))
        
        with st.expander("Show Retrieved Context"):
            st.write(res.get("retrieved_docs", "No context retrieved."))

        st.divider()
        st.subheader("🧠 Teach the System")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Accurate"):
                save_to_memory(
                    res["parsed_problem"]["problem_text"], 
                    res["final_answer"], 
                    True
                )
                st.success("Stored in memory for future reference!")
        
        with col2:
            if st.button("❌ Incorrect"):
                st.text_area("What was wrong?")
                st.info("Feedback logged.")
    
    # 3. Fallback for other errors
    else:
        st.error("⚠️ An unexpected error occurred.")
        st.write(res)