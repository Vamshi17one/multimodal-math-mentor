import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.config import Config

llm = ChatOpenAI(model=Config.MODEL_NAME)

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def process_image(image_bytes):
    """
    Acts as the OCR Agent. Extracts text from math images.
    """
    base64_image = encode_image(image_bytes)
    msg = [
        HumanMessage(content=[
            {"type": "text", "text": "Extract the math problem from this image exactly as written. If it's handwritten, transcribe it carefully. Output only the text."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])
    ]
    response = llm.invoke(msg)
    return response.content

def process_audio(audio_file):
    """
    Uses OpenAI Whisper API for ASR.
    """
    # In a real app, use client.audio.transcriptions.create
    # For this demo code, we assume the setup handles the API call
    from openai import OpenAI
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe", 
        file=audio_file,
        prompt="The following is a math problem. Use standard mathematical notation like 'square root', 'plus', 'integral'."
    )
    return transcript.text