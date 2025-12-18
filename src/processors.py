import io
import asyncio
import numpy as np
import easyocr
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from src.config import Config
from openai import AsyncOpenAI



reader = easyocr.Reader(['en']) 

def _run_ocr_sync(image_bytes):
    """
    Synchronous function to perform OCR using EasyOCR.
    Returns extracted text and checks confidence levels.
    """
    try:
        
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)

        
        results = reader.readtext(image_np, detail=1)
        
        extracted_lines = []
        confidences = []

        for (_, text, conf) in results:
            extracted_lines.append(text)
            confidences.append(conf)

        full_text = "\n".join(extracted_lines)

        
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
        else:
            avg_conf = 0.0

        
        
        CONFIDENCE_THRESHOLD = 0.5
        if avg_conf < CONFIDENCE_THRESHOLD:
            warning_msg = (
                f"[⚠️ SYSTEM WARNING: Low OCR Confidence ({avg_conf:.2f})]\n"
                "The text extraction might be inaccurate. Please review and correct the math notation below carefully.\n"
                "---------------------------------------------------\n"
            )
            return warning_msg + full_text
        
        return full_text

    except Exception as e:
        return f"Error during OCR processing: {str(e)}"

async def process_image(image_bytes):
    """
    Acts as the OCR Agent. 
    Uses EasyOCR in a separate thread to avoid blocking the async event loop.
    """
    loop = asyncio.get_running_loop()
    
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, _run_ocr_sync, image_bytes)
    
    return result

async def process_audio(audio_file):
    """
    Uses OpenAI GPT-4o-Transcribe API for ASR asynchronously.
    """
    # Use dynamic key
    client = AsyncOpenAI(api_key=Config.get_openai_key())
    
    if not hasattr(audio_file, "name"):
        audio_file.name = "audio.mp3" 

    transcript = await client.audio.transcriptions.create(
        model="gpt-4o-transcribe", 
        file=audio_file,
        prompt="The following is a math problem. Use standard mathematical notation like 'square root', 'plus', 'integral', 'pi'."
    )
    return transcript.text