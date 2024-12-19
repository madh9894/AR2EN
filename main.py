from fastapi import FastAPI, HTTPException
from transformers import pipeline
from typing import Dict
import uuid

app = FastAPI()

# Load translation model (Arabic to English)
translator_ar_to_en = pipeline("translation_ar_to_en", model="Helsinki-NLP/opus-mt-ar-en")

# Dictionary to store the status of translation requests
translation_status: Dict[str, Dict] = {}

@app.get("/")
def home():
    """
    Root endpoint to confirm the translation service is running.
    """
    return {"message": "AR2EN Translation Service with Status Tracking is running"}

@app.post("/translate/ar2en")
def translate_ar2en(text: str):
    """
    Translate text from Arabic to English and store status.
    """
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required")
    
    # Generate a unique ID for the translation request
    request_id = str(uuid.uuid4())
    
    # Perform translation
    result = translator_ar_to_en(text, max_length=400)[0]['translation_text']
    
    # Update the status dictionary
    translation_status[request_id] = {"status": "completed", "result": result}

    return {
        "request_id": request_id,
        "status": "completed",
        "translated_text": result
    }

@app.get("/translate/ar2en/status/{id}")
def get_ar2en_status(id: str):
    """
    Retrieve the status of an Arabic to English translation request.
    """
    if id not in translation_status:
        raise HTTPException(status_code=404, detail="Request ID not found")
    return {"request_id": id, "status": translation_status[id]["status"], "result": translation_status[id]["result"]}
