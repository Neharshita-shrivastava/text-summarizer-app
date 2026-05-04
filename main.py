# IMPORTANT: Import torch FIRST before any other libraries
import torch
import os
import platform
import re

# Force load the problematic DLL on Windows (add this section if import order alone doesn't fix)
if platform.system() == "Windows":
    try:
        from importlib.util import find_spec
        if find_spec("torch"):
            import ctypes
            torch_path = find_spec("torch").origin
            if torch_path:
                dll_path = os.path.join(os.path.dirname(torch_path), "lib", "c10.dll")
                if os.path.exists(dll_path):
                    ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass  # Silently continue if DLL loading fails

# Now import other libraries
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# Load the model and tokenizer with error handling
try:
    model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
    tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
    print("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure './saved_summary_model' directory exists and contains the model files")
    raise

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using MPS (Apple Silicon) device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using CUDA device")
else:
    device = torch.device("cpu")
    print("✅ Using CPU device")

model.to(device)

# Setup templates - make sure index.html exists in the same directory
templates = Jinja2Templates(directory=".")

# Create static files directory if needed (optional)
# app.mount("/static", StaticFiles(directory="static"), name="static")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    """Clean and preprocess the input text"""
    if not text:
        return ""
    text = re.sub(r"\r\n", " ", text)  # Remove line breaks
    text = re.sub(r"\s+", " ", text)   # Remove extra spaces
    text = re.sub(r"<.*?>", " ", text) # Remove HTML tags
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue: str) -> str:
    """Generate summary for the given dialogue"""
    if not dialogue or len(dialogue.strip()) == 0:
        return "No text provided for summarization."
    
    # Clean the input
    dialogue = clean_data(dialogue)
    
    # Tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate summary
    with torch.no_grad():  # Disable gradient calculation for inference
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            min_length=30,  # Add minimum length for better summaries
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,  # Prevent repetition
            temperature=0.7  # Add some randomness for better results
        )
    
    # Decode the output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    
    return summary if summary else "Could not generate summary."

# API Endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    """POST endpoint for text summarization"""
    try:
        summary = summarize_dialogue(dialogue_input.dialogue)
        return {"summary": summary, "status": "success"}
    except Exception as e:
        return {"summary": f"Error: {str(e)}", "status": "error"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """GET endpoint for the web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "device": str(device)}

# Run the app (for direct execution)
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Text Summarizer App...")
    print(f"📱 Access the web interface at: http://127.0.0.1:8000")
    print(f"📝 API docs available at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)