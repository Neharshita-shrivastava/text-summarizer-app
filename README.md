# 🧠 Text Summarizer Web App (T5 + FastAPI)
A local NLP-based web application that generates concise summaries from long text using a fine-tuned T5 transformer model. Built using FastAPI and PyTorch.

## Features

- Abstractive text summarization using T5  
- FastAPI backend for efficient inference  
- Simple web interface using HTML + Jinja2  
- Text preprocessing (cleaning & normalization)  
- Beam search decoding for improved summary quality  
- Automatic device selection (CPU / GPU / MPS)

## 🛠️ Tech Stack

- Python  
- FastAPI  
- PyTorch  
- HuggingFace Transformers  
- Jinja2 Templates  

## 📂 Project Structure
project/
│── app.py
│── index.html
│── saved_summary_model/
│── README.md

##  How to Run

```bash
git clone <https://github.com/Neharshita-shrivastava/text-summarizer-app.git>
cd project
pip install -r requirement.txt
uvicorn app:app --reload

## Model Details
- Model: T5-small (fine-tuned)
- Task: Abstractive Text Summarization
- Library: HuggingFace Transformers

## API Endpoint

POST /summarize

Input:
{
  "text": "your long text"
}

Output:
{
  "summary": "short summary"
}

## Example
Input:
"Artificial Intelligence is transforming industries..."

Output:
"AI is transforming industries by improving efficiency."
