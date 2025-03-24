# Training Manual RAG Application

This is a RAG (Retrieval-Augmented Generation) application that uses a local LLaMA model to answer questions about training manuals. It includes features for extracting and displaying relevant figures from the manual.

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- At least 8GB of RAM
- The LLaMA model file (mistral-7b-instruct-v0.1.Q4_0.gguf)
- The training manual PDF file

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd RAG
```

2. Create a virtual environment:
```bash
python -m venv rag2
source rag2/bin/activate  # On Linux/Mac
# or
.\rag2\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place the required files:
   - Put the LLaMA model file (`mistral-7b-instruct-v0.1.Q4_0.gguf`) in the root directory
   - Put the training manual PDF in the root directory

5. Extract figures and generate embeddings:
```bash
python extract_images_from_pdf.py
python process_pdf.py
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

## Project Structure

- `app.py` - Main Flask application
- `extract_images_from_pdf.py` - Script to extract figures from PDF
- `process_pdf.py` - Script to process PDF text and generate embeddings
- `templates/` - HTML templates
- `static/` - Static files and extracted figures
- `embeddings/` - FAISS vector store for text embeddings
- `Figures/` - Extracted figures from the PDF

## Configuration

The application uses several environment variables that can be set in a `.env` file:
- `MODEL_PATH` - Path to the LLaMA model file
- `CUDA_VISIBLE_DEVICES` - GPU device selection

## Notes

- The application uses a quantized LLaMA model for efficient inference
- Figures are automatically extracted and embedded for relevance matching
- The interface supports real-time streaming of model responses 