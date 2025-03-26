# Training Manual RAG Application

This is a RAG (Retrieval-Augmented Generation) application that uses a local LLaMA model to answer questions about training manuals. It includes features for extracting and displaying relevant figures from the manual. The application supports both web-based and Unity-based interfaces.

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- At least 8GB of RAM
- The LLaMA model file (mistral-7b-instruct-v0.1.Q4_0.gguf)
- The training manual PDF file
- Unity 2022.3 or later (for Unity integration)

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

### Web Interface
1. Start the Flask server:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

### Unity Integration
1. Start the Unity-specific Flask server:
```bash
python app-unity.py
```

2. Open the Unity project and ensure the following components are set up:
   - RAGClient component with proper API URL configuration
   - FigureManager component with figure prefab reference
   - FigureDisplay component on the figure prefab
   - UI elements for query input and response display

3. The Unity interface will connect to the Flask server and provide an interactive 3D environment for querying the training manual.

## Project Structure

- `app.py` - Main Flask application (web interface)
- `app-unity.py` - Flask application for Unity integration
- `extract_images_from_pdf.py` - Script to extract figures from PDF
- `process_pdf.py` - Script to process PDF text and generate embeddings
- `templates/` - HTML templates for web interface
- `static/` - Static files and extracted figures
- `embeddings/` - FAISS vector store for text embeddings
- `Figures/` - Extracted figures from the PDF
- `unity/` - Unity integration scripts
  - `RAGClient.cs` - Main Unity client for API communication
  - `FigureManager.cs` - Manages figure display in Unity
  - `FigureDisplay.cs` - Component for displaying individual figures
  - `Figure.cs` - Data structure for figure information

## Configuration

The application uses several environment variables that can be set in a `.env` file:
- `MODEL_PATH` - Path to the LLaMA model file
- `CUDA_VISIBLE_DEVICES` - GPU device selection
- `API_URL` - URL for the Flask server (Unity client)

## Notes

- The application uses a quantized LLaMA model for efficient inference
- Figures are automatically extracted and embedded for relevance matching
- The interface supports real-time streaming of model responses
- Unity integration provides a 3D environment for interactive manual exploration
- The model's temperature parameter (default: 0.8) can be adjusted to control response consistency 