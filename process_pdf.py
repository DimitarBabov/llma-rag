import os
import json
import fitz  # PyMuPDF
import pytesseract
import numpy as np
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

### CONFIG ###
PDF_PATH = "training_manual.pdf"
FIGURES_FOLDER = "Figures"
FIGURES_JSON = "figures.json"
FIGURE_TEXTS_JSON = "figure_texts.json"
VECTOR_DB_PATH = "embeddings/"
TEXT_OUTPUT_PATH = "training_manual_text.txt"

os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# ------------------------------------------------------------------------------
# Load Figures Metadata - which now contains {"title": "...", "filename": "..."}
# ------------------------------------------------------------------------------
with open(FIGURES_JSON, "r", encoding="utf-8") as f:
    figures_dict = json.load(f)

# Initialize Embeddings Model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Dictionary to store extracted text per figure
figure_texts = {}

### 1. Extract Text from the PDF ###
def extract_text_from_pdf():
    doc = fitz.open(PDF_PATH)
    extracted_text = "\n".join(page.get_text("text") for page in doc)
    with open(TEXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    doc.close()
    return extracted_text

### 2. Extract Text from Figures Using OCR ###
def extract_text_from_figures():
    """
    Uses 'filename' from figures_dict to open each figure,
    run OCR, and store text. 
    """
    for figure_key, info in figures_dict.items():
        figure_title = info["title"]
        figure_filename = info["filename"]  # EXACT name from first script

        image_path = os.path.join(FIGURES_FOLDER, figure_filename)

        if os.path.exists(image_path):
            img = Image.open(image_path)
            extracted_text = pytesseract.image_to_string(img).strip()

            if extracted_text:
                figure_texts[figure_key] = {
                    "title": figure_title,
                    "text": extracted_text
                }
                print(f"✅ Extracted text from {figure_filename}: {extracted_text[:50]}...")
            else:
                print(f"⚠️ No text found in {figure_filename}")
        else:
            print(f"⚠️ Image not found: {figure_filename}")

### 3. Create FAISS Vector Store (Including PDF & Figure Texts)
def create_vector_store(pdf_text):
    all_text_data = [pdf_text]  # Start with entire PDF text

    # Add figure texts
    for figure_key, data in figure_texts.items():
        full_text = f"{data['title']}\n{data['text']}"
        all_text_data.append(full_text)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text("\n\n".join(all_text_data))

    # Create embeddings & FAISS index
    vector_store = FAISS.from_texts(chunks, embeddings_model)
    vector_store.save_local(VECTOR_DB_PATH)
    print("✅ FAISS vector store created with PDF and figure texts.")

### RUN PIPELINE ###
print("📄 Extracting text from PDF...")
pdf_text = extract_text_from_pdf()

print("🖼️ Extracting text from figures...")
extract_text_from_figures()

print("🔍 Creating FAISS vector store...")
create_vector_store(pdf_text)

# Save extracted text from figures
with open(FIGURE_TEXTS_JSON, "w", encoding="utf-8") as f:
    json.dump(figure_texts, f, indent=4)

print("✅ Processing complete! Ready to query.")
