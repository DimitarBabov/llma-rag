import os
import json
import numpy as np
from PIL import Image

# FAISS Vector Store & Embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LLaMA (CUDA-Accelerated)
from llama_cpp import Llama

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
VECTOR_DB_PATH = "embeddings/"
FIGURES_JSON = "figures.json"
FIGURES_FOLDER = "Figures/"
MODEL_PATH = "/home/mko0/RAG/mistral-7b-instruct-v0.1.Q4_0.gguf"

# ------------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------------
# 1. Load HuggingFace Embeddings (once)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2", 
    model_kwargs={"device": "cuda"}
)

# 2. Load FAISS vector store
vector_db = FAISS.load_local(
    VECTOR_DB_PATH, 
    embedding_model,
    allow_dangerous_deserialization=True  # For trusted local usage only
)

# 3. Load figure metadata
with open(FIGURES_JSON, "r", encoding="utf-8") as f:
    figures_dict = json.load(f)

# 4. Precompute and store figure embeddings (dictionary: figure_key -> embedding)
figure_embeddings = {}
for figure_key, figure_info in figures_dict.items():
    text_to_embed = figure_info["title"]
    figure_embeddings[figure_key] = embedding_model.embed_query(text_to_embed)

# 5. Load LLaMA Model
model = Llama(
    model_path="/home/mko0/RAG/mistral-7b-instruct-v0.1.Q4_0.gguf",
    n_ctx=4096,         # Increase context length
    n_gpu_layers=32,    # Force all layers to use GPU
    use_mmap=False,     # Avoid memory-mapping to CPU
    use_mlock=True,     # Keep model in memory
    verbose=True        # Enable logging to check CUDA usage
)

# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------
def retrieve_context(query, vector_store):
    """Retrieve relevant chunks of text from the FAISS vector store."""
    docs = vector_store.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])


def get_relevant_figures(text, threshold=0.5):
    """
    Return all figures above a certain similarity threshold.
    Figures are sorted by descending similarity.
    """
    query_embedding = embedding_model.embed_query(text)
    relevant = []
    
    for fig_key, fig_embedding in figure_embeddings.items():
        # Cosine similarity
        score = np.dot(query_embedding, fig_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(fig_embedding)
        )
        if score > threshold:
            relevant.append((fig_key, score))
    
    # Sort by highest similarity first
    relevant.sort(key=lambda x: x[1], reverse=True)
    return relevant


def query_llm_with_figures(query):
    """Generate an LLM response and find relevant figures."""
    # 1. Retrieve context from vector store
    context = retrieve_context(query, vector_db)
    
    # 2. Construct prompt (simple prompt engineering)
    prompt = (
        "You are an AI assistant specialized in summarizing training manuals. "
        "Leverage the provided context to answer the question, including "
        "all relevant details.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    
    # 3. Call LLaMA for completion
    response = model(prompt, max_tokens=256, temperature=0.3)
    response_text = response["choices"][0]["text"].strip()
    
    print("\n📝 LLaMA Response:\n", response_text)
    
    # 4. Identify all relevant figures from the LLM answer
    relevant_figs = get_relevant_figures(response_text, threshold=0.5)
    if relevant_figs:
        print("\n📌 Matched Figures:")
        for fig_key, score in relevant_figs:
            fig_info = figures_dict[fig_key]           

            image_filename = fig_info["filename"]
            image_path = os.path.join(FIGURES_FOLDER, image_filename)
            
            print(f"- {fig_info['title']} (Similarity: {score:.2f})")
            if os.path.exists(image_path):
                Image.open(image_path).show()
            else:
                print(f"⚠️ Image not found: {image_path}")


# ------------------------------------------------------------------------------
# MAIN INTERACTIVE LOOP
# ------------------------------------------------------------------------------
print("\n🔹 Type 'exit' to stop.")

while True:
    user_input = input("\nAsk about the training manual: ")
    if user_input.lower() == "exit":
        print("\n👋 Exiting. Have a great day!")
        break
    query_llm_with_figures(user_input)
