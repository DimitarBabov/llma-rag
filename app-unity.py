from flask import Flask, request, jsonify, Response
import os
import json
import numpy as np
import time

# FAISS Vector Store & Embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LLaMA (CUDA-Accelerated)
from llama_cpp import Llama

###############################################################################
# CONFIG
###############################################################################
VECTOR_DB_PATH = "embeddings/"
FIGURES_JSON = "figures.json"
FIGURES_FOLDER = "static/Figures/"
MODEL_PATH = "/home/mko0/RAG/mistral-7b-instruct-v0.1.Q4_0.gguf"

app = Flask(__name__)

###############################################################################
# SETUP RAG COMPONENTS
###############################################################################
# 1. Load HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2", 
    model_kwargs={"device": "cuda"}
)

# 2. Load FAISS vector store
vector_db = FAISS.load_local(
    VECTOR_DB_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# 3. Load figure metadata
with open(FIGURES_JSON, "r", encoding="utf-8") as f:
    figures_dict = json.load(f)

# 4. Precompute figure embeddings
figure_embeddings = {}
for figure_key, figure_info in figures_dict.items():
    text_to_embed = figure_info["title"]
    figure_embeddings[figure_key] = embedding_model.embed_query(text_to_embed)

# 5. Load LLaMA Model
model = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_gpu_layers=32,
    use_mlock=True,
    verbose=True
)

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def retrieve_context(query, vector_store):
    docs = vector_store.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

def get_relevant_figures(text, threshold=0.5):
    query_embedding = embedding_model.embed_query(text)
    relevant = []
    for fig_key, fig_emb in figure_embeddings.items():
        score = np.dot(query_embedding, fig_emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(fig_emb)
        )
        if score > threshold:
            relevant.append((fig_key, score))
    relevant.sort(key=lambda x: x[1], reverse=True)
    return relevant

def build_prompt(user_query):
    context = retrieve_context(user_query, vector_db)
    return (
        "You are an AI assistant specialized in summarizing training manuals. "
        "Leverage the provided context to answer the question, including "
        "relevant details.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n\n"
        "Answer:"
    )

###############################################################################
# API ENDPOINTS
###############################################################################

@app.route("/api/generator", methods=["POST"])
def generate():
    """
    Streaming API endpoint for Unity
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    def generate_stream():
        user_query = data["query"]
        prompt = build_prompt(user_query)
        
        # Stream the response
        response = model(
            prompt,
            max_tokens=256,
            stream=True,
            temperature=0.0
        )

        accumulated_text = ""
        for chunk in response:
            if chunk:
                text = chunk["choices"][0]["text"]
                accumulated_text += text
                
                # Get relevant figures based on accumulated text
                relevant_figs = get_relevant_figures(accumulated_text)
                figures_data = []
                
                for fig_key, score in relevant_figs:
                    fig_info = figures_dict[fig_key]
                    figures_data.append({
                        "filename": fig_info["filename"],
                        "title": fig_info["title"],
                        "score": float(score)
                    })

                # Create the chunk data
                chunk_data = {
                    "text": text,
                    "figures": figures_data
                }
                
                yield f"data: {json.dumps(chunk_data)}\n\n"
                time.sleep(0.01)  # Small delay to prevent overwhelming the client

        # Send completion message
        yield "data: [DONE]\n\n"

    return Response(generate_stream(), mimetype="text/event-stream")

@app.route("/api/query", methods=["POST"])
def query():
    """
    Non-streaming API endpoint for Unity (fallback)
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    user_query = data["query"]
    prompt = build_prompt(user_query)
    
    # Generate response
    response = model(prompt, max_tokens=256, temperature=0.0)
    answer_text = response["choices"][0]["text"].strip()
    
    # Get relevant figures
    relevant_figs = get_relevant_figures(answer_text)
    figures_data = []
    
    for fig_key, score in relevant_figs:
        fig_info = figures_dict[fig_key]
        figures_data.append({
            "filename": fig_info["filename"],
            "title": fig_info["title"],
            "score": float(score)
        })

    return jsonify({
        "answer": answer_text,
        "figures": figures_data
    })

@app.route("/api/figures", methods=["GET"])
def list_figures():
    """
    API endpoint to get a list of all available figures
    """
    figures_list = []
    for fig_key, fig_info in figures_dict.items():
        figures_list.append({
            "filename": fig_info["filename"],
            "title": fig_info["title"]
        })
    return jsonify({"figures": figures_list})

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Simple health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "embeddings_loaded": True
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001) 