from flask import Flask, request, render_template, Response
import os
import json
import numpy as np

from PIL import Image

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
    allow_dangerous_deserialization=True  # For trusted local usage only
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
#    Ensure llama_cpp >= 0.1.52 for streaming
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
    # 1. Retrieve context
    context = retrieve_context(user_query, vector_db)
    
    # 2. Construct final prompt
    return (
        "You are an AI assistant specialized in summarizing training manuals. "
        "Leverage the provided context to answer the question, including "
        "relevant details.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n\n"
        "Answer:"
    )

###############################################################################
# FLASK ROUTES
###############################################################################

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Display a form; user types query -> we do SSE streaming from /stream
    so the response arrives token-by-token in the browser.
    """
    if request.method == "POST":
        user_query = request.form.get("query", "")
        if not user_query.strip():
            return render_template("index.html",
                                   response="No query provided",
                                   matched_figs=None)
        
        # We'll pass the user's query to the SSE endpoint instead of generating
        # everything here. That endpoint will stream tokens back to the browser.
        return render_template("index.html",
                               user_query=user_query,
                               response=None,
                               matched_figs=None,
                               streaming=True)
    return render_template("index.html", matched_figs=None)

@app.route("/stream")
def stream():
    """
    SSE endpoint: Streams LLaMA tokens token-by-token.
    Also, once done, we compute which figures are relevant and send them in SSE.
    """
    # 1. Get user_query from request args
    user_query = request.args.get("user_query", "")
    if not user_query:
        return Response("data: No query provided\n\n", mimetype="text/event-stream")

    prompt = build_prompt(user_query)

    # We'll define a generator function that yields SSE events
    def token_generator():
        # 2. Call LLaMA with streaming
        stream_resp = model(prompt, max_tokens=256, stream=True, temperature=0.0)
        full_text = []
        
        # 3. For each token in the stream, yield an SSE "message" event
        for chunk in stream_resp:
            if "choices" in chunk:
                token_str = chunk["choices"][0]["text"]
                # Maybe split on spaces if you want truly "word by word".
                # We'll keep it as chunked tokens for now.
                if token_str:
                    # SSE format: "data: <string>\n\n"
                    yield f"data: {token_str}\n\n"
                    full_text.append(token_str)
        
        # 4. Once done streaming the main text, combine it
        answer_text = "".join(full_text).strip()
        
        # 5. Identify relevant figures from the final answer
        relevant_figs = get_relevant_figures(answer_text, threshold=0.5)

        # 6. Build an SSE event with figure data
        # We'll send them as JSON lines or a special format
        if relevant_figs:
            # Start the figure event
            yield f"event: figures\n"
            
            # Convert them into lines or JSON
            # Here is a simple approach:
            # e.g. "Fuel_System.png|Fuel System|0.80" for each figure
            figs_payload = []
            for fig_key, score in relevant_figs:
                fig_info = figures_dict[fig_key]
                # "filename|title|similarity"
                figs_payload.append(f"{fig_info['filename']}|{fig_info['title']}|{score:.2f}")
            
            # SSE data
            yield f"data: {'||'.join(figs_payload)}\n\n"
        else:
            yield f"event: figures\ndata: none\n\n"
    
    # Return a streaming response
    return Response(token_generator(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
