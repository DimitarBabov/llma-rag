# Docker Packaging for RAG Backend

## Overview

Package the RAG backend (Flask API + Mistral-7B + FAISS + embeddings) into a fully self-contained Docker image so the Unity developer can run `docker compose up` and immediately connect their Unity app to the API.

## Todos

- [ ] Create Dockerfile based on nvidia/cuda with Python, deps, pre-cached HF model, and all project data baked in
- [ ] Create docker-compose.yml with GPU reservation and port 5001 mapping
- [ ] Create .dockerignore to exclude .git, PDFs, Q6_K model, unity/, Assets/, __pycache__
- [ ] Update MODEL_PATH in app-unity.py to use container-relative path or env variable
- [ ] Create README-SETUP.md with step-by-step instructions for the Unity developer

## What gets packaged

Everything the Unity developer needs to run the API server, baked into a single Docker image:

- Python runtime + all dependencies from `requirements.txt`
- CUDA runtime (for GPU-accelerated LLM inference)
- Mistral-7B model file (3.9 GB `.gguf`)
- FAISS vector index (`embeddings/` -- 2.2 MB)
- Figure images (`Figures/` -- 27 MB)
- Figure metadata (`figures.json`)
- HuggingFace sentence-transformers model (pre-cached at build time, ~130 MB)
- The API server: `app-unity.py`

**Estimated image size**: ~10-12 GB (dominated by CUDA base image + model file)

## Files to create

### 1. `Dockerfile`

Based on `nvidia/cuda:12.2.0-runtime-ubuntu22.04`:

- Install Python 3.11 + pip
- Install all Python dependencies (with CUDA-enabled `llama-cpp-python`)
- Pre-download the HuggingFace embeddings model so there's no download at runtime
- Copy in all project data (model, embeddings, figures, figures.json, app-unity.py)
- Expose port 5001
- Entrypoint: `python app-unity.py`

### 2. `docker-compose.yml`

```yaml
services:
  rag-backend:
    build: .
    ports:
      - "5001:5001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 3. `.dockerignore`

Exclude unnecessary files from the build context (`.git`, `__pycache__`, `*.pdf`, the Q6_K model variant, `unity/`, `Assets/`, etc.)

### 4. `README-SETUP.md`

Step-by-step instructions for the Unity developer:

1. Install Docker Desktop for Windows
2. Install NVIDIA GPU drivers (if not already)
3. Enable GPU support in Docker Desktop settings
4. Run `docker compose up --build` (first time, takes a while to build)
5. Run `docker compose up` (subsequent times, starts in seconds)
6. API is available at `http://localhost:5001/api`
7. Unity app connects to `http://<machine-ip>:5001/api`

## Code change required

In `app-unity.py`, the `MODEL_PATH` is hardcoded to `/home/mko0/RAG/mistral-7b-instruct-v0.1.Q4_0.gguf`. This needs to be updated to use the path inside the container (e.g. `/app/model/mistral-7b-instruct-v0.1.Q4_0.gguf` or driven by an environment variable). The API URL in the Unity `RAGClient.cs` will need to be updated by the Unity developer to point at the Docker host's IP.

## What the Unity developer's experience looks like

```
git clone <repo>
cd RAG
docker compose up --build    # first time: builds image (~15-20 min)
                              # subsequent: starts in ~30 seconds
```

Server is up. Unity app connects to `http://localhost:5001/api`. Done.
