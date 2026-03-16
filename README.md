# Training Manual RAG Backend

A RAG (Retrieval-Augmented Generation) backend that answers questions about military training manuals using a local Mistral-7B LLM. It retrieves relevant text passages via FAISS vector search and returns matching figures from the manual. The backend runs inside WSL (Windows Subsystem for Linux) and serves a REST API that the Unity client connects to.

## Requirements

- Windows 10 (version 2004+) or Windows 11
- NVIDIA GPU with at least 8 GB VRAM
- NVIDIA GPU drivers installed (Windows side)
- At least 16 GB RAM recommended
- ~12 GB free disk space (WSL + model + dependencies)

## Setup

### Step 1: Install WSL

Open PowerShell **as Administrator** and run:

```powershell
wsl --install
```

This installs WSL 2 with Ubuntu. Restart your computer when prompted. After restart, Ubuntu will open and ask you to create a username and password.

Verify the installation:

```powershell
wsl --version
```

### Step 2: Install NVIDIA CUDA support in WSL

NVIDIA GPU drivers installed on Windows automatically provide CUDA support inside WSL. No separate CUDA install is needed inside WSL.

Verify GPU access from inside WSL (open the Ubuntu terminal):

```bash
nvidia-smi
```

You should see your GPU listed. If this fails, update your NVIDIA drivers from https://www.nvidia.com/drivers.

### Step 3: Clone the repository

Inside the WSL Ubuntu terminal:

```bash
cd ~
git clone https://github.com/DimitarBabov/llma-rag.git
cd llma-rag
git checkout unity
```

### Step 4: Download the LLM model

Download the Mistral-7B GGUF model (~3.9 GB) into the project root:

```bash
wget -O mistral-7b-instruct-v0.1.Q4_0.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_0.gguf"
```

Or download manually from https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF (file: `mistral-7b-instruct-v0.1.Q4_0.gguf`) and copy it into the project folder.

### Step 5: Install Python and create a virtual environment

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv
python3 -m venv rag2
source rag2/bin/activate
```

### Step 6: Install dependencies

Install PyTorch with CUDA first (requires the special PyTorch index):

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Then install everything else:

```bash
pip install -r requirements.txt
```

### Step 7: Update the model path

Open `app-unity.py` and update `MODEL_PATH` on line 20 to match your WSL username:

```python
MODEL_PATH = "/home/<your-wsl-username>/llma-rag/mistral-7b-instruct-v0.1.Q4_0.gguf"
```

### Step 8: Set up port forwarding (WSL to Windows)

The Unity app runs on Windows but the server runs inside WSL. You need to forward port 5001 so they can communicate.

First, find your WSL IP address (inside WSL):

```bash
hostname -I
```

Note the IP (e.g., `172.21.159.120`).

Then open PowerShell **as Administrator** on Windows and run:

```powershell
netsh interface portproxy delete v4tov4 listenport=5001 listenaddress=0.0.0.0
netsh interface portproxy add v4tov4 listenport=5001 listenaddress=0.0.0.0 connectport=5001 connectaddress=<your-wsl-ip>
netsh advfirewall firewall add rule name="Flask Server In" dir=in action=allow protocol=TCP localport=5001
netsh advfirewall firewall add rule name="Flask Server Out" dir=out action=allow protocol=TCP localport=5001
```

Replace `<your-wsl-ip>` with the IP from above. This only needs to be done once (unless the WSL IP changes after a reboot -- re-run if connectivity breaks).

### Step 9: Run the backend

Inside WSL, with the virtual environment activated:

```bash
cd ~/llma-rag
source rag2/bin/activate
python app-unity.py
```

The server starts on port 5001. On first launch, the HuggingFace sentence-transformers model (~130 MB) will be downloaded and cached automatically.

Verify it's working by opening a browser on Windows and going to:

```
http://localhost:5001/api/health
```

You should see `{"status": "healthy", ...}`.

## Running After Initial Setup

Every time you want to start the server:

```bash
wsl
cd ~/llma-rag
source rag2/bin/activate
python app-unity.py
```

## API Endpoints

All endpoints are prefixed with `/api`. Server runs on port `5001`.

### POST `/api/generator` (streaming)

Primary endpoint for Unity. Streams the LLM response token-by-token via Server-Sent Events (SSE).

**Request:**
```json
{ "query": "How do I start the generator?" }
```

**Response:** SSE stream where each event is:
```json
{ "text": "token", "figures": [{ "filename": "Fuel_System.png", "title": "Fuel System", "score": 0.72 }] }
```

Stream ends with `data: [DONE]`.

### POST `/api/query` (non-streaming)

Fallback endpoint that returns the full response at once.

**Request:**
```json
{ "query": "How do I start the generator?" }
```

**Response:**
```json
{
  "answer": "The full response text...",
  "figures": [{ "filename": "Fuel_System.png", "title": "Fuel System", "score": 0.72 }]
}
```

### GET `/api/figures`

Returns all available figures.

### GET `/api/health`

Health check. Returns `{ "status": "healthy" }` when the server is ready.

## Unity Integration

The `unity/` folder contains C# scripts for the Unity client:

| Script | Purpose |
|---|---|
| `RAGClient.cs` | Sends queries to the API, handles SSE streaming, loads figure textures |
| `FigureManager.cs` | Manages figure lifecycle and display ordering by relevance score |
| `FigureDisplay.cs` | UI component for rendering a figure with title and relevance score |
| `Figure.cs` | Data model for figure information |

In `RAGClient.cs`, update the `apiUrl` field to point at the backend:

```csharp
[SerializeField] private string apiUrl = "http://localhost:5001/api";
```

If running Unity on a different machine on the same network, use the Windows machine's LAN IP instead of `localhost`.

## Project Structure

```
llma-rag/
├── app-unity.py              # Flask API server for Unity
├── app.py                    # Flask web interface (standalone)
├── process_pdf.py            # Generates FAISS embeddings from PDF
├── extract_images_from_pdf.py # Extracts figures from PDF
├── requirements.txt          # Python dependencies
├── figures.json              # Figure metadata (title, filename)
├── training_manual.pdf       # Source training manual
├── embeddings/               # FAISS vector index (pre-built)
│   ├── index.faiss
│   └── index.pkl
├── Figures/                  # Extracted figure images (PNG)
├── static/Figures/           # Figures served by Flask
├── unity/                    # Unity C# client scripts
│   ├── RAGClient.cs
│   ├── FigureManager.cs
│   ├── FigureDisplay.cs
│   └── Figure.cs
├── Assets/                   # Unity Android build config
└── setup_port_forward.ps1    # WSL port forwarding (PowerShell)
```

## Troubleshooting

**`nvidia-smi` not found in WSL**: Update your Windows NVIDIA drivers to the latest version from https://www.nvidia.com/drivers. CUDA in WSL requires driver version 470.76 or higher.

**Port 5001 not reachable from Windows**: Re-run the port forwarding commands from Step 8. The WSL IP can change after a reboot. Run `hostname -I` inside WSL to get the current IP.

**Out of GPU memory**: Close other GPU-intensive applications. The model requires ~4 GB VRAM.

**HuggingFace download fails**: The sentence-transformers model is downloaded on first launch. Ensure WSL has internet access. If behind a proxy, configure it in WSL.

## Rebuilding Data from Scratch

If you need to regenerate the embeddings and figures from the PDF (not needed for normal deployment -- pre-built data is included in the repo):

```bash
python extract_images_from_pdf.py    # Extracts figures to Figures/
python process_pdf.py                # Builds FAISS index in embeddings/
```
