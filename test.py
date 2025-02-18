from llama_cpp import Llama
import torch

print("PyTorch CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

# Load LLaMA and ensure it's offloading to GPU
model = Llama(
    model_path="/home/mko0/RAG/mistral-7b-instruct-v0.1.Q4_0.gguf",
    n_ctx=2048,
    n_gpu_layers=30,  # Try using *all* layers on GPU
    use_mlock=True,
    verbose=True
)

# Run a simple inference to test GPU usage
output = model("Hello, how are you?", max_tokens=50)
print(output["choices"][0]["text"])
output = model("Hello, how are you?", max_tokens=50)
print(output["choices"][0]["text"])