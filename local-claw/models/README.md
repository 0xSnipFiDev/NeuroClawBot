# 📁 models/

Place your local GGUF model files here.

## Recommended Models

| Model | VRAM/RAM | Quality | Download |
|-------|----------|---------|----------|
| Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf | ~2 GB RAM | Good | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF) |
| Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf | ~4.5 GB RAM | Better | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) |
| DeepSeek-Coder-6.7B-Instruct-Q4_K_M.gguf | ~4 GB RAM | Excellent | [HuggingFace](https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF) |
| Phi-3-mini-4k-instruct-q4.gguf | ~2.2 GB RAM | Fast | [HuggingFace](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) |

## How to Download

### Using huggingface-cli (recommended)
```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct-GGUF \
    qwen2.5-coder-3b-instruct-q4_k_m.gguf \
    --local-dir ./models
```

### Direct curl
```bash
curl -L -o models/qwen2.5-coder-3b.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
```

## Auto-Detection
The agent automatically detects any `.gguf` file in this folder.
It prefers models with "coder", "instruct", or "deepseek" in the filename.
