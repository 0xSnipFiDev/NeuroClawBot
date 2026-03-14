"""
model/loader.py — Auto-detect and load local GGUF model using llama-cpp-python.
Place your .gguf model file in the models/ directory.
"""
import os
import glob
from pathlib import Path


class ModelLoader:
    """
    Automatically scans the models/ directory for a .gguf file and loads it
    via llama-cpp-python. Supports CPU and partial GPU offloading.
    """

    def __init__(
        self,
        models_dir: str = "./models",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        verbose: bool = False,
    ):
        self.models_dir = models_dir
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.verbose = verbose
        self.llm = None
        self._model_path: str | None = None

    def auto_detect_model(self) -> str:
        """Scan models/ and return the path to the best .gguf file found."""
        pattern = os.path.join(self.models_dir, "**", "*.gguf")
        models = glob.glob(pattern, recursive=True)
        if not models:
            raise FileNotFoundError(
                f"\n[ERROR] No .gguf model found in '{self.models_dir}'.\n"
                "Please download a model and place it there.\n"
                "Example: huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct-GGUF "
                "qwen2.5-coder-3b-instruct-q4_k_m.gguf --local-dir ./models\n"
            )
        # Prefer coding/instruct models
        preferred_keywords = ["coder", "instruct", "deepseek", "phi"]
        preferred = [
            m for m in models
            if any(k in Path(m).stem.lower() for k in preferred_keywords)
        ]
        selected = (preferred or models)[0]
        print(f"[ModelLoader] Auto-detected: {Path(selected).name}")
        return selected

    def load(self, model_path: str | None = None) -> "ModelLoader":
        """Load the model. Returns self for chaining."""
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed. Run:\n"
                "  pip install llama-cpp-python\n"
                "For GPU support: pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )

        self._model_path = model_path or self._model_path or self.auto_detect_model()
        if self.llm is not None:
            return self

        print(f"[ModelLoader] Loading model (n_ctx={self.n_ctx}, gpu_layers={self.n_gpu_layers}, n_threads={self.n_threads or 'auto'})...")

        kwargs = {
            "model_path": self._model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": self.verbose,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads

        self.llm = Llama(**kwargs)
        print(f"[ModelLoader] ✓ Model ready.")
        return self

    def unload(self):
        """Free memory by actively deleting the Llama instance."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            
            from core.system_optimizer import clear_memory
            clear_memory()
            
            print(f"[ModelLoader] 🧹 Unloaded from memory: {self.model_name}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
        repeat_penalty: float = 1.1,
    ) -> str:
        """Generate text from a prompt. Returns the generated string."""
        if self.llm is None:
            self.load()

        result = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=repeat_penalty,
            stop=stop or ["Observation:", "\nUser:", "Human:"],
            echo=False,
        )
        return result["choices"][0]["text"].strip()

    @property
    def model_name(self) -> str:
        if self._model_path:
            return Path(self._model_path).stem
        return "unknown"
