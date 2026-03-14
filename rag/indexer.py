"""
rag/indexer.py — Project file indexer using sentence-transformers + FAISS.

Indexes all code/text files in the project directory into a vector store.
At query time, retrieves the most relevant chunks to inject into the model's context.

Why RAG? Small 3B-7B models have limited context windows (4096 tokens).
We cannot fit an entire codebase. RAG selects only relevant fragments.
"""
import os
import pickle
import warnings
from pathlib import Path
from typing import List

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)  # suppress noisy warnings

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 400        # characters per chunk
CHUNK_OVERLAP = 80      # overlap between consecutive chunks (for continuity)
INDEX_DIR_NAME = ".agent_index"

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".cpp",
    ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt", ".html", ".css",
    ".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".sh", ".bash",
}

SKIP_DIRS = {
    "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".git", ".agent_index", ".idea", ".vscode",
}


class ProjectIndexer:
    """
    Builds and queries a FAISS vector index of project source files.

    Usage:
        indexer = ProjectIndexer(project_root="/path/to/project")
        if not indexer.load_index():
            indexer.build_index()
        results = indexer.retrieve("find all database connection functions", top_k=5)
    """

    def __init__(self, project_root: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.project_root = Path(project_root).resolve()
        self.index_dir = self.project_root / INDEX_DIR_NAME
        self.embedding_model_name = embedding_model
        self._embedder = None
        self._faiss_index = None
        self._metadata: List[dict] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False):
        """Walk the project, chunk files, embed them, and build a FAISS index."""
        self.index_dir.mkdir(exist_ok=True)

        chunks = self._collect_chunks()
        if not chunks:
            print("[Indexer] No files found to index.")
            return

        print(f"[Indexer] Indexing {len(chunks)} chunks from {self.project_root.name}/...")
        embedder = self._get_embedder()
        texts = [c["text"] for c in chunks]

        # Encode in batches (avoids OOM on large projects)
        embeddings = embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Build FAISS flat L2 index
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # Save index + metadata
        faiss.write_index(index, str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "metadata.pkl", "wb") as f:
            pickle.dump(chunks, f)

        self._faiss_index = index
        self._metadata = chunks
        print(f"[Indexer] ✓ Index saved to {INDEX_DIR_NAME}/ ({len(chunks)} chunks, dim={dim})")

    def load_index(self) -> bool:
        """Load an existing index from disk. Returns True if successful."""
        idx_path = self.index_dir / "faiss.index"
        meta_path = self.index_dir / "metadata.pkl"

        if not (idx_path.exists() and meta_path.exists()):
            return False

        try:
            import faiss
            self._faiss_index = faiss.read_index(str(idx_path))
            with open(meta_path, "rb") as f:
                self._metadata = pickle.load(f)
            print(f"[Indexer] Loaded existing index ({len(self._metadata)} chunks).")
            return True
        except Exception as e:
            print(f"[Indexer] Failed to load index: {e}. Will rebuild.")
            return False

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Find the top_k most relevant code chunks for a given query.

        Returns:
            List of dicts: {"file": str, "start": int, "text": str}
        """
        if self._faiss_index is None or not self._metadata:
            return []

        embedder = self._get_embedder()
        q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self._faiss_index.search(q_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._metadata):
                result = dict(self._metadata[idx])
                result["distance"] = float(dist)
                results.append(result)
        return results

    def format_context(self, results: List[dict]) -> str:
        """Format retrieved chunks into a readable context block for the prompt."""
        if not results:
            return ""
        blocks = []
        for r in results:
            approx_line = r["start"] // 40 + 1
            blocks.append(f"[{r['file']} ~line {approx_line}]\n{r['text']}")
        return "\n\n".join(blocks)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. Run:\n"
                    "  pip install sentence-transformers"
                )
            print(f"[Indexer] Loading embedding model '{self.embedding_model_name}'...")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _collect_chunks(self) -> List[dict]:
        """Walk the project and return all text chunks."""
        all_chunks = []
        for fpath in self.project_root.rglob("*"):
            # Skip ignored directories
            if any(part in SKIP_DIRS for part in fpath.parts):
                continue
            if fpath.name.startswith("."):
                continue
            if not fpath.is_file():
                continue
            if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                rel = str(fpath.relative_to(self.project_root))
                chunks = self._chunk_text(text, rel)
                all_chunks.extend(chunks)
            except Exception:
                pass
        return all_chunks

    def _chunk_text(self, text: str, file_path: str) -> List[dict]:
        """Split text into overlapping chunks."""
        chunks = []
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for start in range(0, len(text), step):
            chunk = text[start: start + CHUNK_SIZE]
            if chunk.strip():
                chunks.append({"file": file_path, "start": start, "text": chunk})
        return chunks
