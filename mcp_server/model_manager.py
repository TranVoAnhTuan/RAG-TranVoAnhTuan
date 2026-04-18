import gc
import torch
from contextlib import contextmanager
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from transformers import AutoModel
from mcp_server.config import mcp_settings


class ModelManager:
    """
    Manages Dense, Sparse, and Reranker models for the MCP server.

    Strategy:
    - Sparse model stays on CPU permanently (low memory cost).
    - Dense and Reranker models live on CPU and are temporarily moved to GPU
      only during inference, then immediately released back to CPU.
    """

    def __init__(self):
        self._dense_model = None
        self._sparse_model = None
        self._reranker_model = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _print_vram(self, step_name: str):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(
                f"📊 [VRAM] {step_name} | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB"
            )

    def _clear_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.memory._record_memory_history(enabled=None)

    # ------------------------------------------------------------------
    # Sparse model (CPU only, loaded once)
    # ------------------------------------------------------------------

    def get_sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            print("🚀 [MCP] Loading Sparse Model (CPU)…")
            self._sparse_model = SparseTextEmbedding(
                model_name=mcp_settings.SPARSE_MODEL_NAME
            )
        return self._sparse_model

    # ------------------------------------------------------------------
    # Dense model (CPU-resident; borrowed to GPU during inference)
    # ------------------------------------------------------------------

    def _load_dense_model(self) -> SentenceTransformer:
        if self._dense_model is None:
            print("🚀 [MCP] Loading Dense Model into CPU memory…")
            self._dense_model = SentenceTransformer(
                mcp_settings.DENSE_MODEL_NAME,
                trust_remote_code=True,
                device="cpu",
                model_kwargs={"torch_dtype": torch.float16},
            )
        return self._dense_model

    @contextmanager
    def use_dense_model_on_gpu(self):
        """Temporarily move Dense Model to GPU, yield it, then move back to CPU."""
        print("\n" + "=" * 50)
        self._print_vram("1. BEFORE LOADING DENSE")
        model = self._load_dense_model()
        model.to("cuda")
        torch.cuda.empty_cache()
        self._print_vram("2. USING DENSE (PEAK VRAM)")
        try:
            yield model
        finally:
            print("🧹 [MCP] Unloading Dense Model from VRAM…")
            model.to("cpu")
            torch.cuda.synchronize()
            self._clear_vram()
            self._print_vram("3. AFTER CLEANING DENSE")
            print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    # Reranker model (CPU-resident; borrowed to GPU during inference)
    # ------------------------------------------------------------------

    def _load_reranker_model(self) -> AutoModel:
        if self._reranker_model is None:
            print("🚀 [MCP] Loading Jina Reranker into CPU memory…")
            self._reranker_model = AutoModel.from_pretrained(
                mcp_settings.RERANKER_MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=None,
                low_cpu_mem_usage=True,
            ).to("cpu").eval()
        return self._reranker_model

    @contextmanager
    def use_reranker_model(self):
        """Temporarily move Reranker to GPU, yield it, then move back to CPU."""
        print("\n" + "=" * 50)
        self._print_vram("1. BEFORE LOADING RERANKER")
        model = self._load_reranker_model()
        model.to("cuda")
        torch.cuda.empty_cache()
        self._print_vram("2. USING RERANKER (PEAK VRAM)")
        try:
            yield model
        finally:
            print("🧹 [MCP] Removing Reranker from VRAM…")
            model.to("cpu")
            torch.cuda.synchronize()
            self._clear_vram()
            self._print_vram("3. AFTER CLEANING RERANKER")
            print("=" * 50 + "\n")


# Singleton — shared across the MCP server process
model_manager = ModelManager()
