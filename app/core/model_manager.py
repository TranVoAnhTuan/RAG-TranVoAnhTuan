"""GPU-aware model lifecycle manager for the FastAPI backend.

Manages Dense, Sparse, Reranker, and Tokenizer models.

Design Patterns
---------------
- **Singleton**: module-level ``model_manager`` instance.
- **Template Method**: ``_borrow_gpu`` encapsulates the shared
  GPU borrow → yield → release lifecycle used by both the dense
  encoder and the reranker.
"""

import gc
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML model loading and GPU memory for the FastAPI backend.

    Strategy:
    - Sparse model stays on CPU permanently (low memory cost).
    - Dense and Reranker models live on CPU and are temporarily moved to GPU
      only during inference, then immediately released back to CPU.
    """

    def __init__(self) -> None:
        self._dense_model: SentenceTransformer | None = None
        self._sparse_model: SparseTextEmbedding | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._reranker_model: AutoModel | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_vram(self, step_name: str) -> None:
        """Log current VRAM usage for debugging."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            logger.info(f"📊 [VRAM] {step_name} | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

    def _clear_vram(self) -> None:
        """Force VRAM garbage collection immediately."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.memory._record_memory_history(enabled=None)

    # ------------------------------------------------------------------
    # Template Method: GPU borrow/release lifecycle
    # ------------------------------------------------------------------

    @contextmanager
    def _borrow_gpu(self, model_name: str, load_fn: Callable) -> Iterator:
        """Template Method for the GPU borrow/release lifecycle.

        Steps: log → load → move to GPU → yield → move to CPU → clear VRAM.
        Concrete context managers (``use_dense_model_on_gpu``,
        ``use_reranker_model``) delegate here to avoid duplication.
        """
        logger.info("=" * 50)
        self._log_vram(f"1. BEFORE LOADING {model_name}")

        model = load_fn()
        model.to("cuda")
        torch.cuda.empty_cache()
        self._log_vram(f"2. USING {model_name} (PEAK VRAM)")

        try:
            yield model
        finally:
            logger.info(f"🧹 [GPU] Unloading {model_name} from VRAM...")
            model.to("cpu")
            torch.cuda.synchronize()
            self._clear_vram()
            self._log_vram(f"3. AFTER CLEANING {model_name}")
            logger.info("=" * 50)

    # ------------------------------------------------------------------
    # Sparse model (CPU only, loaded once)
    # ------------------------------------------------------------------

    def get_sparse_model(self) -> SparseTextEmbedding:
        """Return the sparse embedding model, loading on first call."""
        if self._sparse_model is None:
            logger.info("🚀 Loading Sparse Model...")
            self._sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)
        return self._sparse_model

    # ------------------------------------------------------------------
    # Dense model (CPU-resident; borrowed to GPU during inference)
    # ------------------------------------------------------------------

    def get_dense_model(self) -> SentenceTransformer:
        """Return the dense embedding model, loading into CPU on first call."""
        if self._dense_model is None:
            logger.info("🚀 Loading Dense Model into CPU memory...")
            self._dense_model = SentenceTransformer(
                settings.DENSE_MODEL_NAME,
                trust_remote_code=True,
                device="cpu",
                model_kwargs={"torch_dtype": torch.float16},
            )
        return self._dense_model

    @contextmanager
    def use_dense_model_on_gpu(self) -> Iterator[SentenceTransformer]:
        """Temporarily move Dense Model to GPU, yield it, then release."""
        with self._borrow_gpu("DENSE", self.get_dense_model) as model:
            yield model

    # ------------------------------------------------------------------
    # Tokenizer (CPU only)
    # ------------------------------------------------------------------

    def get_tokenizer(self) -> AutoTokenizer:
        """Return the tokenizer, loading on first call."""
        if self._tokenizer is None:
            logger.info("🚀 Loading Tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.DENSE_MODEL_NAME, trust_remote_code=True, device="cpu"
            )
        return self._tokenizer

    # ------------------------------------------------------------------
    # Reranker model (CPU-resident; borrowed to GPU during inference)
    # ------------------------------------------------------------------

    def get_reranker_model(self) -> AutoModel:
        """Return the reranker model, loading into CPU on first call."""
        if self._reranker_model is None:
            logger.info("🚀 Loading Jina Reranker v3 into CPU memory...")
            self._reranker_model = (
                AutoModel.from_pretrained(
                    settings.RERANKER_MODEL_NAME,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    device_map=None,
                    low_cpu_mem_usage=True,
                )
                .to("cpu")
                .eval()
            )
        return self._reranker_model

    @contextmanager
    def use_reranker_model(self) -> Iterator[AutoModel]:
        """Temporarily move Reranker to GPU, yield it, then release."""
        with self._borrow_gpu("RERANKER", self.get_reranker_model) as model:
            yield model


model_manager = ModelManager()
