import torch
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from app.core.config import settings
from transformers import BitsAndBytesConfig
import gc
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self._dense_model = None
        self._sparse_model = None
        self._tokenizer = None

        self._reranker_model = None

    def log_vram(self, step_name):
        """Helper function to log current VRAM usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            logger.info(f"📊 [VRAM] {step_name} | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

    def _clear_vram(self):
        """Internal function to force VRAM garbage collection immediately"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.memory._record_memory_history(enabled=None)

    def get_dense_model(self):
        if self._dense_model is None:
            logger.info("🚀 Loading Dense Model into VRAM...")
            self._dense_model = SentenceTransformer(settings.DENSE_MODEL_NAME, trust_remote_code=True, device="cpu", model_kwargs={"torch_dtype": torch.float16})
        return self._dense_model
    
    @contextmanager
    def use_dense_model_on_gpu(self):
        """Load Dense Model onto GPU, automatically clean up after use"""
        logger.info("="*50)
        self.log_vram("1. BEFORE LOADING DENSE")
        logger.info("⏳ [GPU] Loading Dense Model into VRAM...")

        # Move model from CPU to GPU
        if self._dense_model is None:
            self.get_dense_model() 
        
        self._dense_model.to("cuda")
        torch.cuda.empty_cache()
        self.log_vram("2. USING DENSE (PEAK VRAM)")
        
        try:
            yield self._dense_model
        finally:
            logger.info("🧹 [GPU] Unloading Dense Model from VRAM...")
            self._dense_model.to("cpu") 
            torch.cuda.synchronize()
            self._clear_vram()
            self.log_vram("3. AFTER CLEANING DENSE")
            logger.info("="*50)

    def get_sparse_model(self):
        if self._sparse_model is None:
            logger.info("🚀 Loading Sparse Model...")
            self._sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)
        return self._sparse_model
    
    def get_tokenizer(self):
        if self._tokenizer is None:
            logger.info("🚀 Loading Tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(settings.DENSE_MODEL_NAME, trust_remote_code=True, device= "cpu")
        return self._tokenizer

    def get_reranker_model(self):
        if self._reranker_model is None:
            logger.info("🚀 Loading Jina Reranker v3 into VRAM...")
            
            self._reranker_model = AutoModel.from_pretrained(
                settings.RERANKER_MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=None,
                low_cpu_mem_usage=True,
            ).to("cpu").eval()
            
        return self._reranker_model
    
    @contextmanager
    def use_reranker_model(self):
        """Load Reranker onto GPU, automatically clean up after use"""
        logger.info("="*50)
        self.log_vram("1. BEFORE LOADING RERANKER")
        logger.info("⏳ [GPU] Loading Jina Reranker into VRAM...")
        if self._reranker_model is None:
            self.get_reranker_model()
        self._reranker_model.to("cuda")
        torch.cuda.empty_cache()
        self.log_vram("2. USING RERANKER (PEAK VRAM)")
        try:
            yield self._reranker_model
        finally:
            logger.info("🧹 [GPU] Removing Jina Reranker from VRAM...")
            self._reranker_model.to("cpu")
            
            torch.cuda.synchronize()      
            self._clear_vram()

            self.log_vram("3. AFTER CLEANING RERANKER")
            logger.info("="*50)
    
model_manager = ModelManager()    
