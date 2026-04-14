import torch
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from app.core.config import settings
from transformers import BitsAndBytesConfig
import gc
from contextlib import contextmanager

class ModelManager:
    def __init__(self):
        self._dense_model = None
        self._sparse_model = None
        self._tokenizer = None

        self._reranker_model = None

    def print_vram(self, step_name):
        """Helper function to print current VRAM usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f"📊 [VRAM] {step_name} | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

    def _clear_vram(self):
        """Internal function to force VRAM garbage collection immediately"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.memory._record_memory_history(enabled=None)

    def get_dense_model(self):
        if self._dense_model is None:
            print("🚀 Loading Dense Model into VRAM...")
            self._dense_model = SentenceTransformer(settings.DENSE_MODEL_NAME, trust_remote_code=True, device="cpu", model_kwargs={"torch_dtype": torch.float16})
        return self._dense_model
    
    @contextmanager
    def use_dense_model_on_gpu(self):
        """Load Dense Model onto GPU, automatically clean up after use"""
        print("\n" + "="*50)
        self.print_vram("1. BEFORE LOADING DENSE")
        print("⏳ [GPU] Loading Dense Model into VRAM...")

        # Move model from CPU to GPU
        if self._dense_model is None:
            self.get_dense_model() 
        
        self._dense_model.to("cuda")
        torch.cuda.empty_cache()
        self.print_vram("2. USING DENSE (PEAK VRAM)")
        
        try:
            yield self._dense_model
        finally:
            print("🧹 [GPU] Unloading Dense Model from VRAM...")
            self._dense_model.to("cpu") 
            torch.cuda.synchronize()
            self._clear_vram()
            self.print_vram("3. AFTER CLEANING DENSE")
            print("="*50 + "\n")

    def get_sparse_model(self):
        if self._sparse_model is None:
            print("🚀 Loading Sparse Model...")
            self._sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)
        return self._sparse_model
    
    def get_tokenizer(self):
        if self._tokenizer is None:
            print("🚀 Loading Tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(settings.DENSE_MODEL_NAME, trust_remote_code=True, device= "cpu")
        return self._tokenizer

    def get_reranker_model(self):
        if self._reranker_model is None:
            print("🚀 Loading Jina Reranker v3 into VRAM...")
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=True, 
            #     bnb_4bit_compute_dtype=torch.float16, 
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4"
            # )
            
            # torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            self._reranker_model = AutoModel.from_pretrained(
                settings.RERANKER_MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                # quantization_config=quantization_config,
                device_map=None,
                low_cpu_mem_usage=True,
            ).to("cpu").eval()
            
        return self._reranker_model
    
    @contextmanager
    def use_reranker_model(self):
        """Load Reranker onto GPU, automatically clean up after use"""
        print("\n" + "="*50)
        self.print_vram("1. BEFORE LOADING RERANKER")
        print("⏳ [GPU] Loading Jina Reranker into VRAM...")
        if self._reranker_model is None:
            self.get_reranker_model()
        self._reranker_model.to("cuda")
        torch.cuda.empty_cache()
        self.print_vram("2. USING RERANKER (PEAK VRAM)")
        try:
            yield self._reranker_model
        finally:
            print("🧹 [GPU] Removing Jina Reranker from VRAM...")
            self._reranker_model.to("cpu")
            
            torch.cuda.synchronize()      
            self._clear_vram()

            self.print_vram("3. AFTER CLEANING RERANKER")
            print("="*50 + "\n")
    
model_manager = ModelManager()    
