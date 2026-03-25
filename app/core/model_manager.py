import torch
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from app.core.config import settings
from transformers import BitsAndBytesConfig


class ModelManager:
    def __init__(self):
        self._dense_model = None
        self._sparse_model = None
        self._tokenizer = None

        self._reranker_model = None
        # self._reranker_model = None
        # self._reranker_tokenizer = None

    def get_dense_model(self):
        if self._dense_model is None:
            print("🚀 Loading Dense Model into VRAM...")
            self._dense_model = SentenceTransformer(settings.DENSE_MODEL_NAME, trust_remote_code=True, device="cpu")
        return self._dense_model

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
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # device = "cpu"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, # Tải ở chế độ 4-bit
                bnb_4bit_compute_dtype=torch.float16, # Tính toán nội bộ bằng fp16 để giữ tốc độ
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Ép kiểu float16 trên GPU để tiết kiệm VRAM
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            self._reranker_model = AutoModel.from_pretrained(
                'jinaai/jina-reranker-v3',
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                quantization_config=quantization_config,
            ).to(device).eval()
            
        return self._reranker_model
    
    # def get_reranker_tokenizer(self, model_name="Qwen/Qwen3-Reranker-0.6B"):
    #     if self._reranker_tokenizer is None:
    #         print("🚀 Loading Reranker Tokenizer...")
    #         self._reranker_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    #     return self._reranker_tokenizer

    # def get_reranker_model(self, model_name="Qwen/Qwen3-Reranker-0.6B", use_fp16=False, device="cpu"):
    #     if self._reranker_model is None:
    #         # Đổi thông báo để dễ theo dõi
    #         print("🚀 Loading Reranker Model into RAM (CPU)...") 
            
    #         # Ép cứng device là CPU để không dùng đến VRAM của GPU
    #         device = "cpu"
            
    #         # CPU xử lý float32 ổn định nhất. (Nếu thiếu RAM, có thể thử torch.bfloat16)
    #         torch_dtype = torch.float32 
            
    #         self._reranker_model = AutoModelForCausalLM.from_pretrained(
    #             model_name, 
    #             torch_dtype=torch_dtype
    #         ).to(device).eval()
            
    #     return self._reranker_model

# Khởi tạo một instance duy nhất (Singleton)
model_manager = ModelManager()