import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.model_manager import model_manager


class Qwen3Reranker:
    def __init__(self, model_name="Qwen/Qwen3-Reranker-0.6B", use_fp16=True, device="cpu"):
        
        self.device = device
            
        self.tokenizer = model_manager.get_reranker_tokenizer(model_name)
        self.model = model_manager.get_reranker_model(model_name, use_fp16)
        
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        # Bạn có thể đổi instruction này nều làm bài toán khác (ví dụ: code retrieval)
        self.instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    def _format_instruction(self, query, doc):
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs_text):
        inputs = self.tokenizer(
            pairs_text, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
            
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    @torch.no_grad()
    def compute_score(self, pairs):
        """
        pairs: list of [query, document]
        returns: list of float scores
        """
        # Xử lý trường hợp chỉ truyền vào 1 cặp thay vì list các cặp
        is_single = False
        if isinstance(pairs[0], str):
            pairs = [pairs]
            is_single = True
            
        pairs_text = [self._format_instruction(query, doc) for query, doc in pairs]
        inputs = self._process_inputs(pairs_text)
        
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        
        return scores[0] if is_single else scores