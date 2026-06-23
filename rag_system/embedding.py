# embedding.py
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


class EmbeddingModel:
    """Wrapper around HuggingFace encoder model for embeddings."""
    
    def __init__(self, model_name: str, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device = torch.device(device)  
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        print(f"[Embedder] model param device = {next(self.model.parameters()).device}")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
        sum_mask = input_mask_expanded.sum(1)
        return sum_embeddings / sum_mask

    def embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return self.mean_pooling(outputs, inputs["attention_mask"]).cpu()

    def batch_embed(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.mean_pooling(outputs, inputs["attention_mask"])
            return embeddings.cpu()

