from typing import Union, List
import numpy as np
from sentence_transformers import SentenceTransformer


class Qwen3EmbeddingModel:
    """Wrapper for Qwen3 Embedding Model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"attn_implementation": "flash_attention_2", "dtype": "auto", "device_map": device},
            tokenizer_kwargs={"padding_side": "left"},
        )
    
    def encode_text(self, texts: Union[str, List[str]], batch_size: int = 64, **kwargs) -> np.ndarray:
        """Encode text into embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, **kwargs)
        return embeddings
    
    def encode(self, content: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Universal encode method for text"""
        return self.encode_text(content, **kwargs)
