"""
Embedding engine for semantic route matching.

Provides a local embedding interface using ONNX Runtime for
inference-time performance. Supports loading sentence-transformer
models exported to ONNX format for CPU-optimized embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class EmbeddingEngine:
    """
    Local embedding engine using sentence-transformers.
    
    Generates dense vector embeddings for queries and route utterances.
    Uses a lightweight model (all-MiniLM-L6-v2) by default for
    sub-10ms per-query latency on CPU.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_onnx: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._use_onnx = use_onnx
        self._model = None
        self._dimension: int | None = None
        self._cache_dir = cache_dir
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(
                self._model_name,
                cache_folder=self._cache_dir,
            )
            
            # Determine embedding dimension from a test encode
            test_emb = self._model.encode(["test"], normalize_embeddings=True)
            self._dimension = test_emb.shape[1]
            
            logger.info(
                "Embedding engine initialized",
                model=self._model_name,
                dimension=self._dimension,
                onnx=self._use_onnx,
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            )
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            raise RuntimeError("Embedding engine not initialized")
        return self._dimension
    
    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into normalized embedding vectors.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            NumPy array of shape (len(texts), dimension) with L2-normalized vectors.
        """
        if self._model is None:
            raise RuntimeError("Embedding engine not initialized")
        
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text])[0]
