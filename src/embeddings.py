"""
High-Performance Semantic Embedding Engine for Customer Intent Discovery.

Features:
- Batched inference with progress tracking
- SHA-256 persistent disk and memory vector caching
- Multiple open-source model support (MiniLM, BGE, Multilingual)
- Memory-safe batch execution and L2 normalization
"""

import os
import sys
import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Union, Tuple
import numpy as np

# Suppress noisy TF/Torch warnings if present
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

_MODEL_CACHE: Dict[str, Any] = {}
CACHE_DIR = Path.home() / ".cache" / "intent_discovery_embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Fastest CPU inference, low memory, high quality for short queries (Recommended)",
        "multilingual": False
    },
    "bge-small-en-v1.5": {
        "name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "description": "High retrieval and clustering benchmark score for English text",
        "multilingual": False
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "description": "50+ language support for global/multilingual customer queries",
        "multilingual": True
    }
}


def get_model(model_key: str = "all-MiniLM-L6-v2"):
    """
    Get or load SentenceTransformer model instance (singleton pattern).
    """
    from sentence_transformers import SentenceTransformer
    import torch
    
    if model_key not in SUPPORTED_MODELS:
        model_name = model_key
    else:
        model_name = SUPPORTED_MODELS[model_key]["name"]
        
    if model_name not in _MODEL_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        _MODEL_CACHE[model_name] = model
        
    return _MODEL_CACHE[model_name]


def hash_text(text: str) -> str:
    """Compute SHA-256 hash of cleaned text string."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class DiskEmbeddingCache:
    """Lightweight persistent disk key-value store for embeddings."""
    
    def __init__(self, cache_folder: Path = CACHE_DIR):
        self.cache_folder = cache_folder
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        
    def _path_for_key(self, model_key: str, text_hash: str) -> Path:
        safe_model = model_key.replace("/", "_").replace("\\", "_")
        subfolder = self.cache_folder / safe_model
        subfolder.mkdir(parents=True, exist_ok=True)
        return subfolder / f"{text_hash}.pkl"

    def get_many(self, model_key: str, hashes: List[str]) -> Tuple[Dict[str, np.ndarray], List[str]]:
        found = {}
        missing = []
        for h in hashes:
            p = self._path_for_key(model_key, h)
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        found[h] = pickle.load(f)
                except Exception:
                    missing.append(h)
            else:
                missing.append(h)
        return found, missing

    def put_many(self, model_key: str, items: Dict[str, np.ndarray]):
        for h, vec in items.items():
            p = self._path_for_key(model_key, h)
            try:
                with open(p, "wb") as f:
                    pickle.dump(vec, f)
            except Exception:
                pass


def compute_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    normalize: bool = True,
    use_cache: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> np.ndarray:
    """
    Generate normalized embeddings for an iterable of customer queries.
    Uses caching to avoid recomputing identical queries.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    cache = DiskEmbeddingCache() if use_cache else None
    text_hashes = [hash_text(t) for t in texts]
    
    cached_vectors: Dict[str, np.ndarray] = {}
    missing_hashes: List[str] = text_hashes
    
    if cache is not None:
        cached_vectors, missing_hashes = cache.get_many(model_name, text_hashes)
        
    # Map missing hashes back to original texts
    hash_to_text: Dict[str, str] = {}
    for t, h in zip(texts, text_hashes):
        if h in missing_hashes and h not in hash_to_text:
            hash_to_text[h] = t
            
    unique_missing_hashes = list(hash_to_text.keys())
    unique_missing_texts = [hash_to_text[h] for h in unique_missing_hashes]
    
    if unique_missing_texts:
        if progress_callback:
            progress_callback(0.1, f"Loading model '{model_name}'...")
            
        model = get_model(model_name)
        total_missing = len(unique_missing_texts)
        newly_computed: Dict[str, np.ndarray] = {}
        
        for i in range(0, total_missing, batch_size):
            batch_texts = unique_missing_texts[i:i + batch_size]
            batch_hashes = unique_missing_hashes[i:i + batch_size]
            
            # Encode batch
            batch_embs = model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            
            for h, vec in zip(batch_hashes, batch_embs):
                newly_computed[h] = vec
                cached_vectors[h] = vec
                
            if progress_callback:
                pct = 0.1 + 0.8 * min(1.0, (i + len(batch_texts)) / total_missing)
                progress_callback(pct, f"Embedding queries: {min(i + batch_size, total_missing)}/{total_missing}")
                
        if cache is not None and newly_computed:
            cache.put_many(model_name, newly_computed)
            
    if progress_callback:
        progress_callback(1.0, "Embeddings ready.")
        
    # Assemble final aligned matrix
    embeddings = np.array([cached_vectors[h] for h in text_hashes], dtype=np.float32)
    return embeddings
