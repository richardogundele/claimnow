"""
embeddings.py - Local Text Embeddings using Sentence Transformers

WHY THIS FILE EXISTS:
- Create vector embeddings from text locally (no OpenAI API)
- Embeddings are used for RAG - finding similar documents
- sentence-transformers library runs entirely on your machine

WHAT ARE EMBEDDINGS:
- A way to represent text as numbers (vectors)
- Similar meanings = similar vectors
- "car rental £50/day" and "vehicle hire £55 daily" have similar embeddings
- This enables semantic search (search by meaning, not keywords)

HOW IT WORKS:
1. Load a pre-trained transformer model
2. Pass text through the model
3. Get back a vector (list of numbers) representing the meaning
4. Store vectors in ChromaDB for similarity search
"""

import logging
from typing import List, Optional, Union
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

# Import from sibling modules using package-relative imports
from src.config import settings

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """
    Result of embedding one or more texts.
    
    Attributes:
        embeddings: The vector representations (numpy arrays)
        model: Which model was used
        dimension: Size of each embedding vector
    """
    embeddings: np.ndarray
    model: str
    dimension: int


class EmbeddingModel:
    """
    Wrapper around sentence-transformers for creating embeddings.
    
    USAGE:
        model = EmbeddingModel()
        
        # Single text
        vector = model.embed("Group C car hire in London")
        
        # Multiple texts (more efficient)
        vectors = model.embed_batch([
            "BMW 3 Series rental",
            "Mercedes C-Class hire",
            "Audi A4 lease"
        ])
    
    WHY SENTENCE-TRANSFORMERS:
    - Runs locally, no API calls
    - Models are small enough for laptops (80MB - 500MB)
    - Produces high-quality embeddings for semantic similarity
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name. Defaults to settings.
            
        COMMON MODELS:
        - "all-MiniLM-L6-v2": Small (80MB), fast, good quality
        - "all-mpnet-base-v2": Larger (420MB), better quality
        - "multi-qa-MiniLM-L6-cos-v1": Optimized for Q&A
        
        First run downloads the model to ~/.cache/torch/
        """
        self.model_name = model_name or settings.embedding_model
        
        logger.info(f"Loading embedding model: {self.model_name}")
        
        # Load the model - this downloads it on first run
        # device='cpu' ensures it works without GPU
        self._model = SentenceTransformer(
            self.model_name,
            device='cpu'  # Use CPU (works everywhere, GPU optional)
        )
        
        # Get the embedding dimension from the model
        self.dimension = self._model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Create an embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            numpy array of shape (dimension,) - e.g., (384,)
            
        EXAMPLE:
            vector = model.embed("Toyota Corolla hire £45/day London")
            print(vector.shape)  # (384,)
            print(vector[:5])    # [0.023, -0.156, 0.089, ...]
        """
        # encode() returns numpy array
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for multiple texts efficiently.
        
        WHY BATCH:
        - Processing multiple texts together is faster
        - GPU/CPU parallelization
        - Reduces overhead per text
        
        Args:
            texts: List of texts to embed
            batch_size: How many to process at once
            
        Returns:
            numpy array of shape (num_texts, dimension)
            
        EXAMPLE:
            texts = ["text1", "text2", "text3"]
            vectors = model.embed_batch(texts)
            print(vectors.shape)  # (3, 384)
        """
        if not texts:
            return np.array([])
        
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100  # Show progress for large batches
        )
        
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Returns a score from 0 to 1:
        - 1.0 = identical meaning
        - 0.0 = completely different
        - > 0.7 = quite similar
        - > 0.9 = very similar
        
        EXAMPLE:
            score = model.similarity(
                "BMW 3 Series rental London",
                "BMW 320d hire in London area"
            )
            print(score)  # 0.89 (very similar)
        """
        # Get embeddings for both texts
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        # Cosine similarity (dot product of normalized vectors)
        # Since vectors are normalized, dot product = cosine similarity
        similarity = np.dot(emb1, emb2)
        
        # Ensure it's in [0, 1] range
        return float(max(0, min(1, similarity)))
    
    def find_most_similar(
        self, 
        query: str, 
        candidates: List[str], 
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most similar texts from a list of candidates.
        
        Args:
            query: The text to find matches for
            candidates: List of texts to search through
            top_k: How many results to return
            
        Returns:
            List of (index, text, similarity_score) tuples, sorted by similarity
            
        EXAMPLE:
            results = model.find_most_similar(
                query="BMW saloon hire",
                candidates=[
                    "Mercedes C-Class rental",
                    "BMW 3 Series hire",
                    "Ford Transit van lease",
                    "BMW 5 Series rental"
                ],
                top_k=2
            )
            # Returns: [(1, "BMW 3 Series hire", 0.91), (3, "BMW 5 Series rental", 0.87)]
        """
        if not candidates:
            return []
        
        # Embed the query
        query_embedding = self.embed(query)
        
        # Embed all candidates
        candidate_embeddings = self.embed_batch(candidates)
        
        # Calculate similarities (dot product with normalized vectors)
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # Get indices of top-k highest similarities
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = [
            (int(idx), candidates[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results


# -----------------------------------------------------------------------------
# Singleton instance
# -----------------------------------------------------------------------------
_default_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """
    Get the default embedding model (singleton).
    
    WHY SINGLETON:
    - Model loading takes time (~2 seconds)
    - Model uses memory (~300MB)
    - Reuse the same instance everywhere
    """
    global _default_model
    
    if _default_model is None:
        _default_model = EmbeddingModel()
    
    return _default_model


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------
def embed_text(text: str) -> np.ndarray:
    """
    Quick function to embed a single text.
    
    USAGE:
        from embeddings import embed_text
        
        vector = embed_text("Group C vehicle hire London £65/day")
    """
    model = get_embedding_model()
    return model.embed(text)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Quick function to embed multiple texts.
    
    USAGE:
        from embeddings import embed_texts
        
        vectors = embed_texts(["text1", "text2", "text3"])
    """
    model = get_embedding_model()
    return model.embed_batch(texts)


def text_similarity(text1: str, text2: str) -> float:
    """
    Quick function to calculate similarity between two texts.
    
    USAGE:
        from embeddings import text_similarity
        
        score = text_similarity("car hire", "vehicle rental")
        print(score)  # ~0.85
    """
    model = get_embedding_model()
    return model.similarity(text1, text2)

