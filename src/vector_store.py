"""
vector_store.py - ChromaDB Vector Database Operations

WHY THIS FILE EXISTS:
- Store embeddings (vectors) for semantic search
- ChromaDB is a lightweight, local vector database
- No cloud services needed - runs entirely on your machine
- Persists data to disk so it survives restarts

WHAT IS A VECTOR DATABASE:
- Regular database: search by exact values (WHERE price = 50)
- Vector database: search by similarity (find similar to this vector)
- Used for RAG: find documents/rates similar to a query

HOW CHROMADB WORKS:
1. Create a "collection" (like a table)
2. Add documents with their embeddings
3. Query by providing a vector
4. ChromaDB returns the most similar documents

CHROMADB CONCEPTS:
- Collection: A group of related embeddings (like a table)
- Document: The original text
- Embedding: The vector representation
- Metadata: Extra info stored with each document (e.g., price, date)
- ID: Unique identifier for each document
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings, get_absolute_path
from embeddings import get_embedding_model, EmbeddingModel

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    A single search result from the vector store.
    
    Attributes:
        id: Unique identifier of the document
        document: The original text content
        metadata: Additional data stored with the document
        distance: How far from the query (lower = more similar)
        similarity: Similarity score (higher = more similar, 0-1)
    """
    id: str
    document: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0
    similarity: float = 1.0


@dataclass
class SearchResults:
    """
    Collection of search results.
    
    Attributes:
        results: List of SearchResult objects
        query: The original query text
        total_found: Number of results returned
    """
    results: List[SearchResult]
    query: str = ""
    total_found: int = 0


class VectorStore:
    """
    Interface to ChromaDB for storing and searching embeddings.
    
    USAGE:
        # Create/connect to a collection
        store = VectorStore("market_rates")
        
        # Add documents
        store.add_documents(
            documents=["BMW hire £50/day", "Audi rental £55/day"],
            metadatas=[{"group": "D"}, {"group": "D"}],
            ids=["rate_001", "rate_002"]
        )
        
        # Search
        results = store.search("luxury car rental", top_k=5)
        for r in results.results:
            print(f"{r.document} (similarity: {r.similarity:.2f})")
    """
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[Path] = None,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to use/create
            persist_directory: Where to store the database files
            embedding_model: Model to use for embeddings (uses default if None)
        """
        self.collection_name = collection_name
        
        # Use default paths if not provided
        self.persist_directory = persist_directory or get_absolute_path(
            settings.vectorstore_dir
        )
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Get embedding model
        self._embedding_model = embedding_model or get_embedding_model()
        
        # Initialize ChromaDB client
        # PersistentClient saves data to disk automatically
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False  # Disable telemetry
            )
        )
        
        # Get or create the collection
        # If collection exists, it's loaded; otherwise created
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"Vector store for {collection_name}"}
        )
        
        logger.info(
            f"VectorStore initialized: {collection_name} "
            f"({self._collection.count()} documents)"
        )
    
    @property
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self._collection.count()
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of text documents to add
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of unique IDs (auto-generated if not provided)
            
        Returns:
            List of IDs assigned to the documents
            
        EXAMPLE:
            ids = store.add_documents(
                documents=[
                    "Group C, London, £55/day, 2024",
                    "Group C, Manchester, £48/day, 2024"
                ],
                metadatas=[
                    {"vehicle_group": "C", "region": "London", "rate": 55.0},
                    {"vehicle_group": "C", "region": "Manchester", "rate": 48.0}
                ]
            )
        """
        if not documents:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            # Use hash of document + index for reproducible IDs
            import hashlib
            ids = [
                hashlib.md5(f"{doc}_{i}".encode()).hexdigest()[:16]
                for i, doc in enumerate(documents)
            ]
        
        # Generate embeddings for all documents
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self._embedding_model.embed_batch(documents)
        
        # Convert numpy arrays to lists (ChromaDB requirement)
        embeddings_list = embeddings.tolist()
        
        # Prepare metadatas (empty dicts if not provided)
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Add to ChromaDB
        self._collection.add(
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to {self.collection_name}")
        
        return ids
    
    def add_document(
        self,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> str:
        """
        Add a single document to the vector store.
        
        Convenience method that wraps add_documents for single items.
        """
        ids = self.add_documents(
            documents=[document],
            metadatas=[metadata] if metadata else None,
            ids=[id] if id else None
        )
        return ids[0]
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0
    ) -> SearchResults:
        """
        Search for similar documents.
        
        This is the MAIN SEARCH METHOD for RAG.
        
        Args:
            query: Text to search for
            top_k: Number of results to return
            where: Optional filter on metadata (e.g., {"region": "London"})
            min_similarity: Minimum similarity score (0-1) to include
            
        Returns:
            SearchResults with matching documents
            
        EXAMPLE:
            results = store.search(
                query="Group C vehicle hire London area",
                top_k=10,
                where={"vehicle_group": "C"}  # Filter to Group C only
            )
            
            for r in results.results:
                print(f"Rate: {r.metadata['rate']} - {r.document}")
        """
        # Generate embedding for the query
        query_embedding = self._embedding_model.embed(query)
        
        # Build query parameters
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        # Add metadata filter if provided
        if where:
            query_params["where"] = where
        
        # Execute search
        results = self._collection.query(**query_params)
        
        # Parse results
        # ChromaDB returns lists of lists (for batch queries)
        # We're doing single query, so take first element
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
            
            for i in range(len(ids)):
                # Convert distance to similarity
                # ChromaDB uses L2 distance by default
                # Lower distance = more similar
                # We convert to similarity score (0-1)
                distance = distances[i]
                similarity = 1 / (1 + distance)  # Simple conversion
                
                # Skip if below minimum similarity
                if similarity < min_similarity:
                    continue
                
                search_results.append(SearchResult(
                    id=ids[i],
                    document=documents[i],
                    metadata=metadatas[i],
                    distance=distance,
                    similarity=similarity
                ))
        
        return SearchResults(
            results=search_results,
            query=query,
            total_found=len(search_results)
        )
    
    def search_by_metadata(
        self,
        where: Dict[str, Any],
        limit: int = 100
    ) -> List[SearchResult]:
        """
        Search documents by metadata only (no semantic search).
        
        Useful for exact filters like "get all Group C rates".
        
        Args:
            where: Metadata filter
            limit: Maximum results to return
            
        EXAMPLE:
            results = store.search_by_metadata(
                where={"vehicle_group": "C", "year": 2024}
            )
        """
        results = self._collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        search_results = []
        
        if results["ids"]:
            for i in range(len(results["ids"])):
                search_results.append(SearchResult(
                    id=results["ids"][i],
                    document=results["documents"][i] if results["documents"] else "",
                    metadata=results["metadatas"][i] if results["metadatas"] else {}
                ))
        
        return search_results
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from {self.collection_name}")
    
    def clear(self) -> None:
        """
        Delete ALL documents from the collection.
        
        WARNING: This cannot be undone!
        """
        # Get all IDs
        all_ids = self._collection.get()["ids"]
        
        if all_ids:
            self._collection.delete(ids=all_ids)
            logger.info(f"Cleared {len(all_ids)} documents from {self.collection_name}")
    
    def get_by_id(self, id: str) -> Optional[SearchResult]:
        """
        Get a specific document by its ID.
        
        Args:
            id: The document ID
            
        Returns:
            SearchResult if found, None otherwise
        """
        results = self._collection.get(
            ids=[id],
            include=["documents", "metadatas"]
        )
        
        if results["ids"]:
            return SearchResult(
                id=results["ids"][0],
                document=results["documents"][0] if results["documents"] else "",
                metadata=results["metadatas"][0] if results["metadatas"] else {}
            )
        
        return None
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> None:
        """
        Update the metadata for a document.
        
        Args:
            id: The document ID
            metadata: New metadata (replaces existing)
        """
        self._collection.update(
            ids=[id],
            metadatas=[metadata]
        )


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------
_stores: Dict[str, VectorStore] = {}


def get_vector_store(collection_name: str) -> VectorStore:
    """
    Get a vector store by name (cached).
    
    USAGE:
        from vector_store import get_vector_store
        
        rates_store = get_vector_store("market_rates")
        results = rates_store.search("Group C London")
    """
    global _stores
    
    if collection_name not in _stores:
        _stores[collection_name] = VectorStore(collection_name)
    
    return _stores[collection_name]


def get_rates_store() -> VectorStore:
    """
    Get the market rates vector store.
    
    This is the main store used for RAG rate matching.
    """
    return get_vector_store(settings.rates_collection_name)