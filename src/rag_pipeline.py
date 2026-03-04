"""
rag_pipeline.py - Retrieval-Augmented Generation Pipeline

WHY THIS FILE EXISTS:
- Implements RAG: Retrieve relevant context, then Generate with LLM
- LLM doesn't "know" market rates - we give it relevant data
- This approach is more accurate than pure LLM and cheaper than fine-tuning

HOW RAG WORKS:
1. User asks: "Is £65/day fair for a Group C car in London?"
2. RETRIEVE: Search vector store for similar rate records
3. AUGMENT: Build a prompt that includes the retrieved rates
4. GENERATE: LLM answers based on actual market data

WHY RAG INSTEAD OF FINE-TUNING:
- Fine-tuning: Train model on all 65M rates (expensive, inflexible)
- RAG: Store rates in vector DB, retrieve relevant ones (cheap, updatable)
- RAG also provides citations - we can show which rates informed the decision

COMPONENTS USED:
- VectorStore: For retrieving similar documents
- BedrockClient: For LLM generation
- Prompts: Carefully crafted instructions for the LLM
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Import from sibling modules using package-relative imports
from src.llm_client import BedrockClient, get_llm_client, Message
from src.vector_store import VectorStore, get_rates_store, SearchResult
from src.config import settings

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """
    Context retrieved for a RAG query.
    
    Attributes:
        query: The original user query
        retrieved_documents: Documents found by similarity search
        context_text: Formatted text to include in LLM prompt
        source_ids: IDs of documents used (for citations)
    """
    query: str
    retrieved_documents: List[SearchResult] = field(default_factory=list)
    context_text: str = ""
    source_ids: List[str] = field(default_factory=list)


@dataclass
class RAGResponse:
    """
    Response from the RAG pipeline.
    
    Attributes:
        answer: The LLM's generated response
        context: The retrieved context used
        sources: List of source documents (for transparency)
        success: Whether the operation succeeded
        error: Error message if failed
    """
    answer: str
    context: RAGContext
    sources: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    
    USAGE:
        rag = RAGPipeline()
        
        response = rag.query(
            "Is £65/day fair for a Group C vehicle in London?"
        )
        
        print(response.answer)
        print(f"Based on {len(response.sources)} rate records")
    
    THE RAG PROCESS:
    1. Take user's question
    2. Search vector store for relevant documents
    3. Format documents into context string
    4. Build prompt: system instructions + context + question
    5. Send to LLM
    6. Return answer with sources
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm_client: Optional[BedrockClient] = None,
        top_k: int = None,
        min_similarity: float = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Vector store for retrieval (default: rates store)
            llm_client: LLM client for generation (default: Bedrock)
            top_k: Number of documents to retrieve
            min_similarity: Minimum similarity threshold
        """
        self.vector_store = vector_store or get_rates_store()
        self.llm_client = llm_client or get_llm_client()
        self.top_k = top_k or settings.rag_top_k
        self.min_similarity = min_similarity or settings.rag_similarity_threshold
        
        logger.info(f"RAG Pipeline initialized (top_k={self.top_k})")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RAGContext:
        """
        Retrieve relevant documents for a query.
        
        This is the "R" in RAG - Retrieval.
        
        Args:
            query: The user's question
            top_k: Override default number of results
            filters: Metadata filters (e.g., {"vehicle_group": "C"})
            
        Returns:
            RAGContext with retrieved documents and formatted context
            
        EXAMPLE:
            context = rag.retrieve(
                query="Group C hire rates in London",
                filters={"vehicle_group": "C", "region": "London"}
            )
            print(context.context_text)
        """
        k = top_k or self.top_k
        
        # Search the vector store
        search_results = self.vector_store.search(
            query=query,
            top_k=k,
            where=filters,
            min_similarity=self.min_similarity
        )
        
        # Format retrieved documents into context text
        context_parts = []
        source_ids = []
        
        for i, result in enumerate(search_results.results, start=1):
            # Format each document with its metadata
            doc_text = f"[Source {i}] {result.document}"
            
            # Add relevant metadata
            if result.metadata:
                meta_str = ", ".join(
                    f"{k}: {v}" for k, v in result.metadata.items()
                    if k not in ["embedding"]  # Exclude embedding data
                )
                if meta_str:
                    doc_text += f" ({meta_str})"
            
            context_parts.append(doc_text)
            source_ids.append(result.id)
        
        context_text = "\n".join(context_parts)
        
        return RAGContext(
            query=query,
            retrieved_documents=search_results.results,
            context_text=context_text,
            source_ids=source_ids
        )
    
    def generate(
        self,
        query: str,
        context: RAGContext,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response using LLM with retrieved context.
        
        This is the "G" in RAG - Generation.
        
        Args:
            query: The user's question
            context: Retrieved context from retrieve()
            system_prompt: Custom system instructions (optional)
            
        Returns:
            The LLM's generated response
        """
        # Default system prompt for rate analysis
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        # Build the user prompt with context
        user_prompt = self._build_user_prompt(query, context)
        
        # Generate response
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1  # Low temperature for factual responses
        )
        
        if not response.success:
            logger.error(f"LLM generation failed: {response.error}")
            return f"Error generating response: {response.error}"
        
        return response.text
    
    def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> RAGResponse:
        """
        Full RAG pipeline: retrieve + generate.
        
        This is the MAIN METHOD you'll use.
        
        Args:
            query: The user's question
            filters: Optional metadata filters for retrieval
            system_prompt: Optional custom system prompt
            
        Returns:
            RAGResponse with answer and sources
            
        EXAMPLE:
            response = rag.query(
                "Is £65/day fair for a Group C vehicle in London?",
                filters={"vehicle_group": "C"}
            )
            
            print("Answer:", response.answer)
            print("Sources used:", len(response.sources))
        """
        try:
            # Step 1: Retrieve relevant documents
            context = self.retrieve(query, filters=filters)
            
            # Check if we found any relevant documents
            if not context.retrieved_documents:
                return RAGResponse(
                    answer="I couldn't find any relevant rate data to answer this question. "
                           "Please ensure the rate database has been populated.",
                    context=context,
                    success=True
                )
            
            # Step 2: Generate response with context
            answer = self.generate(query, context, system_prompt)
            
            # Step 3: Format sources for transparency
            sources = [
                {
                    "id": doc.id,
                    "text": doc.document,
                    "metadata": doc.metadata,
                    "similarity": doc.similarity
                }
                for doc in context.retrieved_documents
            ]
            
            return RAGResponse(
                answer=answer,
                context=context,
                sources=sources,
                success=True
            )
            
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            return RAGResponse(
                answer="",
                context=RAGContext(query=query),
                success=False,
                error=str(e)
            )
    
    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for rate analysis.
        
        This prompt instructs the LLM on how to analyze rates.
        """
        return """You are an expert motor insurance claims analyst specializing in 
credit hire rate analysis. Your job is to determine if claimed hire rates are 
fair compared to market rates.

GUIDELINES:
1. Base your analysis ONLY on the provided rate data
2. Compare the claimed rate against the market rates shown
3. Consider vehicle group, region, and time period
4. Provide a clear verdict: FAIR, POTENTIALLY_INFLATED, or FLAGGED
5. Explain your reasoning with specific numbers
6. If data is insufficient, say so clearly

VERDICTS:
- FAIR: Rate is within normal market range (±15% of average)
- POTENTIALLY_INFLATED: Rate is 15-40% above market average
- FLAGGED: Rate is >40% above market average or has other concerns

Always cite the source numbers when referencing specific rates."""
    
    def _build_user_prompt(self, query: str, context: RAGContext) -> str:
        """
        Build the user prompt with context.
        
        This combines the query with retrieved documents.
        """
        prompt = f"""Based on the following market rate data:

--- MARKET RATE DATA ---
{context.context_text}
--- END DATA ---

USER QUESTION:
{query}

Please analyze and provide your assessment."""
        
        return prompt
    
    def query_with_chat(
        self,
        messages: List[Dict[str, str]],
        filters: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        RAG query with conversation history.
        
        For multi-turn conversations where context from previous
        messages matters.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            filters: Optional metadata filters
            
        Returns:
            RAGResponse with answer
        """
        # Get the latest user message
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            return RAGResponse(
                answer="No user message found.",
                context=RAGContext(query=""),
                success=False,
                error="No user message"
            )
        
        latest_query = user_messages[-1]["content"]
        
        # Retrieve context for the latest query
        context = self.retrieve(latest_query, filters=filters)
        
        # Build messages with system prompt and context
        chat_messages = [
            Message("system", self._get_default_system_prompt()),
            Message("system", f"Relevant market data:\n{context.context_text}")
        ]
        
        # Add conversation history
        for msg in messages:
            chat_messages.append(Message(msg["role"], msg["content"]))
        
        # Generate with chat
        response = self.llm_client.chat(chat_messages)
        
        sources = [
            {"id": doc.id, "text": doc.document, "metadata": doc.metadata}
            for doc in context.retrieved_documents
        ]
        
        return RAGResponse(
            answer=response.text if response.success else "",
            context=context,
            sources=sources,
            success=response.success,
            error=response.error
        )


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------
_default_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Get the default RAG pipeline (singleton).
    
    USAGE:
        from rag_pipeline import get_rag_pipeline
        
        rag = get_rag_pipeline()
        response = rag.query("Is this rate fair?")
    """
    global _default_pipeline
    
    if _default_pipeline is None:
        _default_pipeline = RAGPipeline()
    
    return _default_pipeline


def rag_query(query: str, filters: Optional[Dict[str, Any]] = None) -> RAGResponse:
    """
    Quick function for RAG queries.
    
    USAGE:
        from rag_pipeline import rag_query
        
        response = rag_query("Is £65/day fair for Group C in London?")
        print(response.answer)
    """
    pipeline = get_rag_pipeline()
    return pipeline.query(query, filters)