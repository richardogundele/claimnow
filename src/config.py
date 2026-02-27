"""
config.py - Central Configuration for ClaimsNOW

WHY THIS FILE EXISTS:
- All settings in ONE place (no magic numbers scattered in code)
- Easy to change settings without editing business logic
- Environment variables for sensitive data (API keys, paths)
- Pydantic validates settings automatically

WHAT PYDANTIC DOES:
- BaseSettings loads values from environment variables
- Validates types (ensures PORT is an int, not a string)
- Provides defaults when env vars are missing
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    HOW IT WORKS:
    1. Pydantic looks for environment variables matching field names
    2. If found, uses the env var value
    3. If not found, uses the default value specified here
    4. Type conversion is automatic (e.g., "8000" -> 8000 for int fields)
    
    EXAMPLE:
    If you set OLLAMA_MODEL=llama2 in your .env file,
    settings.ollama_model will be "llama2" instead of "mistral"
    """
    
    # -------------------------------------------------------------------------
    # Project Paths
    # -------------------------------------------------------------------------
    # BASE_DIR: The root folder of the project
    # All other paths are relative to this
    base_dir: Path = Field(
        default=Path(__file__).parent.parent,  # Goes up from src/ to project root
        description="Root directory of the project"
    )
    
    # Where to store uploaded documents temporarily
    upload_dir: Path = Field(
        default=Path("data/uploads"),
        description="Directory for uploaded PDF files"
    )
    
    # Where ChromaDB stores its vector database
    vectorstore_dir: Path = Field(
        default=Path("vectorstore"),
        description="Directory for ChromaDB persistence"
    )
    
    # Where trained ML models are saved
    models_dir: Path = Field(
        default=Path("models"),
        description="Directory for saved ML models"
    )
    
    # -------------------------------------------------------------------------
    # Ollama / Local LLM Settings
    # -------------------------------------------------------------------------
    # Ollama runs on localhost by default
    # Change this if running Ollama on a different machine
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API"
    )
    
    # Which model to use - "mistral" is a good balance of speed and quality
    # Other options: "llama2", "codellama", "mixtral" (needs more RAM)
    ollama_model: str = Field(
        default="mistral",
        description="Ollama model name to use for LLM tasks"
    )
    
    # Temperature controls randomness in LLM output
    # 0.0 = deterministic (same input = same output)
    # 1.0 = more creative/random
    # 0.1 = low randomness, good for extraction tasks
    llm_temperature: float = Field(
        default=0.1,
        description="LLM temperature (0.0-1.0, lower = more deterministic)"
    )
    
    # Maximum tokens (words/pieces) the LLM can generate
    llm_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for LLM response"
    )
    
    # -------------------------------------------------------------------------
    # Embedding Model Settings
    # -------------------------------------------------------------------------
    # sentence-transformers model for creating embeddings
    # "all-MiniLM-L6-v2" is small (80MB) and fast
    # For better quality, use "all-mpnet-base-v2" (420MB)
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    # Vector dimension - must match the embedding model's output
    # all-MiniLM-L6-v2 outputs 384-dimensional vectors
    embedding_dimension: int = Field(
        default=384,
        description="Dimension of embedding vectors"
    )
    
    # -------------------------------------------------------------------------
    # ChromaDB / RAG Settings
    # -------------------------------------------------------------------------
    # Name of the collection storing rate embeddings
    rates_collection_name: str = Field(
        default="market_rates",
        description="ChromaDB collection name for rate data"
    )
    
    # How many similar results to retrieve in RAG queries
    # More results = more context for LLM, but also more noise
    rag_top_k: int = Field(
        default=5,
        description="Number of similar documents to retrieve"
    )
    
    # Minimum similarity score (0-1) to consider a match relevant
    # Higher = stricter matching, fewer but better results
    rag_similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for RAG retrieval"
    )
    
    # -------------------------------------------------------------------------
    # Scoring Thresholds
    # -------------------------------------------------------------------------
    # These thresholds determine claim verdicts
    # A claim scoring above 0.7 is considered FAIR
    fair_threshold: float = Field(
        default=0.7,
        description="Score above this = FAIR verdict"
    )
    
    # A claim scoring below 0.4 is FLAGGED for review
    flagged_threshold: float = Field(
        default=0.4,
        description="Score below this = FLAGGED verdict"
    )
    
    # Between 0.4 and 0.7 = POTENTIALLY_INFLATED
    
    # -------------------------------------------------------------------------
    # API Settings
    # -------------------------------------------------------------------------
    api_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server"
    )
    
    api_port: int = Field(
        default=8000,
        description="Port for the API server"
    )
    
    # Enable debug mode (more verbose logging)
    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    
    # -------------------------------------------------------------------------
    # Pydantic Settings Configuration
    # -------------------------------------------------------------------------
    class Config:
        """
        Pydantic configuration for settings.
        
        env_file: Load variables from .env file
        env_file_encoding: Use UTF-8 encoding
        case_sensitive: OLLAMA_MODEL and ollama_model are different
        extra: ignore unknown env vars (like old AWS settings)
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra env vars not defined in Settings


# -----------------------------------------------------------------------------
# Create a global settings instance
# -----------------------------------------------------------------------------
# This is the SINGLETON pattern - one settings object used everywhere
# Import this in other files: from config import settings
settings = Settings()


# -----------------------------------------------------------------------------
# Ensure directories exist
# -----------------------------------------------------------------------------
def ensure_directories():
    """
    Create necessary directories if they don't exist.
    
    WHY: Avoids "directory not found" errors when saving files
    Called once at startup in main.py
    """
    directories = [
        settings.base_dir / settings.upload_dir,
        settings.base_dir / settings.vectorstore_dir,
        settings.base_dir / settings.models_dir,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Helper function to get absolute paths
# -----------------------------------------------------------------------------
def get_absolute_path(relative_path: Path) -> Path:
    """
    Convert a relative path to absolute path based on project root.
    
    EXAMPLE:
    get_absolute_path(Path("data/rates.csv"))
    Returns: /home/user/claimnow/data/rates.csv
    """
    if relative_path.is_absolute():
        return relative_path
    return settings.base_dir / relative_path