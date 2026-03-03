"""
config.py - All the Rules for ClaimsNOW

WHY THIS FILE EXISTS:
- Put all the rules in one spot so we find them easy
- Change the rules without changing the brain code
- Keep secret stuff (like passwords) as special notes
- The program checks all the rules are right by itself

WHAT THE HELPER DOES:
- Reads special notes from outside the program
- Checks that numbers are numbers and words are words
- Uses what we tell it if we don't leave notes
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    All the rules that tell the program what to do.
    
    HOW IT WORKS:
    1. Look for special notes with the same name as our rules
    2. If we find a note, use that number or word
    3. If we don't find a note, use what we wrote here
    4. Change it to the right type by itself ("8000" becomes 8000)
    
    EXAMPLE:
    If someone writes OLLAMA_MODEL=llama2 in a note,
    the program uses llama2 instead of mistral
    """
    
    # -------------------------------------------------------------------------
    # Where we keep things - Folders!
    # -------------------------------------------------------------------------
    # base_dir: The main folder with everything in it
    # All other folders are inside this one
    base_dir: Path = Field(
        default=Path(__file__).parent.parent,  # Go up up to the main folder
        description="The main folder with everything"
    )
    
    # Where to save files that people give us
    upload_dir: Path = Field(
        default=Path("data/uploads"),
        description="Folder for files people upload"
    )
    
    # Where the smart memory box saves its brain-stuff
    vectorstore_dir: Path = Field(
        default=Path("vectorstore"),
        description="Folder where the brain box saves things"
    )
    
    # Where we save the smart teachers we made
    models_dir: Path = Field(
        default=Path("models"),
        description="Folder for the smart teachers"
    )
    
    # -------------------------------------------------------------------------
    # The Smart Helper Settings (AWS Bedrock)
    # -------------------------------------------------------------------------
    # The AWS Region where Bedrock is located
    aws_region: str = Field(
        default="us-east-1",
        description="Region where AWS Bedrock is located"
    )
    
    # Which smart helper to use on Bedrock - Claude 3 Haiku is nice and fast
    bedrock_llm_model_id: str = Field(
        default="anthropic.claude-3-haiku-20240307-v1:0",
        description="Which smart helper to talk to on Bedrock"
    )
    
    # How wild or boring the smart helper is
    # 0.0 = always the same answer
    # 1.0 = very crazy and different
    # 0.1 = mostly same answer, safe
    llm_temperature: float = Field(
        default=0.1,
        description="Wild level (0.0-1.0, small = boring and safe)"
    )
    
    # Longest answer the smart helper can give
    llm_max_tokens: int = Field(
        default=2048,
        description="Max words the smart helper can say"
    )
    
    # -------------------------------------------------------------------------
    # How to Turn Words Into Numbers (Amazon Titan)
    # -------------------------------------------------------------------------
    # A teacher that turns words into number patterns
    bedrock_embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="The teacher on Bedrock that turns words into numbers"
    )
    
    # How many numbers we use for each word
    # Titan v2 default depends on the dimension argument sent, assuming 512 for balance
    embedding_dimension: int = Field(
        default=512,
        description="How many numbers for each word"
    )
    
    # -------------------------------------------------------------------------
    # Brain Box Stuff
    # -------------------------------------------------------------------------
    # The name of the box that saves money number-words
    rates_collection_name: str = Field(
        default="market_rates",
        description="The name of the money box"
    )
    
    # How many close matches to bring back
    # More matches = more info, but also more clutter
    rag_top_k: int = Field(
        default=5,
        description="How many close matches to find"
    )
    
    # How close does it need to match (0-1 scale)
    # Higher = picky matching, fewer but better matches
    rag_similarity_threshold: float = Field(
        default=0.7,
        description="How same does it need to be"
    )
    
    # -------------------------------------------------------------------------
    # Passing Scores - Is it Good or Bad?
    # -------------------------------------------------------------------------
    # These numbers decide if something is okay
    # Score above 0.7 = YES it's good!
    fair_threshold: float = Field(
        default=0.7,
        description="Score above this = GOOD!"
    )
    
    # Score below 0.4 = NEEDS CHECKING!
    flagged_threshold: float = Field(
        default=0.4,
        description="Score below this = NEEDS CHECKING!"
    )
    
    # Between 0.4 and 0.7 = MAYBE OKAY
    
    # -------------------------------------------------------------------------
    # How to Talk to the Program Settings
    # -------------------------------------------------------------------------
    api_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server"
    )
    
    api_port: int = Field(
        default=8000,
        description="Port for the API server"
    )
    
    # Show all the secrets and steps (helpful for fixing things)
    debug: bool = Field(
        default=True,
        description="Show all the steps and secrets"
    )
    
    # -------------------------------------------------------------------------
    # The Smart Rule Checker Setup
    # -------------------------------------------------------------------------
    class Config:
        """
        Smart rule checker settings.
        
        env_file: Read notes from the .env file
        env_file_encoding: Use the right alphabet
        case_sensitive: BIG_WORD and big_word are different
        extra: Say no thanks to notes we don't know
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Don't care about rules we don't know


# -----------------------------------------------------------------------------
# Make ONE Copy of All the Rules
# -----------------------------------------------------------------------------
# One copy of the rules for the whole program, everywhere
# Bring it in: from config import settings
settings = Settings()


# -----------------------------------------------------------------------------
# Make Sure All Folders Are There
# -----------------------------------------------------------------------------
def ensure_directories():
    """
    Make all the folders if they're not there yet.
    
    WHY: So we don't mess up when we save files
    We do this one time when the program starts
    """
    directories = [
        settings.base_dir / settings.upload_dir,
        settings.base_dir / settings.vectorstore_dir,
        settings.base_dir / settings.models_dir,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Turn a Shortcut Into a Full Address
# -----------------------------------------------------------------------------
def get_absolute_path(relative_path: Path) -> Path:
    """
    Turn a shortcut address into the full real address.
    
    EXAMPLE:
    get_absolute_path(Path("data/rates.csv"))
    Gives back: /home/user/claimnow/data/rates.csv
    """
    if relative_path.is_absolute():
        return relative_path
    return settings.base_dir / relative_path