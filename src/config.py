# =============================================================================
# ClaimsNOW - Configuration Module
# =============================================================================
# This module centralizes all application configuration settings.
# It loads values from environment variables (.env file) and provides
# sensible defaults where appropriate.
#
# Usage:
#   from src.config import settings
#   bucket_name = settings.S3_BUCKET_NAME
# =============================================================================

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Pydantic's BaseSettings automatically reads from:
    1. Environment variables (highest priority)
    2. .env file (if present)
    3. Default values defined here (lowest priority)
    
    This pattern keeps sensitive credentials out of code.
    """
    
    # -------------------------------------------------------------------------
    # AWS Credentials
    # -------------------------------------------------------------------------
    # These authenticate your application with AWS services.
    # In production, prefer IAM roles over access keys.
    
    AWS_ACCESS_KEY_ID: Optional[str] = Field(
        default=None,
        description="AWS access key for authentication. Leave None to use IAM role."
    )
    
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(
        default=None,
        description="AWS secret key for authentication. Keep this secure!"
    )
    
    AWS_REGION: str = Field(
        default="eu-west-2",  # London region - good for UK insurance data
        description="AWS region where your resources are deployed."
    )
    
    # -------------------------------------------------------------------------
    # S3 Configuration - File Storage
    # -------------------------------------------------------------------------
    # S3 stores uploaded PDF documents and analysis results.
    
    S3_BUCKET_NAME: str = Field(
        default="claimsnow-documents",
        description="S3 bucket name for storing uploaded PDFs."
    )
    
    S3_UPLOAD_PREFIX: str = Field(
        default="uploads/",
        description="S3 key prefix for uploaded documents."
    )
    
    S3_RESULTS_PREFIX: str = Field(
        default="results/",
        description="S3 key prefix for analysis result files."
    )
    
    # -------------------------------------------------------------------------
    # DynamoDB Configuration - Database
    # -------------------------------------------------------------------------
    # DynamoDB stores market rate reference data and claim analysis history.
    
    DYNAMODB_RATES_TABLE: str = Field(
        default="claimsnow-market-rates",
        description="DynamoDB table containing market hire rate benchmarks."
    )
    
    DYNAMODB_CLAIMS_TABLE: str = Field(
        default="claimsnow-claims",
        description="DynamoDB table storing claim analysis results for audit."
    )
    
    # -------------------------------------------------------------------------
    # Textract Configuration - Document OCR
    # -------------------------------------------------------------------------
    # Textract extracts text and tables from PDF documents.
    
    TEXTRACT_MAX_PAGES: int = Field(
        default=50,
        description="Maximum pages to process per document (cost control)."
    )
    
    # -------------------------------------------------------------------------
    # Bedrock Configuration - AI/LLM
    # -------------------------------------------------------------------------
    # Bedrock provides access to Claude for intelligent extraction.
    
    BEDROCK_MODEL_ID: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="Bedrock model ID for Claude. Sonnet balances speed and quality."
    )
    
    BEDROCK_MAX_TOKENS: int = Field(
        default=4096,
        description="Maximum tokens in Claude's response."
    )
    
    BEDROCK_TEMPERATURE: float = Field(
        default=0.0,
        description="Temperature for Claude. 0.0 = deterministic (best for extraction)."
    )
    
    # -------------------------------------------------------------------------
    # Scoring Thresholds - Claim Classification
    # -------------------------------------------------------------------------
    # These thresholds determine how claims are classified based on
    # the inflation ratio (claimed_rate / market_rate_high).
    #
    # Example: If market rate is £50/day and claim is £60/day:
    #   inflation_ratio = 60 / 50 = 1.2
    #   1.2 is between 1.1 and 1.4, so verdict = "POTENTIALLY_INFLATED"
    
    THRESHOLD_FAIR: float = Field(
        default=1.1,
        description="Below this ratio, claim is classified as FAIR."
    )
    
    THRESHOLD_INFLATED: float = Field(
        default=1.4,
        description="Between FAIR and this, claim is POTENTIALLY_INFLATED. Above = FLAGGED."
    )
    
    # -------------------------------------------------------------------------
    # Confidence Scoring Weights
    # -------------------------------------------------------------------------
    # Each extracted field contributes to the overall confidence score.
    # If a field is missing, confidence is reduced by its weight.
    # Total weights should sum to 1.0 (100%).
    
    CONFIDENCE_WEIGHT_HIRE_DATES: float = Field(
        default=0.20,
        description="Weight for hire start/end dates in confidence calculation."
    )
    
    CONFIDENCE_WEIGHT_VEHICLE_CLASS: float = Field(
        default=0.15,
        description="Weight for vehicle class/category field."
    )
    
    CONFIDENCE_WEIGHT_DAILY_RATE: float = Field(
        default=0.30,
        description="Weight for daily hire rate. Critical field - highest weight."
    )
    
    CONFIDENCE_WEIGHT_TOTAL_CLAIMED: float = Field(
        default=0.20,
        description="Weight for total amount claimed."
    )
    
    CONFIDENCE_WEIGHT_HIRE_COMPANY: float = Field(
        default=0.15,
        description="Weight for hire company name."
    )
    
    # -------------------------------------------------------------------------
    # API Configuration
    # -------------------------------------------------------------------------
    # Settings for the FastAPI REST API.
    
    API_HOST: str = Field(
        default="0.0.0.0",
        description="Host address for the API server."
    )
    
    API_PORT: int = Field(
        default=8000,
        description="Port number for the API server."
    )
    
    API_DEBUG: bool = Field(
        default=False,
        description="Enable debug mode. Set True for development only."
    )
    
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated list of allowed CORS origins for frontend."
    )
    
    # -------------------------------------------------------------------------
    # Pydantic Settings Configuration
    # -------------------------------------------------------------------------
    
    class Config:
        """
        Pydantic configuration for settings loading.
        """
        # Load variables from .env file in project root
        env_file = ".env"
        
        # Encoding for the .env file
        env_file_encoding = "utf-8"
        
        # Allow extra fields (forwards compatibility)
        extra = "ignore"


# =============================================================================
# Global Settings Instance
# =============================================================================
# Import this instance throughout the application:
#   from src.config import settings
#
# This ensures all modules use the same configuration values.

settings = Settings()


# =============================================================================
# Verdict Constants
# =============================================================================
# String constants for claim verdicts to ensure consistency across modules.

class Verdict:
    """
    Claim verdict constants.
    
    Using a class with string constants instead of an Enum because:
    1. Easier JSON serialization
    2. Simpler string comparisons
    3. More readable in logs and API responses
    """
    FAIR = "FAIR"
    POTENTIALLY_INFLATED = "POTENTIALLY_INFLATED"
    FLAGGED = "FLAGGED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # When extraction fails


# =============================================================================
# Vehicle Class Constants
# =============================================================================
# Standard vehicle classifications used in UK motor insurance.

class VehicleClass:
    """
    Standard UK vehicle hire groups.
    
    These map to industry-standard classifications:
    - Groups A-J: Standard sizing from mini to premium
    - Named categories: SUV, Van, etc. for special vehicle types
    """
    GROUP_A = "GROUP_A"  # Mini (e.g., Fiat 500)
    GROUP_B = "GROUP_B"  # Economy (e.g., Ford Fiesta)
    GROUP_C = "GROUP_C"  # Compact (e.g., VW Golf)
    GROUP_D = "GROUP_D"  # Intermediate (e.g., Ford Focus)
    GROUP_E = "GROUP_E"  # Standard (e.g., VW Passat)
    GROUP_F = "GROUP_F"  # Full Size (e.g., Ford Mondeo)
    GROUP_G = "GROUP_G"  # Premium (e.g., BMW 3 Series)
    GROUP_H = "GROUP_H"  # Luxury (e.g., Mercedes E-Class)
    GROUP_I = "GROUP_I"  # Executive (e.g., BMW 5 Series)
    GROUP_J = "GROUP_J"  # Prestige (e.g., Mercedes S-Class)
    SUV = "SUV"          # Sports Utility Vehicle
    MPV = "MPV"          # Multi-Purpose Vehicle (people carrier)
    VAN = "VAN"          # Commercial van
    UNKNOWN = "UNKNOWN"  # When vehicle class cannot be determined


# =============================================================================
# UK Regions for Rate Matching
# =============================================================================
# Hire rates vary by UK region. These align with typical insurer zones.

class Region:
    """
    UK geographic regions for rate matching.
    
    Hire rates vary significantly by region due to:
    - Local competition levels
    - Urban vs rural pricing
    - Regional cost of living differences
    """
    LONDON = "LONDON"
    SOUTH_EAST = "SOUTH_EAST"
    SOUTH_WEST = "SOUTH_WEST"
    EAST_ANGLIA = "EAST_ANGLIA"
    EAST_MIDLANDS = "EAST_MIDLANDS"
    WEST_MIDLANDS = "WEST_MIDLANDS"
    NORTH_WEST = "NORTH_WEST"
    NORTH_EAST = "NORTH_EAST"
    YORKSHIRE = "YORKSHIRE"
    WALES = "WALES"
    SCOTLAND = "SCOTLAND"
    NORTHERN_IRELAND = "NORTHERN_IRELAND"
    UNKNOWN = "UNKNOWN"  # When region cannot be determined


# =============================================================================
# Helper Functions
# =============================================================================

def get_cors_origins_list() -> list[str]:
    """
    Parse CORS origins from comma-separated string to list.
    
    Returns:
        List of allowed origin URLs for CORS.
    
    Example:
        settings.CORS_ORIGINS = "http://localhost:3000,http://localhost:5173"
        get_cors_origins_list() -> ["http://localhost:3000", "http://localhost:5173"]
    """
    return [origin.strip() for origin in settings.CORS_ORIGINS.split(",")]


def is_aws_configured() -> bool:
    """
    Check if AWS credentials are properly configured.
    
    Returns:
        True if credentials are available (either explicit or via IAM role).
    
    Note:
        This doesn't validate the credentials work, just that they exist.
        For IAM roles, credentials may not be explicitly set but boto3
        will automatically use the role.
    """
    # If explicit credentials are set, check both are present
    if settings.AWS_ACCESS_KEY_ID or settings.AWS_SECRET_ACCESS_KEY:
        return bool(settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY)
    
    # If no explicit credentials, assume IAM role will be used
    # boto3 handles this automatically in AWS environments
    return True
