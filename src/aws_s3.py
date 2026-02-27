# =============================================================================
# ClaimsNOW - AWS S3 Integration Module
# =============================================================================
# This module handles all interactions with Amazon S3 for file storage.
#
# S3 is used for:
# 1. Storing uploaded PDF court pack documents
# 2. Storing analysis result JSON files for auditing
# 3. Generating secure download links for the frontend
#
# The module provides a clean interface that abstracts away S3 complexity.
# =============================================================================

import uuid
import logging
from datetime import datetime
from typing import Optional, Tuple
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from src.config import settings

# Set up logging for this module
# Logs help track uploads, downloads, and any errors
logger = logging.getLogger(__name__)


# =============================================================================
# S3 Client Initialization
# =============================================================================

def get_s3_client():
    """
    Create and return an S3 client configured with our settings.
    
    The client is created fresh each time to ensure credentials are current.
    In production with IAM roles, credentials auto-refresh.
    
    Returns:
        boto3.client: Configured S3 client ready for operations.
    
    Raises:
        NoCredentialsError: If AWS credentials are not configured.
    """
    # Build client configuration from settings
    client_config = {
        "service_name": "s3",
        "region_name": settings.AWS_REGION,
    }
    
    # Only add explicit credentials if they're configured
    # Otherwise, boto3 uses IAM role credentials automatically
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        client_config["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        client_config["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
    
    return boto3.client(**client_config)


# =============================================================================
# File Upload Functions
# =============================================================================

def generate_s3_key(filename: str, prefix: str = None) -> str:
    """
    Generate a unique S3 key (path) for a file.
    
    Keys are structured as: prefix/YYYY/MM/DD/uuid_originalname
    This structure:
    - Prevents filename collisions
    - Organizes files by date for easy browsing
    - Keeps original filename for reference
    
    Args:
        filename: Original filename (e.g., "invoice.pdf")
        prefix: S3 key prefix (e.g., "uploads/"). Uses settings default if None.
    
    Returns:
        str: Full S3 key like "uploads/2026/02/27/abc123_invoice.pdf"
    
    Example:
        >>> generate_s3_key("claim_001.pdf")
        "uploads/2026/02/27/f47ac10b_claim_001.pdf"
    """
    # Use configured prefix if none provided
    if prefix is None:
        prefix = settings.S3_UPLOAD_PREFIX
    
    # Get current date for folder structure
    now = datetime.utcnow()
    date_path = now.strftime("%Y/%m/%d")
    
    # Generate UUID prefix to ensure uniqueness
    # Using first 8 chars of UUID for readability while maintaining uniqueness
    unique_id = str(uuid.uuid4())[:8]
    
    # Combine all parts into the final key
    # Example: uploads/2026/02/27/f47ac10b_invoice.pdf
    s3_key = f"{prefix}{date_path}/{unique_id}_{filename}"
    
    return s3_key


def upload_file(
    file_bytes: bytes,
    filename: str,
    content_type: str = "application/pdf"
) -> Tuple[bool, str, Optional[str]]:
    """
    Upload a file to S3.
    
    This is the main function for uploading PDF court packs.
    It handles the upload and returns the S3 key for later retrieval.
    
    Args:
        file_bytes: The file content as bytes.
        filename: Original filename for reference.
        content_type: MIME type of the file. Defaults to PDF.
    
    Returns:
        Tuple containing:
        - success (bool): True if upload succeeded
        - s3_key (str): The S3 key where file was stored (empty if failed)
        - error (str|None): Error message if failed, None if success
    
    Example:
        >>> with open("claim.pdf", "rb") as f:
        ...     success, s3_key, error = upload_file(f.read(), "claim.pdf")
        >>> if success:
        ...     print(f"Uploaded to: {s3_key}")
    """
    # Generate unique S3 key for this file
    s3_key = generate_s3_key(filename)
    
    try:
        # Get S3 client
        s3_client = get_s3_client()
        
        # Upload the file with metadata
        # ContentType helps browsers/apps handle the file correctly
        s3_client.put_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_bytes,
            ContentType=content_type,
            # Add metadata for traceability
            Metadata={
                "original_filename": filename,
                "uploaded_at": datetime.utcnow().isoformat(),
                "source": "claimsnow-api"
            }
        )
        
        logger.info(f"Successfully uploaded file to s3://{settings.S3_BUCKET_NAME}/{s3_key}")
        return True, s3_key, None
        
    except NoCredentialsError:
        # AWS credentials not configured
        error_msg = "AWS credentials not configured. Check your .env file."
        logger.error(error_msg)
        return False, "", error_msg
        
    except ClientError as e:
        # AWS returned an error (bucket doesn't exist, permission denied, etc.)
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = f"S3 upload failed: {error_code} - {str(e)}"
        logger.error(error_msg)
        return False, "", error_msg
        
    except Exception as e:
        # Catch-all for unexpected errors
        error_msg = f"Unexpected error during S3 upload: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, "", error_msg


def upload_result(
    result_json: str,
    claim_id: str
) -> Tuple[bool, str, Optional[str]]:
    """
    Upload analysis result JSON to S3 for auditing.
    
    Results are stored separately from uploads to:
    - Keep audit trail of all analyses
    - Allow easy retrieval of historical results
    - Support compliance and governance requirements
    
    Args:
        result_json: JSON string of the analysis result.
        claim_id: Unique identifier for this claim analysis.
    
    Returns:
        Tuple containing:
        - success (bool): True if upload succeeded
        - s3_key (str): The S3 key where result was stored
        - error (str|None): Error message if failed
    """
    # Generate key in results prefix folder
    filename = f"{claim_id}.json"
    s3_key = generate_s3_key(filename, prefix=settings.S3_RESULTS_PREFIX)
    
    try:
        s3_client = get_s3_client()
        
        # Upload JSON result
        s3_client.put_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            Body=result_json.encode("utf-8"),
            ContentType="application/json",
            Metadata={
                "claim_id": claim_id,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Successfully uploaded result to s3://{settings.S3_BUCKET_NAME}/{s3_key}")
        return True, s3_key, None
        
    except Exception as e:
        error_msg = f"Failed to upload result: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, "", error_msg


# =============================================================================
# File Download Functions
# =============================================================================

def download_file(s3_key: str) -> Tuple[bool, Optional[bytes], Optional[str]]:
    """
    Download a file from S3.
    
    Used to retrieve uploaded PDFs for processing or
    to fetch stored analysis results.
    
    Args:
        s3_key: The S3 key of the file to download.
    
    Returns:
        Tuple containing:
        - success (bool): True if download succeeded
        - content (bytes|None): File content if success, None if failed
        - error (str|None): Error message if failed
    
    Example:
        >>> success, content, error = download_file("uploads/2026/02/27/abc_claim.pdf")
        >>> if success:
        ...     with open("local_copy.pdf", "wb") as f:
        ...         f.write(content)
    """
    try:
        s3_client = get_s3_client()
        
        # Get the object from S3
        response = s3_client.get_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key
        )
        
        # Read the content from the response body
        content = response["Body"].read()
        
        logger.info(f"Successfully downloaded s3://{settings.S3_BUCKET_NAME}/{s3_key}")
        return True, content, None
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        
        # Handle specific error cases
        if error_code == "NoSuchKey":
            error_msg = f"File not found: {s3_key}"
        elif error_code == "AccessDenied":
            error_msg = f"Access denied to file: {s3_key}"
        else:
            error_msg = f"S3 download failed: {error_code}"
            
        logger.error(error_msg)
        return False, None, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error during S3 download: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


# =============================================================================
# Presigned URL Functions
# =============================================================================

def generate_presigned_url(
    s3_key: str,
    expiration_seconds: int = 3600
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Generate a presigned URL for secure file access.
    
    Presigned URLs allow temporary access to private S3 files
    without exposing AWS credentials. Perfect for:
    - Letting users download their uploaded PDFs
    - Sharing analysis results
    - Frontend file preview
    
    Args:
        s3_key: The S3 key of the file.
        expiration_seconds: How long the URL is valid. Default 1 hour.
    
    Returns:
        Tuple containing:
        - success (bool): True if URL was generated
        - url (str|None): The presigned URL if success
        - error (str|None): Error message if failed
    
    Security Note:
        Anyone with the URL can access the file until it expires.
        Use short expiration times for sensitive documents.
    """
    try:
        s3_client = get_s3_client()
        
        # Generate the presigned URL
        url = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": settings.S3_BUCKET_NAME,
                "Key": s3_key
            },
            ExpiresIn=expiration_seconds
        )
        
        logger.info(f"Generated presigned URL for {s3_key}, expires in {expiration_seconds}s")
        return True, url, None
        
    except Exception as e:
        error_msg = f"Failed to generate presigned URL: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


def generate_upload_presigned_url(
    filename: str,
    content_type: str = "application/pdf",
    expiration_seconds: int = 3600
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Generate a presigned URL for direct frontend uploads.
    
    This allows the frontend to upload files directly to S3
    without going through our API server. Benefits:
    - Faster uploads (direct to S3)
    - Reduced server load
    - Better for large files
    
    Args:
        filename: Intended filename for the upload.
        content_type: MIME type of the file.
        expiration_seconds: How long the URL is valid.
    
    Returns:
        Tuple containing:
        - success (bool): True if URL was generated
        - url (str|None): The presigned upload URL
        - s3_key (str|None): The S3 key where file will be stored
        - error (str|None): Error message if failed
    
    Usage (Frontend):
        fetch(presigned_url, {
            method: 'PUT',
            body: file,
            headers: { 'Content-Type': 'application/pdf' }
        })
    """
    # Generate the S3 key that will be used
    s3_key = generate_s3_key(filename)
    
    try:
        s3_client = get_s3_client()
        
        # Generate presigned URL for PUT operation
        url = s3_client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": settings.S3_BUCKET_NAME,
                "Key": s3_key,
                "ContentType": content_type
            },
            ExpiresIn=expiration_seconds
        )
        
        logger.info(f"Generated upload presigned URL for {s3_key}")
        return True, url, s3_key, None
        
    except Exception as e:
        error_msg = f"Failed to generate upload presigned URL: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, None, error_msg


# =============================================================================
# Utility Functions
# =============================================================================

def file_exists(s3_key: str) -> bool:
    """
    Check if a file exists in S3.
    
    Useful for validation before processing or
    avoiding duplicate uploads.
    
    Args:
        s3_key: The S3 key to check.
    
    Returns:
        bool: True if file exists, False otherwise.
    """
    try:
        s3_client = get_s3_client()
        
        # head_object returns metadata without downloading
        # It raises an exception if the file doesn't exist
        s3_client.head_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key
        )
        return True
        
    except ClientError as e:
        # NoSuchKey means file doesn't exist
        if e.response.get("Error", {}).get("Code") == "404":
            return False
        # For other errors, log and return False
        logger.warning(f"Error checking file existence: {str(e)}")
        return False


def delete_file(s3_key: str) -> Tuple[bool, Optional[str]]:
    """
    Delete a file from S3.
    
    Use with caution - deletions are permanent unless
    bucket versioning is enabled.
    
    Args:
        s3_key: The S3 key of the file to delete.
    
    Returns:
        Tuple containing:
        - success (bool): True if deletion succeeded
        - error (str|None): Error message if failed
    """
    try:
        s3_client = get_s3_client()
        
        s3_client.delete_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key
        )
        
        logger.info(f"Deleted file: s3://{settings.S3_BUCKET_NAME}/{s3_key}")
        return True, None
        
    except Exception as e:
        error_msg = f"Failed to delete file: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg
