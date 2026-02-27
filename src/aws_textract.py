# =============================================================================
# ClaimsNOW - AWS Textract Integration Module
# =============================================================================
# This module handles document text extraction using Amazon Textract.
#
# Textract is AWS's OCR service that can:
# 1. Extract raw text from PDFs and images
# 2. Detect and extract tables (useful for invoice line items)
# 3. Identify form fields (key-value pairs)
#
# For court pack documents, we primarily use:
# - Text extraction for hire details in paragraphs
# - Table extraction for itemized charges
# =============================================================================

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import boto3
from botocore.exceptions import ClientError

from src.config import settings

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# Textract Client Initialization
# =============================================================================

def get_textract_client():
    """
    Create and return a Textract client configured with our settings.
    
    Returns:
        boto3.client: Configured Textract client.
    """
    client_config = {
        "service_name": "textract",
        "region_name": settings.AWS_REGION,
    }
    
    # Add explicit credentials if configured
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        client_config["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        client_config["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
    
    return boto3.client(**client_config)


# =============================================================================
# Document Analysis Functions
# =============================================================================

def start_document_analysis(
    s3_bucket: str,
    s3_key: str,
    feature_types: List[str] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Start an asynchronous Textract document analysis job.
    
    For multi-page PDFs, we use async analysis because:
    - Synchronous API has page limits
    - Async allows processing larger documents
    - We can poll for completion while doing other work
    
    Args:
        s3_bucket: S3 bucket containing the document.
        s3_key: S3 key (path) of the document.
        feature_types: Analysis features to enable. Options:
            - "TABLES": Extract tables
            - "FORMS": Extract form key-value pairs
            Default: ["TABLES"] for invoice processing
    
    Returns:
        Tuple containing:
        - success (bool): True if job started successfully
        - job_id (str|None): Textract job ID for polling
        - error (str|None): Error message if failed
    
    Example:
        >>> success, job_id, error = start_document_analysis(
        ...     "my-bucket", "uploads/claim.pdf"
        ... )
        >>> if success:
        ...     # Poll for completion using job_id
        ...     result = wait_for_analysis(job_id)
    """
    # Default to table extraction for invoices
    if feature_types is None:
        feature_types = ["TABLES"]
    
    try:
        textract = get_textract_client()
        
        # Start the async analysis job
        response = textract.start_document_analysis(
            DocumentLocation={
                "S3Object": {
                    "Bucket": s3_bucket,
                    "Name": s3_key
                }
            },
            FeatureTypes=feature_types
        )
        
        job_id = response["JobId"]
        logger.info(f"Started Textract job {job_id} for s3://{s3_bucket}/{s3_key}")
        
        return True, job_id, None
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_msg = f"Textract start failed: {error_code} - {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error starting Textract: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


def get_analysis_results(job_id: str) -> Tuple[str, Optional[Dict], Optional[str]]:
    """
    Get the current status and results of a Textract job.
    
    Textract jobs can be in these states:
    - IN_PROGRESS: Still processing
    - SUCCEEDED: Complete, results available
    - FAILED: Processing failed
    - PARTIAL_SUCCESS: Some pages processed
    
    Args:
        job_id: The Textract job ID returned from start_document_analysis.
    
    Returns:
        Tuple containing:
        - status (str): Job status (IN_PROGRESS, SUCCEEDED, FAILED)
        - results (dict|None): Textract response if complete
        - error (str|None): Error message if failed
    """
    try:
        textract = get_textract_client()
        
        # Get the analysis results
        # For large documents, results are paginated
        response = textract.get_document_analysis(JobId=job_id)
        
        status = response["JobStatus"]
        
        if status == "SUCCEEDED":
            # Collect all pages of results
            all_blocks = response.get("Blocks", [])
            
            # Handle pagination for large documents
            next_token = response.get("NextToken")
            while next_token:
                # Get next page of results
                page_response = textract.get_document_analysis(
                    JobId=job_id,
                    NextToken=next_token
                )
                all_blocks.extend(page_response.get("Blocks", []))
                next_token = page_response.get("NextToken")
            
            # Build complete results object
            results = {
                "JobStatus": status,
                "Blocks": all_blocks,
                "DocumentMetadata": response.get("DocumentMetadata", {})
            }
            
            logger.info(f"Textract job {job_id} completed with {len(all_blocks)} blocks")
            return status, results, None
            
        elif status == "FAILED":
            error_msg = response.get("StatusMessage", "Analysis failed")
            logger.error(f"Textract job {job_id} failed: {error_msg}")
            return status, None, error_msg
            
        else:
            # Still in progress
            return status, None, None
            
    except ClientError as e:
        error_msg = f"Error getting Textract results: {str(e)}"
        logger.error(error_msg)
        return "ERROR", None, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return "ERROR", None, error_msg


def wait_for_analysis(
    job_id: str,
    max_wait_seconds: int = 300,
    poll_interval_seconds: int = 5
) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Wait for a Textract analysis job to complete.
    
    This function polls the job status until completion or timeout.
    Used when you need results before proceeding.
    
    Args:
        job_id: The Textract job ID to wait for.
        max_wait_seconds: Maximum time to wait. Default 5 minutes.
        poll_interval_seconds: Time between status checks. Default 5 seconds.
    
    Returns:
        Tuple containing:
        - success (bool): True if job completed successfully
        - results (dict|None): Textract results if successful
        - error (str|None): Error message if failed or timed out
    
    Example:
        >>> success, job_id, _ = start_document_analysis(bucket, key)
        >>> success, results, error = wait_for_analysis(job_id)
        >>> if success:
        ...     text = extract_text_from_results(results)
    """
    start_time = time.time()
    
    logger.info(f"Waiting for Textract job {job_id}...")
    
    while True:
        # Check if we've exceeded the timeout
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            error_msg = f"Textract job timed out after {max_wait_seconds} seconds"
            logger.error(error_msg)
            return False, None, error_msg
        
        # Get current job status
        status, results, error = get_analysis_results(job_id)
        
        if status == "SUCCEEDED":
            logger.info(f"Textract job {job_id} completed in {elapsed:.1f} seconds")
            return True, results, None
            
        elif status in ["FAILED", "ERROR"]:
            return False, None, error
            
        # Still in progress, wait before next poll
        logger.debug(f"Job {job_id} still in progress, waiting {poll_interval_seconds}s...")
        time.sleep(poll_interval_seconds)


# =============================================================================
# Synchronous Analysis (For Single-Page Documents)
# =============================================================================

def analyze_document_sync(
    document_bytes: bytes,
    feature_types: List[str] = None
) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Analyze a document synchronously (in real-time).
    
    Use this for:
    - Single-page documents
    - Quick testing
    - When you need immediate results
    
    Limitations:
    - Max 1 page for images
    - Max 3000 pages for PDFs (but slower than async)
    
    Args:
        document_bytes: The document content as bytes.
        feature_types: Analysis features (TABLES, FORMS). Default: TABLES.
    
    Returns:
        Tuple containing:
        - success (bool): True if analysis succeeded
        - results (dict|None): Textract response if successful
        - error (str|None): Error message if failed
    """
    if feature_types is None:
        feature_types = ["TABLES"]
    
    try:
        textract = get_textract_client()
        
        # Run synchronous analysis
        response = textract.analyze_document(
            Document={"Bytes": document_bytes},
            FeatureTypes=feature_types
        )
        
        logger.info(f"Sync analysis completed with {len(response.get('Blocks', []))} blocks")
        return True, response, None
        
    except ClientError as e:
        error_msg = f"Textract analysis failed: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


def detect_text_sync(document_bytes: bytes) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Simple text detection without table/form analysis.
    
    Faster and cheaper than full analysis. Use when you only
    need raw text extraction without structure.
    
    Args:
        document_bytes: The document content as bytes.
    
    Returns:
        Tuple containing:
        - success (bool): True if detection succeeded
        - results (dict|None): Textract response if successful
        - error (str|None): Error message if failed
    """
    try:
        textract = get_textract_client()
        
        # Simple text detection (faster, cheaper)
        response = textract.detect_document_text(
            Document={"Bytes": document_bytes}
        )
        
        logger.info(f"Text detection completed with {len(response.get('Blocks', []))} blocks")
        return True, response, None
        
    except ClientError as e:
        error_msg = f"Text detection failed: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


# =============================================================================
# Result Parsing Functions
# =============================================================================

def extract_text_from_results(textract_results: Dict) -> str:
    """
    Extract plain text from Textract results.
    
    Textract returns "blocks" representing different elements:
    - PAGE: A page in the document
    - LINE: A line of text
    - WORD: A single word
    - TABLE: A detected table
    - CELL: A table cell
    
    This function collects all LINE blocks to reconstruct the text.
    
    Args:
        textract_results: The response from Textract analysis.
    
    Returns:
        str: All extracted text, with lines separated by newlines.
    
    Example:
        >>> text = extract_text_from_results(results)
        >>> print(text)
        "HIRE INVOICE\\nCompany: ABC Hire Ltd\\nDaily Rate: £50.00\\n..."
    """
    blocks = textract_results.get("Blocks", [])
    
    # Filter for LINE blocks and extract their text
    lines = []
    for block in blocks:
        if block.get("BlockType") == "LINE":
            text = block.get("Text", "")
            if text:
                lines.append(text)
    
    # Join all lines with newlines
    full_text = "\n".join(lines)
    
    logger.debug(f"Extracted {len(lines)} lines of text")
    return full_text


def extract_tables_from_results(textract_results: Dict) -> List[List[List[str]]]:
    """
    Extract tables from Textract results.
    
    Tables are returned as a list of tables, where each table is a
    2D list (rows x columns) of cell text values.
    
    This is useful for processing invoice line items, which often
    appear in tabular format.
    
    Args:
        textract_results: The response from Textract analysis.
    
    Returns:
        List of tables, each table is a list of rows, each row is a list of cells.
    
    Example:
        >>> tables = extract_tables_from_results(results)
        >>> for table in tables:
        ...     for row in table:
        ...         print(row)  # ["Item", "Qty", "Price"]
    """
    blocks = textract_results.get("Blocks", [])
    
    # Build a lookup map of block ID to block data
    # This helps us navigate the parent-child relationships
    block_map = {block["Id"]: block for block in blocks}
    
    tables = []
    
    # Find all TABLE blocks
    for block in blocks:
        if block.get("BlockType") != "TABLE":
            continue
        
        table_data = []
        
        # Get relationships (child blocks)
        relationships = block.get("Relationships", [])
        
        # Find child cells
        cell_ids = []
        for rel in relationships:
            if rel.get("Type") == "CHILD":
                cell_ids.extend(rel.get("Ids", []))
        
        # Build a grid from cells
        # Cells have RowIndex and ColumnIndex properties
        max_row = 0
        max_col = 0
        cell_grid = {}
        
        for cell_id in cell_ids:
            cell_block = block_map.get(cell_id, {})
            if cell_block.get("BlockType") != "CELL":
                continue
            
            row_idx = cell_block.get("RowIndex", 1) - 1  # Convert to 0-indexed
            col_idx = cell_block.get("ColumnIndex", 1) - 1
            
            max_row = max(max_row, row_idx)
            max_col = max(max_col, col_idx)
            
            # Get cell text from child WORD blocks
            cell_text = _get_cell_text(cell_block, block_map)
            cell_grid[(row_idx, col_idx)] = cell_text
        
        # Convert grid to 2D list
        for row in range(max_row + 1):
            row_data = []
            for col in range(max_col + 1):
                cell_text = cell_grid.get((row, col), "")
                row_data.append(cell_text)
            table_data.append(row_data)
        
        if table_data:
            tables.append(table_data)
    
    logger.debug(f"Extracted {len(tables)} tables")
    return tables


def _get_cell_text(cell_block: Dict, block_map: Dict) -> str:
    """
    Extract text content from a table cell block.
    
    Cell blocks contain WORD child blocks that hold the actual text.
    This helper function navigates that relationship.
    
    Args:
        cell_block: The CELL block from Textract.
        block_map: Lookup map of all blocks by ID.
    
    Returns:
        str: The text content of the cell.
    """
    words = []
    
    # Find child relationships
    relationships = cell_block.get("Relationships", [])
    
    for rel in relationships:
        if rel.get("Type") == "CHILD":
            for word_id in rel.get("Ids", []):
                word_block = block_map.get(word_id, {})
                if word_block.get("BlockType") == "WORD":
                    text = word_block.get("Text", "")
                    if text:
                        words.append(text)
    
    return " ".join(words)


def extract_key_value_pairs(textract_results: Dict) -> Dict[str, str]:
    """
    Extract form key-value pairs from Textract results.
    
    Textract can identify form fields like:
    - "Invoice Number: INV-001"
    - "Date: 15/02/2026"
    - "Total: £1,234.00"
    
    This is useful for structured invoice fields.
    
    Args:
        textract_results: The response from Textract analysis (must include FORMS feature).
    
    Returns:
        dict: Mapping of field names to their values.
    
    Example:
        >>> pairs = extract_key_value_pairs(results)
        >>> print(pairs)
        {"Invoice Number": "INV-001", "Date": "15/02/2026", "Total": "£1,234.00"}
    """
    blocks = textract_results.get("Blocks", [])
    
    # Build block lookup map
    block_map = {block["Id"]: block for block in blocks}
    
    key_value_pairs = {}
    
    # Find KEY_VALUE_SET blocks with EntityType = KEY
    for block in blocks:
        if block.get("BlockType") != "KEY_VALUE_SET":
            continue
        
        # Skip VALUE blocks (we process from KEY side)
        if block.get("EntityTypes", []) != ["KEY"]:
            continue
        
        # Get the key text
        key_text = _get_block_text(block, block_map)
        
        # Find the linked VALUE block
        value_text = ""
        relationships = block.get("Relationships", [])
        for rel in relationships:
            if rel.get("Type") == "VALUE":
                for value_id in rel.get("Ids", []):
                    value_block = block_map.get(value_id, {})
                    value_text = _get_block_text(value_block, block_map)
                    break
        
        if key_text:
            key_value_pairs[key_text] = value_text
    
    logger.debug(f"Extracted {len(key_value_pairs)} key-value pairs")
    return key_value_pairs


def _get_block_text(block: Dict, block_map: Dict) -> str:
    """
    Get text content from a block by following child relationships.
    
    Args:
        block: The block to extract text from.
        block_map: Lookup map of all blocks by ID.
    
    Returns:
        str: The text content of the block.
    """
    words = []
    
    relationships = block.get("Relationships", [])
    for rel in relationships:
        if rel.get("Type") == "CHILD":
            for child_id in rel.get("Ids", []):
                child_block = block_map.get(child_id, {})
                if child_block.get("BlockType") == "WORD":
                    text = child_block.get("Text", "")
                    if text:
                        words.append(text)
    
    return " ".join(words)


# =============================================================================
# High-Level Convenience Functions
# =============================================================================

def extract_document_content(
    s3_bucket: str,
    s3_key: str,
    include_tables: bool = True
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    High-level function to extract all content from a document.
    
    This is the main function to call from the pipeline.
    It handles the full process:
    1. Starts async analysis
    2. Waits for completion
    3. Extracts text, tables, and key-value pairs
    4. Returns everything in a structured format
    
    Args:
        s3_bucket: S3 bucket containing the document.
        s3_key: S3 key of the document.
        include_tables: Whether to extract tables. Default True.
    
    Returns:
        Tuple containing:
        - success (bool): True if extraction succeeded
        - content (dict|None): Extracted content with keys:
            - "text": Full extracted text
            - "tables": List of tables (if include_tables=True)
            - "key_value_pairs": Detected form fields
            - "page_count": Number of pages processed
        - error (str|None): Error message if failed
    
    Example:
        >>> success, content, error = extract_document_content(bucket, key)
        >>> if success:
        ...     print(content["text"])
        ...     for table in content["tables"]:
        ...         process_table(table)
    """
    # Determine which features to enable
    feature_types = ["TABLES", "FORMS"] if include_tables else ["FORMS"]
    
    # Start the analysis job
    success, job_id, error = start_document_analysis(
        s3_bucket, s3_key, feature_types
    )
    
    if not success:
        return False, None, error
    
    # Wait for completion
    success, results, error = wait_for_analysis(job_id)
    
    if not success:
        return False, None, error
    
    # Extract all content types
    text = extract_text_from_results(results)
    tables = extract_tables_from_results(results) if include_tables else []
    key_values = extract_key_value_pairs(results)
    
    # Get page count from metadata
    page_count = results.get("DocumentMetadata", {}).get("Pages", 0)
    
    # Package everything into a clean structure
    content = {
        "text": text,
        "tables": tables,
        "key_value_pairs": key_values,
        "page_count": page_count,
        "job_id": job_id  # Include for audit trail
    }
    
    logger.info(f"Extracted content: {page_count} pages, {len(tables)} tables, {len(key_values)} form fields")
    
    return True, content, None
