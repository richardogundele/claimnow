"""
document_parser.py - Local PDF Text Extraction

WHY THIS FILE EXISTS:
- Extract text from court pack PDFs without cloud services
- Handle both digital PDFs (with embedded text) and scanned images
- Provide structured output for downstream processing

LIBRARIES USED:
- PyMuPDF (fitz): Fast PDF parsing, extracts text and metadata
- pdfplumber: Better at extracting tables
- pytesseract: OCR for scanned documents (requires Tesseract installed)
- Pillow (PIL): Image processing for OCR

HOW PDF TEXT EXTRACTION WORKS:
1. Digital PDF: Text is embedded as characters - just read it
2. Scanned PDF: Pages are images - need OCR to "read" the image
3. We try digital extraction first, fall back to OCR if needed
"""

import io
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# PyMuPDF is imported as 'fitz' (historical name)
import fitz
import pdfplumber
from PIL import Image

# Set up logging - helps debug extraction issues
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Classes - Structured Output
# -----------------------------------------------------------------------------
@dataclass
class PageContent:
    """
    Represents extracted content from a single PDF page.
    
    DATACLASS EXPLANATION:
    @dataclass automatically generates __init__, __repr__, etc.
    Instead of writing:
        def __init__(self, page_number, text, ...):
            self.page_number = page_number
            ...
    
    We just define the fields and Python does the rest.
    """
    page_number: int              # 1-indexed page number
    text: str                     # Extracted text content
    tables: List[List[str]] = field(default_factory=list)  # Any tables found
    confidence: float = 1.0       # OCR confidence (1.0 for digital text)
    extraction_method: str = "digital"  # "digital" or "ocr"


@dataclass
class DocumentContent:
    """
    Represents the full extracted content from a PDF document.
    
    Contains all pages plus document-level metadata.
    """
    filename: str                           # Original filename
    total_pages: int                        # Number of pages
    pages: List[PageContent]                # Content from each page
    full_text: str = ""                     # Combined text from all pages
    metadata: Dict[str, Any] = field(default_factory=dict)  # PDF metadata
    

class DocumentParser:
    """
    Extracts text and tables from PDF documents.
    
    USAGE:
        parser = DocumentParser()
        content = parser.parse("invoice.pdf")
        print(content.full_text)
    
    DESIGN DECISIONS:
    1. Try PyMuPDF first (fastest)
    2. Use pdfplumber for tables (better table detection)
    3. Fall back to OCR if text extraction yields little content
    """
    
    def __init__(self, ocr_enabled: bool = True, ocr_language: str = "eng"):
        """
        Initialize the document parser.
        
        Args:
            ocr_enabled: Whether to use OCR for scanned documents
            ocr_language: Tesseract language code (eng, fra, deu, etc.)
        """
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        
        # Check if Tesseract is available (only if OCR is enabled)
        self._tesseract_available = False
        if ocr_enabled:
            self._check_tesseract()
    
    def _check_tesseract(self) -> None:
        """
        Check if Tesseract OCR is installed on the system.
        
        WHY: Tesseract must be installed separately (not just pip install)
        On Windows: Download installer from GitHub
        On Mac: brew install tesseract
        On Linux: apt install tesseract-ocr
        """
        try:
            import pytesseract
            # This will raise an error if Tesseract is not installed
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            logger.info("Tesseract OCR is available")
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}. OCR disabled.")
            self._tesseract_available = False
    
    def parse(self, file_path: str | Path) -> DocumentContent:
        """
        Parse a PDF document and extract all text content.
        
        This is the MAIN METHOD you'll call from other code.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            DocumentContent with extracted text and metadata
            
        Raises:
            FileNotFoundError: If the PDF doesn't exist
            ValueError: If the file is not a valid PDF
        """
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        
        # Validate it's a PDF
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")
        
        logger.info(f"Parsing document: {file_path.name}")
        
        # Extract content using PyMuPDF
        pages = self._extract_with_pymupdf(file_path)
        
        # Extract tables using pdfplumber (better table detection)
        self._extract_tables_with_pdfplumber(file_path, pages)
        
        # Check if we got enough text - if not, try OCR
        total_text_length = sum(len(page.text) for page in pages)
        
        if total_text_length < 100 and self._tesseract_available:
            # Probably a scanned document - use OCR
            logger.info("Low text content detected, attempting OCR...")
            pages = self._extract_with_ocr(file_path)
        
        # Combine all page text into one string
        full_text = "\n\n".join(
            f"--- Page {page.page_number} ---\n{page.text}"
            for page in pages
        )
        
        # Get document metadata
        metadata = self._extract_metadata(file_path)
        
        return DocumentContent(
            filename=file_path.name,
            total_pages=len(pages),
            pages=pages,
            full_text=full_text,
            metadata=metadata
        )
    
    def _extract_with_pymupdf(self, file_path: Path) -> List[PageContent]:
        """
        Extract text using PyMuPDF (fitz).
        
        WHY PYMUPDF:
        - Very fast (written in C)
        - Good text extraction for digital PDFs
        - Can also extract images, links, annotations
        
        HOW IT WORKS:
        1. Open the PDF as a fitz.Document
        2. Iterate through each page
        3. Call get_text() to extract text
        4. Store in PageContent objects
        """
        pages = []
        
        # Open the PDF document
        # Using 'with' ensures the file is closed properly
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                # get_text("text") returns plain text
                # Other options: "dict" (structured), "html", "xml"
                text = page.get_text("text")
                
                # Clean up the text
                text = self._clean_text(text)
                
                pages.append(PageContent(
                    page_number=page_num,
                    text=text,
                    extraction_method="digital"
                ))
                
        logger.info(f"Extracted {len(pages)} pages with PyMuPDF")
        return pages
    
    def _extract_tables_with_pdfplumber(
        self, 
        file_path: Path, 
        pages: List[PageContent]
    ) -> None:
        """
        Extract tables using pdfplumber and add to existing pages.
        
        WHY PDFPLUMBER:
        - Better table detection than PyMuPDF
        - Can identify table boundaries and cells
        - Returns tables as lists of lists (like a 2D array)
        
        NOTE: This modifies the pages list in place
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, pdf_page in enumerate(pdf.pages):
                    # extract_tables() returns a list of tables
                    # Each table is a list of rows
                    # Each row is a list of cell values
                    tables = pdf_page.extract_tables()
                    
                    if tables and i < len(pages):
                        # Convert None values to empty strings
                        cleaned_tables = []
                        for table in tables:
                            cleaned_table = [
                                [cell if cell else "" for cell in row]
                                for row in table
                            ]
                            cleaned_tables.append(cleaned_table)
                        
                        pages[i].tables = cleaned_tables
                        
        except Exception as e:
            # Don't fail the whole extraction if table parsing fails
            logger.warning(f"Table extraction failed: {e}")
    
    def _extract_with_ocr(self, file_path: Path) -> List[PageContent]:
        """
        Extract text using OCR (Optical Character Recognition).
        
        WHEN THIS IS USED:
        - Scanned documents (pages are images, not text)
        - PDFs where regular extraction returns little/no text
        
        HOW OCR WORKS:
        1. Render each PDF page as an image
        2. Pass the image to Tesseract
        3. Tesseract identifies characters and returns text
        4. Also returns confidence score (how sure it is)
        """
        import pytesseract
        
        pages = []
        
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                # Render page to an image
                # Matrix(2, 2) = 2x zoom for better OCR accuracy
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Run OCR
                # image_to_data returns detailed info including confidence
                ocr_data = pytesseract.image_to_data(
                    img, 
                    lang=self.ocr_language,
                    output_type=pytesseract.Output.DICT
                )
                
                # Extract text and calculate average confidence
                text_parts = []
                confidences = []
                
                for i, word in enumerate(ocr_data["text"]):
                    if word.strip():  # Skip empty strings
                        text_parts.append(word)
                        conf = ocr_data["conf"][i]
                        if conf > 0:  # -1 means no confidence available
                            confidences.append(conf)
                
                text = " ".join(text_parts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                pages.append(PageContent(
                    page_number=page_num,
                    text=self._clean_text(text),
                    confidence=avg_confidence / 100,  # Convert to 0-1 range
                    extraction_method="ocr"
                ))
                
        logger.info(f"Extracted {len(pages)} pages with OCR")
        return pages
    
    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract PDF metadata (author, creation date, etc.).
        
        PDF METADATA:
        - title: Document title
        - author: Who created it
        - subject: Document subject
        - creator: Software used to create
        - creationDate: When it was created
        - modDate: When last modified
        """
        metadata = {}
        
        try:
            with fitz.open(file_path) as doc:
                # doc.metadata is a dictionary
                metadata = dict(doc.metadata) if doc.metadata else {}
                
                # Add file info
                metadata["file_size_bytes"] = file_path.stat().st_size
                metadata["page_count"] = doc.page_count
                
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing excess whitespace.
        
        WHY CLEANING IS NEEDED:
        - PDFs often have weird spacing
        - Multiple spaces, tabs, extra newlines
        - This normalizes it for consistent processing
        """
        if not text:
            return ""
        
        # Replace multiple whitespace with single space
        import re
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def parse_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> DocumentContent:
        """
        Parse a PDF from bytes (useful for uploaded files).
        
        WHY THIS EXISTS:
        When a user uploads a file via the API, we receive bytes.
        This method handles that case without writing to disk.
        
        Args:
            pdf_bytes: Raw PDF file bytes
            filename: Name to assign to the document
            
        Returns:
            DocumentContent with extracted text
        """
        # Create a BytesIO object (acts like a file in memory)
        pdf_stream = io.BytesIO(pdf_bytes)
        
        pages = []
        
        # PyMuPDF can open from bytes via stream parameter
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                text = self._clean_text(text)
                
                pages.append(PageContent(
                    page_number=page_num,
                    text=text,
                    extraction_method="digital"
                ))
        
        full_text = "\n\n".join(
            f"--- Page {page.page_number} ---\n{page.text}"
            for page in pages
        )
        
        return DocumentContent(
            filename=filename,
            total_pages=len(pages),
            pages=pages,
            full_text=full_text,
            metadata={"source": "bytes_upload"}
        )


# -----------------------------------------------------------------------------
# Convenience function for quick parsing
# -----------------------------------------------------------------------------
def parse_document(file_path: str | Path) -> DocumentContent:
    """
    Quick function to parse a document without creating a parser instance.
    
    USAGE:
        from document_parser import parse_document
        content = parse_document("claim.pdf")
        print(content.full_text)
    """
    parser = DocumentParser()
    return parser.parse(file_path)