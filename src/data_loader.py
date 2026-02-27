"""
data_loader.py - Load Rate Data into Vector Store

WHY THIS FILE EXISTS:
- Load market rate data from Excel/CSV files
- Parse GTA (General Terms of Agreement) rate tables
- Populate the ChromaDB vector store for RAG queries
- Support various data formats (Excel, CSV)

GTA RATE FILE STRUCTURE (typical):
- Vehicle groups (A, B, C, D, E, F, G, H, I)
- Daily, weekly, and monthly rates
- May have regional variations
- Effective dates for rate validity

USAGE:
    # From command line
    python -m src.data_loader
    
    # Or import and use
    from data_loader import load_gta_rates
    load_gta_rates("data/rates.xlsx")
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from vector_store import get_rates_store, VectorStore
from config import settings, get_absolute_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class RateRecord:
    """
    A single market rate record.
    
    This is what gets stored in the vector database.
    """
    vehicle_group: str
    daily_rate: float
    weekly_rate: Optional[float] = None
    monthly_rate: Optional[float] = None
    region: str = "UK"
    company: str = "GTA"
    year: int = 2025
    source: str = ""
    
    def to_text(self) -> str:
        """
        Convert to text for embedding.
        
        This text is what gets embedded and searched.
        """
        parts = [
            f"Group {self.vehicle_group} vehicle hire",
            f"in {self.region}",
            f"£{self.daily_rate:.2f} per day"
        ]
        
        if self.weekly_rate:
            parts.append(f"£{self.weekly_rate:.2f} per week")
        
        parts.append(f"{self.year}")
        parts.append(f"from {self.company}")
        
        return ", ".join(parts)
    
    def to_metadata(self) -> Dict[str, Any]:
        """
        Convert to metadata dict for filtering.
        """
        metadata = {
            "vehicle_group": self.vehicle_group,
            "daily_rate": self.daily_rate,
            "region": self.region,
            "company": self.company,
            "year": self.year,
            "source": self.source
        }
        
        if self.weekly_rate:
            metadata["weekly_rate"] = self.weekly_rate
        if self.monthly_rate:
            metadata["monthly_rate"] = self.monthly_rate
        
        return metadata


class GTADataLoader:
    """
    Loader for GTA (General Terms of Agreement) rate files.
    
    GTA files typically contain:
    - Vehicle classification groups (A-I)
    - Standard rates for each group
    - May have multiple sheets for different rate types
    
    USAGE:
        loader = GTADataLoader()
        rates = loader.load_excel("data/gta_rates.xlsx")
        loader.save_to_vectorstore(rates)
    """
    
    # Standard vehicle groups
    VEHICLE_GROUPS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the loader.
        
        Args:
            vector_store: Vector store to save rates to
        """
        self.vector_store = vector_store or get_rates_store()
    
    def load_excel(self, file_path: str | Path) -> List[RateRecord]:
        """
        Load rates from an Excel file.
        
        This attempts to parse various GTA file formats.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            List of RateRecord objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading rates from: {file_path.name}")
        
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
        
        all_rates = []
        
        for sheet_name in sheet_names:
            logger.info(f"Processing sheet: {sheet_name}")
            
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Try to parse rates from this sheet
                rates = self._parse_sheet(df, sheet_name, file_path.name)
                
                if rates:
                    logger.info(f"  Found {len(rates)} rates in '{sheet_name}'")
                    all_rates.extend(rates)
                else:
                    logger.info(f"  No rates found in '{sheet_name}'")
                    
            except Exception as e:
                logger.warning(f"  Error processing sheet '{sheet_name}': {e}")
        
        logger.info(f"Total rates loaded: {len(all_rates)}")
        
        return all_rates
    
    def _parse_sheet(
        self, 
        df: pd.DataFrame, 
        sheet_name: str,
        source_file: str
    ) -> List[RateRecord]:
        """
        Parse a single Excel sheet to extract rates.
        
        This uses heuristics to find rate data in various formats.
        """
        rates = []
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Print first few rows for debugging
        logger.debug(f"Sheet columns: {list(df.columns)}")
        logger.debug(f"First rows:\n{df.head()}")
        
        # Strategy 1: Look for 'Group' column
        rates.extend(self._parse_by_group_column(df, sheet_name, source_file))
        
        # Strategy 2: Look for columns named A, B, C, etc.
        if not rates:
            rates.extend(self._parse_by_letter_columns(df, sheet_name, source_file))
        
        # Strategy 3: Parse row by row looking for patterns
        if not rates:
            rates.extend(self._parse_by_pattern(df, sheet_name, source_file))
        
        return rates
    
    def _parse_by_group_column(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        source_file: str
    ) -> List[RateRecord]:
        """
        Parse when there's a 'Group' column with A, B, C, etc.
        """
        rates = []
        
        # Find the group column
        group_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'group' in col_lower or 'class' in col_lower or 'category' in col_lower:
                group_col = col
                break
        
        if group_col is None:
            return rates
        
        # Find rate columns
        rate_col = None
        weekly_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'daily' in col_lower or 'day' in col_lower or 'rate' in col_lower:
                if rate_col is None:
                    rate_col = col
            if 'week' in col_lower:
                weekly_col = col
        
        if rate_col is None:
            # Try to find any numeric column
            for col in df.columns:
                if col != group_col and pd.api.types.is_numeric_dtype(df[col]):
                    rate_col = col
                    break
        
        if rate_col is None:
            return rates
        
        # Extract rates
        for _, row in df.iterrows():
            group = str(row.get(group_col, "")).strip().upper()
            
            # Check if it's a valid vehicle group
            if group in self.VEHICLE_GROUPS or any(g in group for g in self.VEHICLE_GROUPS):
                # Extract the letter
                for g in self.VEHICLE_GROUPS:
                    if g in group:
                        group = g
                        break
                
                # Get rate value
                daily_rate = row.get(rate_col)
                
                if pd.notna(daily_rate) and self._is_valid_rate(daily_rate):
                    weekly_rate = None
                    if weekly_col and pd.notna(row.get(weekly_col)):
                        weekly_rate = float(row.get(weekly_col))
                    
                    rates.append(RateRecord(
                        vehicle_group=group,
                        daily_rate=float(daily_rate),
                        weekly_rate=weekly_rate,
                        region="UK",
                        company="GTA",
                        year=2025,
                        source=f"{source_file}:{sheet_name}"
                    ))
        
        return rates
    
    def _parse_by_letter_columns(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        source_file: str
    ) -> List[RateRecord]:
        """
        Parse when columns are named A, B, C, D, etc.
        """
        rates = []
        
        # Check if we have letter columns
        letter_cols = [col for col in df.columns if str(col).strip().upper() in self.VEHICLE_GROUPS]
        
        if not letter_cols:
            return rates
        
        # Look for rows with 'daily' or 'rate' labels
        for idx, row in df.iterrows():
            # Check first column for label
            first_val = str(row.iloc[0]).lower() if pd.notna(row.iloc[0]) else ""
            
            is_daily_row = 'daily' in first_val or 'day' in first_val
            is_weekly_row = 'week' in first_val
            
            if is_daily_row:
                for col in letter_cols:
                    group = str(col).strip().upper()
                    rate = row.get(col)
                    
                    if pd.notna(rate) and self._is_valid_rate(rate):
                        rates.append(RateRecord(
                            vehicle_group=group,
                            daily_rate=float(rate),
                            region="UK",
                            company="GTA",
                            year=2025,
                            source=f"{source_file}:{sheet_name}"
                        ))
        
        return rates
    
    def _parse_by_pattern(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        source_file: str
    ) -> List[RateRecord]:
        """
        Parse by looking for patterns like "Group A" followed by numbers.
        """
        rates = []
        
        # Convert entire dataframe to string and search
        for idx, row in df.iterrows():
            row_str = " ".join(str(v) for v in row.values if pd.notna(v))
            
            for group in self.VEHICLE_GROUPS:
                # Look for "Group A" or just "A" followed by a number
                import re
                
                patterns = [
                    rf'Group\s*{group}\D+(\d+\.?\d*)',
                    rf'\b{group}\s+(\d+\.?\d*)',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, row_str, re.IGNORECASE)
                    if match:
                        rate = float(match.group(1))
                        if self._is_valid_rate(rate):
                            # Check if we already have this group
                            if not any(r.vehicle_group == group for r in rates):
                                rates.append(RateRecord(
                                    vehicle_group=group,
                                    daily_rate=rate,
                                    region="UK",
                                    company="GTA",
                                    year=2025,
                                    source=f"{source_file}:{sheet_name}"
                                ))
                        break
        
        return rates
    
    def _is_valid_rate(self, value) -> bool:
        """
        Check if a value looks like a valid daily rate.
        
        Valid rates are typically between £10 and £500 per day.
        """
        try:
            rate = float(value)
            return 10 <= rate <= 500
        except (ValueError, TypeError):
            return False
    
    def save_to_vectorstore(self, rates: List[RateRecord]) -> int:
        """
        Save rate records to the vector store.
        
        Args:
            rates: List of RateRecord objects
            
        Returns:
            Number of records saved
        """
        if not rates:
            logger.warning("No rates to save")
            return 0
        
        # Convert to documents and metadata
        documents = [rate.to_text() for rate in rates]
        metadatas = [rate.to_metadata() for rate in rates]
        
        # Generate IDs based on content
        import hashlib
        ids = [
            hashlib.md5(f"{rate.vehicle_group}_{rate.daily_rate}_{rate.region}_{i}".encode()).hexdigest()[:16]
            for i, rate in enumerate(rates)
        ]
        
        # Add to vector store
        self.vector_store.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Saved {len(rates)} rates to vector store")
        logger.info(f"Total records in store: {self.vector_store.count}")
        
        return len(rates)
    
    def clear_vectorstore(self) -> None:
        """
        Clear all rates from the vector store.
        
        WARNING: This deletes all data!
        """
        self.vector_store.clear()
        logger.info("Vector store cleared")


def load_gta_rates(file_path: str | Path) -> int:
    """
    Load GTA rates from a file into the vector store.
    
    USAGE:
        from data_loader import load_gta_rates
        
        count = load_gta_rates("data/gta_rates.xlsx")
        print(f"Loaded {count} rates")
    """
    loader = GTADataLoader()
    rates = loader.load_excel(file_path)
    return loader.save_to_vectorstore(rates)


def load_csv_rates(file_path: str | Path, has_header: bool = True) -> int:
    """
    Load rates from a CSV file.
    
    Expected columns:
    - vehicle_group (or group)
    - daily_rate (or rate)
    - region (optional)
    - company (optional)
    - year (optional)
    """
    file_path = Path(file_path)
    
    df = pd.read_csv(file_path, header=0 if has_header else None)
    
    # Normalize column names
    df.columns = [str(col).lower().strip().replace(" ", "_") for col in df.columns]
    
    # Find columns
    group_col = next((c for c in df.columns if 'group' in c), None)
    rate_col = next((c for c in df.columns if 'rate' in c or 'daily' in c), None)
    region_col = next((c for c in df.columns if 'region' in c), None)
    company_col = next((c for c in df.columns if 'company' in c), None)
    year_col = next((c for c in df.columns if 'year' in c), None)
    
    if not group_col or not rate_col:
        raise ValueError("CSV must have 'group' and 'rate' columns")
    
    rates = []
    
    for _, row in df.iterrows():
        group = str(row[group_col]).strip().upper()
        
        if group in GTADataLoader.VEHICLE_GROUPS:
            rates.append(RateRecord(
                vehicle_group=group,
                daily_rate=float(row[rate_col]),
                region=row[region_col] if region_col and pd.notna(row[region_col]) else "UK",
                company=row[company_col] if company_col and pd.notna(row[company_col]) else "Unknown",
                year=int(row[year_col]) if year_col and pd.notna(row[year_col]) else 2025,
                source=file_path.name
            ))
    
    loader = GTADataLoader()
    return loader.save_to_vectorstore(rates)


def generate_sample_rates() -> int:
    """
    Generate sample rate data for testing.
    
    Creates synthetic but realistic rate data across
    all vehicle groups and regions.
    """
    import random
    
    # Base rates by vehicle group (realistic UK market rates)
    base_rates = {
        "A": 35,   # Small city car (Fiat 500, Toyota Aygo)
        "B": 42,   # Supermini (Ford Fiesta, VW Polo)
        "C": 52,   # Compact (Ford Focus, VW Golf)
        "D": 65,   # Mid-size (BMW 3 Series, Audi A4)
        "E": 85,   # Executive (BMW 5 Series, Mercedes E-Class)
        "F": 110,  # Luxury (BMW 7 Series, Mercedes S-Class)
        "G": 75,   # Estate/Wagon
        "H": 95,   # SUV
        "I": 130,  # Premium SUV/Sports
    }
    
    regions = [
        "London", "Manchester", "Birmingham", "Leeds", "Liverpool",
        "Newcastle", "Bristol", "Edinburgh", "Glasgow", "Cardiff",
        "South East", "South West", "East Midlands", "West Midlands",
        "Yorkshire", "North West", "North East", "Scotland", "Wales"
    ]
    
    companies = [
        "Enterprise", "Hertz", "Avis", "Budget", "Europcar",
        "Sixt", "Thrifty", "National", "Alamo", "Dollar"
    ]
    
    rates = []
    
    for group, base_rate in base_rates.items():
        for region in regions:
            # Regional variation (-10% to +20%)
            regional_factor = 1.0
            if "London" in region:
                regional_factor = 1.15  # London premium
            elif region in ["South East", "Edinburgh"]:
                regional_factor = 1.08
            elif region in ["Wales", "North East"]:
                regional_factor = 0.92
            
            for company in random.sample(companies, k=random.randint(3, 6)):
                # Company variation (-5% to +10%)
                company_factor = random.uniform(0.95, 1.10)
                
                daily_rate = round(base_rate * regional_factor * company_factor, 2)
                
                # Weekly rate (typically 6x daily)
                weekly_rate = round(daily_rate * 6, 2)
                
                rates.append(RateRecord(
                    vehicle_group=group,
                    daily_rate=daily_rate,
                    weekly_rate=weekly_rate,
                    region=region,
                    company=company,
                    year=2025,
                    source="synthetic_sample"
                ))
    
    logger.info(f"Generated {len(rates)} sample rates")
    
    loader = GTADataLoader()
    return loader.save_to_vectorstore(rates)


def print_summary():
    """
    Print a summary of what's in the vector store.
    """
    store = get_rates_store()
    
    print("\n" + "=" * 50)
    print("VECTOR STORE SUMMARY")
    print("=" * 50)
    print(f"Total records: {store.count}")
    
    # Sample a few records
    if store.count > 0:
        results = store.search("vehicle hire rate", top_k=5)
        
        print("\nSample records:")
        for r in results.results:
            print(f"  - {r.document}")
            print(f"    Metadata: {r.metadata}")
    
    print("=" * 50 + "\n")


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load rate data into vector store")
    parser.add_argument(
        "--file",
        type=str,
        help="Path to Excel or CSV file to load"
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate sample rate data"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the vector store before loading"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of vector store contents"
    )
    
    args = parser.parse_args()
    
    loader = GTADataLoader()
    
    # Clear if requested
    if args.clear:
        loader.clear_vectorstore()
    
    # Load from file
    if args.file:
        file_path = Path(args.file)
        
        if not file_path.exists():
            # Try relative to data folder
            file_path = get_absolute_path(Path("data") / args.file)
        
        if file_path.suffix in [".xlsx", ".xls"]:
            rates = loader.load_excel(file_path)
            loader.save_to_vectorstore(rates)
        elif file_path.suffix == ".csv":
            load_csv_rates(file_path)
        else:
            print(f"Unsupported file type: {file_path.suffix}")
    
    # Generate sample data
    if args.generate_sample:
        generate_sample_rates()
    
    # Print summary
    if args.summary or args.file or args.generate_sample:
        print_summary()
    
    # If no arguments, show help
    if not any([args.file, args.generate_sample, args.clear, args.summary]):
        parser.print_help()
        print("\nExamples:")
        print("  python -m src.data_loader --file data/GTA-Grouping-Car-Rates-wef-1st-July-2025-v-01_07_2025-FINAL-17-2-26.xlsx")
        print("  python -m src.data_loader --generate-sample")
        print("  python -m src.data_loader --summary")
        print("  python -m src.data_loader --clear --generate-sample")
