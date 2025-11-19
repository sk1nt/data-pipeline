"""
Combined tick data extraction for MNQ - CLI and service in one.
"""

from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
import sys
import pytz

from ..lib.scid_parser import parse_scid_file_backwards_generator, is_tick_record
from ..lib.logging_config import setup_logging, logger
from ..models.mnq_tick_record import MnqTickRecord
from ..lib.database import db
from ..services.depth_extractor import DepthExtractor


def extract_futures_ticks(scid_dir: str, days_back: int = 70,
                      parallel: bool = True, workers: int = 10, verbose: bool = False, max_recent_records: int = 0):
    """
    Extract futures tick data with optimized backwards reading.

    Args:
        scid_dir: Directory containing SCID files
        days_back: Number of days back to extract
        chunk_size_mb: Chunk size in MB for backwards reading
        parallel: Use parallel processing
        workers: Number of parallel workers
        verbose: Enable verbose logging
        max_recent_records: Maximum recent records to process per day (for memory efficiency)
    """
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(log_level)

    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Convert max_recent_records: 0 means unlimited
        actual_max_records = None if max_recent_records == 0 else max_recent_records

        logger.info(f"Extracting ticks from {start_date.date()} to {end_date.date()}")
        logger.info(f"Using {workers} workers, max_recent_records={max_recent_records} ({'unlimited' if max_recent_records == 0 else 'limited'}), parallel={parallel}")

        # Initialize extractors
        tick_extractor = TickExtractor(scid_dir)
        depth_extractor = DepthExtractor(scid_dir)

        # Extract and save ticks
        if parallel:
            tick_extractor.extract_and_save_parallel(start_date, end_date, max_workers=workers, max_recent_records=actual_max_records)
        else:
            ticks = tick_extractor.extract_ticks_date_range(start_date, end_date, actual_max_records)
            tick_extractor.save_ticks_to_db_optimized(ticks)

        # Extract and save depth data
        logger.info("Starting depth data extraction...")
        depth_extractor.extract_and_save_parallel(start_date, end_date, max_workers=workers)

        logger.info("✓ Tick and depth extraction completed successfully")
        print("✓ Tick and depth extraction completed successfully")

    except Exception as e:
        logger.error(f"✗ Tick extraction failed: {e}")
        print(f"✗ Tick extraction failed: {e}", file=sys.stderr)
        sys.exit(1)


class TickExtractor:
    """
    Core tick extraction logic.
    """

    def __init__(self, scid_dir: str):
        self.scid_dir = Path(scid_dir)
        if not self.scid_dir.exists():
            raise FileNotFoundError(f"SCID directory not found: {scid_dir}")

    def _normalize_futures_ticker(self, filename: str) -> str:
        """
        Normalize futures contract filename to base ticker name.
        
        Examples:
            MNQ.scid -> MNQ
            MNQM25-CME.scid -> MNQ
            MNQM25_FUT_CME.scid -> MNQ
            MNQU25-CME.scid -> MNQ
        
        Args:
            filename: SCID filename
            
        Returns:
            Normalized base ticker
        """
        import re
        
        # Extract base futures symbol - stop before month code (letter) + year (digits)
        # Pattern: letters followed by letter+digits (month/year), or just letters
        match = re.match(r'^([A-Z]+?)(?=[A-Z]\d|$)', filename)
        if match:
            base = match.group(1)
            # Common futures mappings - ensure we return the standard symbol
            futures_map = {
                'MNQ': 'MNQ',  # Micro E-mini Nasdaq
                'ES': 'ES',    # E-mini S&P 500
                'NQ': 'NQ',    # E-mini Nasdaq 100
                'CL': 'CL',    # Crude Oil
                'GC': 'GC',    # Gold
                'SI': 'SI',    # Silver
                'HG': 'HG',    # Copper
                'NG': 'NG',    # Natural Gas
                'ZB': 'ZB',    # Treasury Bond
                'ZN': 'ZN',    # Treasury Note 10Y
                'ZF': 'ZF',    # Treasury Note 5Y
                'ZT': 'ZT',    # Treasury Note 2Y
                'GE': 'GE',    # Eurodollar
                'EUR': 'EUR',  # Euro FX
                'JPY': 'JPY',  # Japanese Yen
                'GBP': 'GBP',  # British Pound
                'CHF': 'CHF',  # Swiss Franc
                'CAD': 'CAD',  # Canadian Dollar
                'AUD': 'AUD',  # Australian Dollar
                'NZD': 'NZD',  # New Zealand Dollar
            }
            return futures_map.get(base, base)
        
        # Fallback - take first 2-4 letters
        match = re.match(r'^([A-Z]{2,4})', filename)
        if match:
            return match.group(1)
        
        # Fallback
        return filename.replace('.scid', '').split('-')[0].split('_')[0]

    def extract_ticks_for_date(self, date: datetime, max_recent_records: Optional[int] = None) -> List[MnqTickRecord]:
        """Extract tick data for a specific date by scanning into memory first."""
        ticks = []
        mnq_files = list(self.scid_dir.glob("MNQ*.scid"))

        if not mnq_files:
            logger.warning(f"No MNQ SCID files found in {self.scid_dir}")
            return ticks

        # Use the most recent file (largest/last modified)
        scid_file = max(mnq_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Using SCID file: {scid_file}")

        # Extract base ticker from filename
        ticker = self._normalize_futures_ticker(scid_file.name)
        logger.info(f"Normalized ticker: {ticker}")

        try:
            # Scan all records for this date into memory, then filter
            logger.info(f"Scanning records for {date.date()} into memory...")
            
            # 4:15 PM Eastern = 8:15 PM UTC
            eastern_tz = pytz.timezone('US/Eastern')
            market_open_eastern = eastern_tz.localize(datetime.combine(date.date(), datetime.strptime('16:15', '%H:%M').time()))
            market_open_utc = market_open_eastern.astimezone(pytz.UTC).replace(tzinfo=None)
            
            logger.info(f"Filtering for records after {market_open_utc} (4:15 PM Eastern)")

            # Load all records for this date into memory
            day_records = []
            for record in parse_scid_file_backwards_generator(str(scid_file), date, max_recent_records):
                day_records.append(record)

            logger.info(f"Loaded {len(day_records)} records for {date.date()}")

            # Filter for market hours (after 4:15 PM Eastern / 8:15 PM UTC)
            market_records = [r for r in day_records if r['timestamp'] >= market_open_utc]
            logger.info(f"Filtered to {len(market_records)} records after market open")

            # Extract ticks from filtered records
            tick_count = 0
            for record in market_records:
                if is_tick_record(record):
                    tick = MnqTickRecord.from_scid_record(record)
                    tick.ticker = ticker
                    ticks.append(tick)
                    tick_count += 1

            logger.info(f"Extracted {tick_count} ticks from {len(market_records)} market records")

        except Exception as e:
            logger.error(f"Error processing {scid_file}: {e}")

        return ticks

    def _extract_ticks_from_file(self, scid_file: Path, date: datetime, max_recent_records: Optional[int]) -> List[MnqTickRecord]:
        """Extract ticks from a single SCID file."""
        ticks = []
        
        # Extract base ticker from filename (normalize futures contract to base name)
        ticker = self._normalize_futures_ticker(scid_file.name)
        
        try:
            # Use generator with max_records limit for efficiency
            record_count = 0
            tick_count = 0

            for record in parse_scid_file_backwards_generator(str(scid_file), date, max_recent_records):
                record_count += 1
                if is_tick_record(record):
                    tick = MnqTickRecord.from_scid_record(record)
                    tick.ticker = ticker  # Set normalized ticker
                    ticks.append(tick)
                    tick_count += 1

            logger.debug(f"Processed {record_count} records from {scid_file.name}, extracted {tick_count} ticks")

        except Exception as e:
            logger.error(f"Error processing {scid_file}: {e}")

        return ticks

    def extract_ticks_date_range(self, start_date: datetime, end_date: datetime, max_recent_records: Optional[int] = None) -> List[MnqTickRecord]:
        """Extract tick data for a date range."""
        all_ticks = []
        current_date = start_date

        while current_date <= end_date:
            ticks = self.extract_ticks_for_date(current_date, max_recent_records)
            all_ticks.extend(ticks)
            current_date += timedelta(days=1)

        logger.info(f"Total ticks extracted: {len(all_ticks)}")
        return all_ticks

    def save_ticks_to_db_optimized(self, ticks: List[MnqTickRecord]):
        """Save tick records to DuckDB with transaction optimization."""
        if not ticks:
            logger.warning("No ticks to save")
            return

        with db.connect() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                data = [(tick.timestamp, tick.price, tick.volume, tick.tick_type, tick.ticker) for tick in ticks]
                conn.executemany("INSERT INTO mnq_ticks (timestamp, price, volume, tick_type, ticker) VALUES (?, ?, ?, ?, ?)", data)
                conn.execute("COMMIT")
                logger.info(f"Saved {len(ticks)} ticks to database (optimized)")
            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"Failed to save ticks: {e}")
                raise

    def extract_and_save_parallel(self, start_date: datetime, end_date: datetime, max_workers: int = 4, max_recent_records: Optional[int] = None):
        """Extract ticks for date range sequentially."""
        logger.info(f"Starting sequential tick extraction from {start_date.date()} to {end_date.date()}")

        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)

        logger.info(f"Processing {len(dates)} days sequentially")

        all_ticks = []
        for date in dates:
            ticks = self.extract_ticks_for_date(date, max_recent_records)
            all_ticks.extend(ticks)
            logger.info(f"Completed {date.date()}: {len(ticks)} ticks")

        self.save_ticks_to_db_optimized(all_ticks)
        logger.info("Sequential tick extraction completed")


# CLI entry point
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract futures tick and depth data')
    parser.add_argument('--scid-dir', required=True, help='Directory containing SCID files')
    parser.add_argument('--days-back', type=int, default=70, help='Number of days back to extract')
    parser.add_argument('--date', help='Specific date to extract (YYYY-MM-DD format)')
    # chunk-size removed - parser reads one record at a time now
    parser.add_argument('--parallel', action='store_true', default=True, help='Use parallel processing')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    parser.add_argument('--tick-workers', type=int, default=1, help='Number of workers for tick extraction')
    parser.add_argument('--depth-workers', type=int, default=1, help='Number of workers for depth extraction')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--max-recent-records', type=int, default=0, help='Maximum recent records to process per day (0 = unlimited)')

    args = parser.parse_args()

    # Calculate date range
    if args.date:
        # Specific date mode
        from datetime import datetime
        start_date = end_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        # Days back mode
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days_back)

    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)

    logger.info(f"Extracting data from {start_date.date()} to {end_date.date()}")
    logger.info(f"Using tick_workers={args.tick_workers}, depth_workers={args.depth_workers}, max_recent_records={args.max_recent_records} ({'unlimited' if args.max_recent_records == 0 else 'limited'})")

    # Initialize extractors
    tick_extractor = TickExtractor(args.scid_dir)
    depth_extractor = DepthExtractor(args.scid_dir)

    # Extract and save ticks
    logger.info("Starting tick data extraction...")
    tick_extractor.extract_and_save_parallel(start_date, end_date, max_workers=args.tick_workers, chunk_size_mb=args.chunk_size, max_recent_records=0 if args.max_recent_records == 0 else args.max_recent_records)

    # Extract and save depth data
    logger.info("Starting depth data extraction...")
    depth_extractor.extract_and_save_parallel(start_date, end_date, max_workers=args.depth_workers)

    logger.info("✓ Tick and depth extraction completed successfully")
    print("✓ Tick and depth extraction completed successfully")
