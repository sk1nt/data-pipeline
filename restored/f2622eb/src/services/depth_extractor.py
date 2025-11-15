"""
Depth data extraction service for MNQ.
"""

from pathlib import Path
from typing import List
from datetime import datetime, timedelta
import concurrent.futures

from ..lib.logging_config import logger
from ..models.depth_snapshot import DepthSnapshot
from ..lib.parquet_handler import ParquetHandler


class DepthExtractor:
    """
    Service for extracting depth data from SCID files.

    Note: This is a simplified implementation. Full depth extraction
    requires parsing separate .scdd files with incremental updates.
    """

    def __init__(self, scid_dir: str, output_dir: str = "data"):
        """
        Initialize depth extractor.

        Args:
            scid_dir: Directory containing SCID files
            output_dir: Directory for output Parquet files
        """
        self.scid_dir = Path(scid_dir)
        if not self.scid_dir.exists():
            raise FileNotFoundError(f"SCID directory not found: {scid_dir}")

        self.parquet_handler = ParquetHandler(output_dir)

    def extract_depth_for_date(self, date: datetime) -> List[DepthSnapshot]:
        """
        Extract depth data for a specific date.

        Note: This simplified version creates depth snapshots from tick data.
        Real implementation would parse .scdd files for full order book.

        Args:
            date: Date to extract

        Returns:
            List of depth snapshots
        """
        snapshots = []

        # Find SCID files for the date
        date_str = date.strftime("%Y%m%d")
        scid_files = list(self.scid_dir.glob(f"*{date_str}*.scid"))

        if not scid_files:
            logger.warning(f"No SCID files found for date {date_str}")
            return snapshots

        for scid_file in scid_files:
            logger.info(f"Processing SCID file for depth: {scid_file}")
            try:
                # For now, create mock depth from tick data
                # In practice, this would parse .scdd files
                mock_records = self._create_mock_depth_records(scid_file)
                snapshots.extend(mock_records)

            except Exception as e:
                logger.error(f"Error processing depth for {scid_file}: {e}")
                continue

        return snapshots

    def _create_mock_depth_records(self, scid_file: Path) -> List[DepthSnapshot]:
        """
        Create mock depth records from SCID file.

        This is a placeholder - real implementation needs .scdd parsing.

        Args:
            scid_file: SCID file path

        Returns:
            List of depth snapshots
        """
        # Simplified: create one snapshot per minute
        # In reality, depth updates much more frequently
        snapshots = []

        base_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)

        for i in range(390):  # 6.5 hours of trading
            timestamp = base_time + timedelta(minutes=i)

            # Mock depth levels
            snapshot = DepthSnapshot(
                timestamp=timestamp,
                bid_price_1=4500.00 - (i * 0.01),  # Declining prices
                bid_size_1=100 + (i % 50),
                ask_price_1=4500.25 + (i * 0.01),  # Rising prices
                ask_size_1=150 + (i % 30)
            )
            snapshots.append(snapshot)

        return snapshots

    def extract_depth_date_range(self, start_date: datetime, end_date: datetime) -> List[DepthSnapshot]:
        """
        Extract depth data for a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of depth snapshots
        """
        all_snapshots = []
        current_date = start_date

        while current_date <= end_date:
            snapshots = self.extract_depth_for_date(current_date)
            all_snapshots.extend(snapshots)
            current_date += timedelta(days=1)

        logger.info(f"Total depth snapshots extracted: {len(all_snapshots)}")
        return all_snapshots

    def save_depth_to_parquet(self, snapshots: List[DepthSnapshot], date: datetime):
        """
        Save depth snapshots to Parquet file.

        Args:
            snapshots: List of depth snapshots
            date: Date for the file
        """
        if not snapshots:
            logger.warning("No depth snapshots to save")
            return

        file_path = self.parquet_handler.save_depth_data(snapshots, date)
        logger.info(f"Saved {len(snapshots)} depth snapshots to {file_path}")

    def extract_and_save_parallel(self, start_date: datetime, end_date: datetime, max_workers: int = 4):
        """
        Extract depth for date range sequentially and save to Parquet.

        Args:
            start_date: Start date
            end_date: End date
            max_workers: Maximum number of parallel workers (ignored, sequential processing)
        """
        logger.info(f"Starting sequential depth extraction from {start_date.date()} to {end_date.date()}")

        # Generate list of dates to process
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)

        logger.info(f"Processing {len(dates)} days sequentially")

        # Process dates sequentially
        for date in dates:
            try:
                self._extract_and_save_single_date(date)
                logger.info(f"Completed depth extraction for {date.date()}")
            except Exception as e:
                logger.error(f"Failed to extract depth for {date.date()}: {e}")

        logger.info("Sequential depth extraction completed")

    def _extract_and_save_single_date(self, date: datetime):
        """
        Extract and save depth data for a single date.

        Args:
            date: Date to process
        """
        snapshots = self.extract_depth_for_date(date)
        if snapshots:
            self.save_depth_to_parquet(snapshots, date)