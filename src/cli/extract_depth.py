"""
CLI command for extracting MNQ depth data.
"""

import click
from datetime import datetime, timedelta

from ..lib.logging_config import setup_logging
from ..services.depth_extractor import DepthExtractor


@click.command()
@click.option(
    "--scid-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing SCID files",
)
@click.option(
    "--days-back",
    default=70,
    type=int,
    help="Number of days back to extract (default: 70)",
)
@click.option(
    "--output-dir",
    default="data",
    type=click.Path(),
    help="Output directory for Parquet files",
)
@click.option(
    "--parallel", is_flag=True, help="Use parallel processing for faster extraction"
)
@click.option(
    "--workers", default=4, type=int, help="Number of parallel workers (default: 4)"
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def extract_depth(scid_dir, days_back, output_dir, verbose, parallel, workers):
    """
    Extract MNQ market depth data from SierraChart SCID files.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger = setup_logging(log_level)

    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        logger.info(f"Extracting depth from {start_date.date()} to {end_date.date()}")

        # Initialize extractor
        extractor = DepthExtractor(scid_dir, output_dir)

        # Extract and save
        if parallel:
            logger.info(f"Using parallel processing with {workers} workers")
            extractor.extract_and_save_parallel(
                start_date, end_date, max_workers=workers
            )
        else:
            extractor.extract_and_save(start_date, end_date)

        logger.info("Depth extraction completed successfully")
        click.echo("✓ Depth extraction completed successfully")

    except Exception as e:
        logger.error(f"Depth extraction failed: {e}")
        click.echo(f"✗ Depth extraction failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    extract_depth()
