"""
CLI command for validating extracted MNQ data.
"""

import click
from datetime import datetime, timedelta

from ..lib.logging_config import setup_logging
from ..services.data_validator import DataValidator


@click.command()
@click.option(
    "--db-path",
    default="data/tick_mbo_data.db",
    type=click.Path(),
    help="Path to DuckDB database file",
)
@click.option(
    "--parquet-dir",
    default="data",
    type=click.Path(),
    help="Directory containing Parquet files",
)
@click.option(
    "--days-back",
    default=70,
    type=int,
    help="Number of days back to validate (default: 70)",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def validate_data(db_path, parquet_dir, days_back, verbose):
    """
    Validate extracted MNQ tick and depth data.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger = setup_logging(log_level)

    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        logger.info(f"Validating data from {start_date.date()} to {end_date.date()}")

        # Initialize validator
        validator = DataValidator()

        # Run validation
        results = validator.validate_dataset(
            db_path=db_path,
            parquet_dir=parquet_dir,
            start_date=start_date,
            end_date=end_date,
        )

        # Display results
        click.echo("=== Validation Results ===")
        click.echo(f"Total ticks: {results['total_ticks']}")
        click.echo(f"Total depth records: {results['total_depth_records']}")
        click.echo(f"Duplicate ticks: {results['duplicate_ticks']}")
        click.echo(f"Completeness score: {results['completeness_score']:.2%}")

        if results["validation_errors"]:
            click.echo("\nErrors:")
            for error in results["validation_errors"]:
                click.echo(f"  - {error}")

        # Determine success
        is_valid = (
            results["completeness_score"] > 0.5
            and results["duplicate_ticks"] == 0
            and not results["validation_errors"]
        )

        if is_valid:
            logger.info("Data validation passed")
            click.echo("\n✓ Data validation passed")
        else:
            logger.warning("Data validation found issues")
            click.echo("\n⚠ Data validation found issues")

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        click.echo(f"✗ Data validation failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    validate_data()
