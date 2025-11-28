#!/usr/bin/env python3
"""Data Import Script: Import tick and GEX data from legacy exports.

This script imports data from a legacy market-data export directory (default:
data/legacy_source), focusing on:
- GEX data for NQ_NDX (full gex_zero data)
- Baseline tick data for MNQ and NQ
- Market depth data for MNQ

Usage:
    python import_data.py --dry-run    # Preview what would be imported
    python import_data.py --import     # Perform actual import
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging
import sys
import shutil

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger(__name__)

# Import project modules
try:
    PROJECT_MODULES_AVAILABLE = False  # Not used for GEX only import
except ImportError:
    PROJECT_MODULES_AVAILABLE = False

# Import new modules
try:
    from src.db.duckdb_utils import DuckDBUtils
    from src.lineage.lineage_tracker import LineageTracker

    NEW_MODULES_AVAILABLE = True
except ImportError:
    LOG.warning("New modules not available, falling back to legacy import")
    NEW_MODULES_AVAILABLE = False


class DataImporter:
    """Handles importing data from legacy exports."""

    def __init__(self, legacy_source_path: str, dry_run: bool = False):
        self.legacy_source_path = Path(legacy_source_path)
        self.dry_run = dry_run
        self.db_utils = (
            DuckDBUtils(db_path="data/gex_data.db") if NEW_MODULES_AVAILABLE else None
        )
        self.lineage_tracker = LineageTracker() if NEW_MODULES_AVAILABLE else None
        self.parquet_output_dir = Path("data/parquet/gexbot")
        self.stats = {
            "gex_files_processed": 0,
            "gex_records_imported": 0,
            "parquet_partitions_written": 0,
            "errors": [],
        }

    def scan_data_sources(self) -> Dict[str, Any]:
        """Scan available GEX data sources only."""
        sources = {
            "gex_data": {
                "path": self.legacy_source_path / "gex_bridge/history",
                "files": [],
                "description": "GEX snapshots for NQ_NDX from gex_bridge/history",
            }
        }

        # Scan for GEX files
        gex_path = sources["gex_data"]["path"]
        if gex_path.exists():
            json_files = list(gex_path.glob("**/*.json"))
            sources["gex_data"]["files"] = json_files
            sources["gex_data"]["count"] = len(json_files)
        else:
            sources["gex_data"]["count"] = 0
            LOG.warning(f"GEX path not found: {gex_path}")

        return sources

    def import_gex_data(self) -> None:
        """Import GEX data for NQ_NDX from gex_bridge/history only."""
        LOG.info("Starting GEX data import for NQ_NDX from gex_bridge/history...")

        gex_path = self.legacy_source_path / "gex_bridge/history"
        if not gex_path.exists():
            LOG.warning(f"GEX path not found: {gex_path}")
            return

        if not (NEW_MODULES_AVAILABLE and self.db_utils and self.lineage_tracker):
            LOG.error("Required modules unavailable; cannot perform GEX import")
            return

        json_files = sorted(gex_path.glob("*.json"))
        LOG.info(f"Found {len(json_files)} GEX JSON files")
        self.stats["gex_files_processed"] = len(json_files)

        snapshot_schema = """
        timestamp BIGINT,
        ticker VARCHAR,
        spot_price DOUBLE,
        zero_gamma DOUBLE,
        net_gex DOUBLE,
        min_dte INTEGER,
        sec_min_dte INTEGER,
        major_pos_vol DOUBLE,
        major_pos_oi DOUBLE,
        major_neg_vol DOUBLE,
        major_neg_oi DOUBLE,
        sum_gex_vol DOUBLE,
        sum_gex_oi DOUBLE,
        delta_risk_reversal DOUBLE,
        max_priors VARCHAR,
        strikes VARCHAR
        """
        strike_schema = """
        timestamp BIGINT,
        ticker VARCHAR,
        strike DOUBLE,
        gamma DOUBLE,
        oi_gamma DOUBLE,
        priors VARCHAR
        """
        self.db_utils.create_table_if_not_exists("gex_snapshots", snapshot_schema)
        self.db_utils.create_table_if_not_exists("gex_strikes", strike_schema)

        if self.dry_run:
            with self.db_utils:
                snapshot_total = 0
                strike_total = 0
                for file_path in json_files:
                    source = self._build_json_relation(file_path)
                    snap_cnt, strike_cnt = self.db_utils.conn.execute(
                        f"SELECT COUNT(*) AS snapshots, "
                        f"COALESCE(SUM(list_count(strikes)), 0) AS strike_cnt "
                        f"FROM {source}"
                    ).fetchone()
                    snapshot_total += snap_cnt
                    strike_total += strike_cnt

            LOG.info(
                f"Dry run: {snapshot_total} snapshots, {strike_total} strike rows detected"
            )
            self.stats["gex_records_imported"] = snapshot_total
            return

        with self.db_utils:
            conn = self.db_utils.conn
            conn.execute("BEGIN TRANSACTION")
            # Ensure new fields present in schema
            try:
                conn.execute(
                    "ALTER TABLE gex_snapshots ADD COLUMN IF NOT EXISTS strikes VARCHAR"
                )
            except Exception:
                pass
            conn.execute("DELETE FROM gex_snapshots")
            conn.execute("DELETE FROM gex_strikes")

            snapshot_insert_template = """
            INSERT INTO gex_snapshots
            SELECT
                CAST(timestamp * 1000 AS BIGINT) AS timestamp,
                ticker,
                spot AS spot_price,
                zero_gamma,
                sum_gex_vol AS net_gex,
                min_dte,
                sec_min_dte,
                major_pos_vol,
                major_pos_oi,
                major_neg_vol,
                major_neg_oi,
                sum_gex_vol,
                sum_gex_oi,
                delta_risk_reversal,
                CAST(max_priors AS VARCHAR) AS max_priors,
                CAST(strikes AS VARCHAR) AS strikes
            FROM {source}
            """

            strike_insert_template = """
            INSERT INTO gex_strikes
            SELECT
                CAST(timestamp * 1000 AS BIGINT) AS timestamp,
                ticker,
                CAST(strike_array[1] AS DOUBLE) AS strike,
                CAST(strike_array[2] AS DOUBLE) AS gamma,
                CAST(strike_array[3] AS DOUBLE) AS oi_gamma,
                CAST(strike_array[4] AS VARCHAR) AS priors
            FROM {source}
            , UNNEST(strikes) AS strike_row(strike_array)
            """

            snapshot_total = 0
            strike_total = 0

            for file_path in json_files:
                source = self._build_json_relation(file_path)

                snap_cnt, strike_cnt = conn.execute(
                    f"SELECT COUNT(*) AS snapshots, "
                    f"COALESCE(SUM(list_count(strikes)), 0) AS strike_cnt "
                    f"FROM {source}"
                ).fetchone()

                conn.execute(snapshot_insert_template.format(source=source))
                conn.execute(strike_insert_template.format(source=source))

                snapshot_total += snap_cnt
                strike_total += strike_cnt

            conn.execute("COMMIT")

        snapshot_count = self.db_utils.execute_query(
            "SELECT COUNT(*) AS cnt FROM gex_snapshots"
        )[0]["cnt"]
        strike_count = self.db_utils.execute_query(
            "SELECT COUNT(*) AS cnt FROM gex_strikes"
        )[0]["cnt"]

        self.lineage_tracker.record_import(
            "gex_duckdb_bulk_import", snapshot_count, "gex"
        )

        self.stats["gex_records_imported"] = snapshot_count
        LOG.info(
            f"GEX import complete via DuckDB: {snapshot_count} snapshots, "
            f"{strike_count} strike rows"
        )

    def run_import(self) -> Dict[str, Any]:
        """Run the complete import process."""
        LOG.info(
            f"Starting data import from {self.legacy_source_path} (dry_run={self.dry_run})"
        )

        # Scan data sources
        sources = self.scan_data_sources()
        LOG.info("Data sources scanned:")
        for name, info in sources.items():
            LOG.info(f"  {name}: {info.get('count', 0)} files")

        # Run imports
        self.import_gex_data()
        # Tick, depth, and database imports removed per user request

        # Export strike data to Parquet lake if applicable
        self.export_gex_strikes_to_parquet()

        # Report results
        LOG.info("Import completed!")
        LOG.info(f"Stats: {self.stats}")

        return {
            "sources_scanned": sources,
            "import_stats": self.stats,
            "dry_run": self.dry_run,
            "timestamp": datetime.now().isoformat(),
        }

    def export_gex_strikes_to_parquet(self) -> None:
        """Mirror DuckDB strike data into partitioned Parquet files."""
        if self.dry_run:
            LOG.info("Dry-run mode: skipping Parquet export")
            return

        if not self.db_utils:
            LOG.warning("DuckDB utilities unavailable; skipping Parquet export")
            return

        with self.db_utils:
            if not self.db_utils.table_exists("gex_strikes"):
                LOG.info("No gex_strikes table present; skipping Parquet export")
                return

        if self.parquet_output_dir.exists():
            shutil.rmtree(self.parquet_output_dir)
        self.parquet_output_dir.mkdir(parents=True, exist_ok=True)

        export_query = """
            SELECT
                timestamp,
                ticker,
                strike,
                gamma,
                oi_gamma,
                priors,
                year(timestamp) AS year,
                month(timestamp) AS month
            FROM gex_strikes
        """

        with self.db_utils:
            self.db_utils.export_query_to_parquet(
                export_query,
                str(self.parquet_output_dir),
                partition_by=["year", "month"],
            )

            count_result = self.db_utils.execute_query(
                "SELECT COUNT(*) AS cnt FROM gex_strikes"
            )

        total_rows = count_result[0]["cnt"] if count_result else 0
        self.stats["parquet_partitions_written"] = total_rows
        LOG.info(
            f"Exported {total_rows} strike rows to Parquet under {self.parquet_output_dir} "
            "partitioned by year/month"
        )

    def _build_json_relation(self, file_path: Path) -> str:
        """Return a DuckDB read_json_auto expression for the given file."""
        escaped = str(file_path.resolve()).replace("'", "''")
        return f"read_json_auto('{escaped}', maximum_depth=4)"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Import data from legacy exports")
    parser.add_argument(
        "--legacy-path",
        default="data/legacy_source",
        help="Path to legacy export directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview import without actually storing data",
    )
    parser.add_argument(
        "--perform-import", action="store_true", help="Perform actual data import"
    )

    args = parser.parse_args()

    if not args.dry_run and not args.perform_import:
        print("Please specify --dry-run or --import")
        return

    # Validate legacy path
    legacy_path = Path(args.legacy_path)
    if not legacy_path.exists():
        print(f"Error: legacy export path not found: {legacy_path}")
        return

    # Run import
    importer = DataImporter(legacy_path, dry_run=args.dry_run)
    results = importer.run_import()

    # Save results
    output_file = f"import_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    print(
        f"Total records processed: {sum(v for k, v in results['import_stats'].items() if k != 'errors') - len(results['import_stats']['errors'])}"
    )


if __name__ == "__main__":
    main()
