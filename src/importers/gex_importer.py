import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from src.models.gex_snapshot import GEXSnapshot
from src.db.duckdb_utils import DuckDBUtils
from src.lineage.lineage_tracker import LineageTracker


class GEXImporter:
    def __init__(self, db_utils: DuckDBUtils, lineage_tracker: LineageTracker):
        self.db_utils = db_utils
        self.lineage_tracker = lineage_tracker
        self.table_name = "gex_snapshots"

    def create_table(self):
        schema = """
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
        strike_data VARCHAR
        """
        self.db_utils.create_table_if_not_exists(self.table_name, schema)

    def import_file(self, file_path: Path, dry_run: bool = False) -> Dict[str, Any]:
        self.create_table()  # Ensure table exists
        try:
            with open(file_path, "r") as f:
                raw_data = json.load(f)

            # Handle the gex_bridge/history JSON format
            if isinstance(raw_data, list):
                # Multiple snapshots in array
                snapshots_data = raw_data
            else:
                # Single snapshot
                snapshots_data = [raw_data]

            valid_snapshots = []
            errors = []

            for snapshot_data in snapshots_data:
                try:
                    # Convert gex_bridge format to GEXSnapshot format
                    converted_data = self._convert_gex_bridge_format(snapshot_data)
                    snapshot = GEXSnapshot(**converted_data)

                    validation_errors = snapshot.validate_strike_data()
                    if validation_errors:
                        errors.extend(validation_errors)
                        continue

                    valid_snapshots.append(snapshot)
                except Exception as e:
                    errors.append(f"Error converting snapshot: {str(e)}")

            if not dry_run and valid_snapshots:
                data_dicts = []
                for snapshot in valid_snapshots:
                    d = snapshot.to_dict()
                    epoch_ms = int(snapshot.timestamp.timestamp() * 1000)
                    d["timestamp"] = epoch_ms
                    d["strike_data"] = json.dumps(d["strike_data"])
                    data_dicts.append(d)

                with self.db_utils:
                    self.db_utils.insert_data(self.table_name, data_dicts)

                self.lineage_tracker.record_import(
                    str(file_path), len(valid_snapshots), "gex"
                )

            return {
                "file": str(file_path),
                "total_snapshots": len(snapshots_data),
                "valid_snapshots": len(valid_snapshots),
                "errors": errors,
                "dry_run": dry_run,
            }

        except Exception as e:
            return {"file": str(file_path), "error": str(e), "dry_run": dry_run}

    def _convert_gex_bridge_format(
        self, snapshot_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert gex_bridge/history JSON format to GEXSnapshot format."""
        # Convert timestamp to datetime
        timestamp_val = snapshot_data["timestamp"]
        if isinstance(timestamp_val, str):
            # ISO format string
            timestamp = datetime.fromisoformat(timestamp_val.replace("Z", "+00:00"))
        else:
            # Unix timestamp
            timestamp = datetime.fromtimestamp(timestamp_val)

        # Convert strikes array to strike_data format
        strike_data = []
        for strike_item in snapshot_data["strikes"]:
            if len(strike_item) >= 4:
                strike_price, vol_gex, oi_gex, priors = (
                    strike_item[0],
                    strike_item[1],
                    strike_item[2],
                    strike_item[3],
                )

                # Create strike dict with all available fields
                strike_dict = {
                    "strike": strike_price,
                    "gamma": vol_gex,  # Use vol_gex as the primary gamma value
                    "oi_gamma": oi_gex,  # Store OI GEX separately
                    "priors": priors if isinstance(priors, list) else [],
                }
                strike_data.append(strike_dict)

        # Use sum_gex_vol as net_gex, or calculate from strikes if not available
        net_gex = snapshot_data.get("sum_gex_vol", sum(s["gamma"] for s in strike_data))

        return {
            "timestamp": timestamp,
            "ticker": snapshot_data["ticker"],
            "spot_price": snapshot_data["spot_price"],
            "zero_gamma": snapshot_data["zero_gamma"],
            "net_gex": net_gex,
            "min_dte": snapshot_data.get("min_dte"),
            "sec_min_dte": snapshot_data.get("sec_min_dte"),
            "major_pos_vol": snapshot_data.get("major_pos_vol"),
            "major_pos_oi": snapshot_data.get("major_pos_oi"),
            "major_neg_vol": snapshot_data.get("major_neg_vol"),
            "major_neg_oi": snapshot_data.get("major_neg_oi"),
            "sum_gex_vol": snapshot_data.get("sum_gex_vol"),
            "sum_gex_oi": snapshot_data.get("sum_gex_oi"),
            "delta_risk_reversal": snapshot_data.get("delta_risk_reversal"),
            "max_priors": json.dumps(snapshot_data.get("max_priors", [])),
            "strike_data": strike_data,
        }

    def import_snapshot(
        self, snapshot: GEXSnapshot, dry_run: bool = False
    ) -> Dict[str, Any]:
        """Import a single GEXSnapshot object directly."""
        try:
            # Validate the snapshot
            validation_errors = snapshot.validate_strike_data()
            if validation_errors:
                return {
                    "success": False,
                    "errors": validation_errors,
                    "dry_run": dry_run,
                }

            if not dry_run:
                # Convert to dict and serialize strike_data
                data_dict = snapshot.to_dict()
                epoch_ms = int(snapshot.timestamp.timestamp() * 1000)
                data_dict["timestamp"] = epoch_ms
                data_dict["strike_data"] = json.dumps(data_dict["strike_data"])

                with self.db_utils:
                    self.db_utils.insert_data(self.table_name, [data_dict])

                # Record lineage (using a synthetic file path for database imports)
                self.lineage_tracker.record_import(
                    f"database_import_{snapshot.timestamp.isoformat()}", 1, "gex"
                )

            return {
                "success": True,
                "timestamp": snapshot.timestamp.isoformat(),
                "ticker": snapshot.ticker,
                "strike_count": len(snapshot.strike_data),
                "dry_run": dry_run,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "dry_run": dry_run}

    def import_files(
        self, file_paths: List[Path], dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """Import multiple GEX files."""
        self.create_table()
        results = []
        for file_path in file_paths:
            result = self.import_file(file_path, dry_run)
            results.append(result)
        return results
