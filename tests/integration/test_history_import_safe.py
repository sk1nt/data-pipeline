import json
from pathlib import Path

import duckdb

from src.import_gex_history_safe import safe_import


def make_sample_json(path: Path):
    sample = [
        {
            "timestamp": 1761053415,
            "ticker": "NQ",
            "spot": 16000.5,
            "zero_gamma": 0.12,
        },
        {
            "timestamp": 1761057015,
            "ticker": "NQ",
            "spot": 16010.0,
            "zero_gamma": 0.11,
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample))


def test_safe_import_stages_and_inserts(tmp_path, monkeypatch):
    db_path = tmp_path / "gex_data.db"
    # create empty duckdb
    duckdb.connect(str(db_path)).close()

    src_file = tmp_path / "sample_history.json"
    make_sample_json(src_file)

    result = safe_import(src_file, duckdb_path=db_path, publish=True)
    assert "job_id" in result
    assert result["records"] == 2

    con = duckdb.connect(str(db_path))
    try:
        count = con.execute("SELECT COUNT(*) FROM strikes").fetchone()[0]
        assert count == 2
    finally:
        con.close()
