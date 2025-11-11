import json
import sys
from pathlib import Path

import duckdb

# ensure project src/ is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.import_gex_history_safe import safe_import
from src.import_job_store import ImportJobStore


def make_sample_json(path: Path):
    sample = [
        {
            "timestamp": 1761053415,
            "ticker": "NQ",
            "spot": 16000.5,
            "zero_gamma": 0.12,
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample))


def test_idempotent_import(tmp_path):
    db_path = tmp_path / "gex_data.db"
    history_db = tmp_path / "gex_data.db"
    # create empty duckdb databases
    duckdb.connect(str(db_path)).close()
    duckdb.connect(str(history_db)).close()

    src_file = tmp_path / "sample_history.json"
    make_sample_json(src_file)

    # first import
    res1 = safe_import(src_file, duckdb_path=db_path, publish=True, history_db_path=history_db)
    assert res1.get("records") == 1

    # second import (same file) should be skipped due to checksum
    # point safe_import to use the test history DB by monkeypatching default
    job_store = ImportJobStore(db_path=history_db)
    # compute checksum and register as completed to simulate earlier run
    ch = job_store.compute_checksum(src_file)
    jid = job_store.create_job(None, ch, None)
    job_store.mark_started(jid)
    job_store.mark_completed(jid, 1)

    res2 = safe_import(src_file, duckdb_path=db_path, publish=True, history_db_path=history_db)
    assert res2.get("skipped") is True

    con = duckdb.connect(str(db_path))
    try:
        count = con.execute("SELECT COUNT(*) FROM strikes").fetchone()[0]
        assert count == 1
    finally:
        con.close()
