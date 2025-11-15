from __future__ import annotations

import time
import json
from pathlib import Path
import duckdb

from src.import_gex_history_safe import safe_import


def make_sample(path: Path, n=1000):
    sample = []
    base_ts = 1761050000
    for i in range(n):
        sample.append({
            "timestamp": base_ts + i * 60,
            "ticker": "NQ",
            "spot": 16000.0 + i * 0.1,
            "zero_gamma": 0.1,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample))


def run_bench(tmp_path: Path, records=1000):
    src = tmp_path / "bench_sample.json"
    make_sample(src, n=records)
    db = tmp_path / "bench_gex.db"
    duckdb.connect(str(db)).close()

    t0 = time.time()
    res = safe_import(src, duckdb_path=db, publish=True)
    t1 = time.time()
    elapsed = t1 - t0
    rps = res.get("records", 0) / max(elapsed, 1e-6)
    print(f"Imported {res.get('records',0)} records in {elapsed:.3f}s -> {rps:.1f} r/s")
    return rps


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--records", type=int, default=1000)
    args = p.parse_args()
    rps = run_bench(Path("/tmp/bench_import"), records=args.records)
    THRESH = 500
    if rps < THRESH:
        print(f"FAIL: throughput {rps:.1f} < {THRESH}")
        raise SystemExit(2)
    print("OK")
