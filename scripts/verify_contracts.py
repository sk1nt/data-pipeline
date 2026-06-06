#!/usr/bin/env python3
"""Validate local Redis contracts (no sibling repo required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts import RedisChannels, SweepAlertPayload  # noqa: E402

REQUIRED_CHANNELS = (
    "gex:snapshot:stream",
    "market:dom:{symbol}",
    "market:cvd:{symbol}",
    "sweep:alert:{symbol}",
    "sweep:monitor:{symbol}",
    "sweep:danger:{symbol}",
    "sweep:ack:{symbol}",
)


def main() -> int:
    version_file = ROOT / "contracts" / "CONTRACT_VERSION"
    if not version_file.exists():
        print("FAIL: missing contracts/CONTRACT_VERSION")
        return 1
    version = version_file.read_text().strip()
    print(f"contract version: {version}")

    ch = RedisChannels()
    checks = [
        ch.gex_snapshot_stream,
        ch.market_dom,
        ch.market_cvd,
        ch.sweep_alert,
        ch.sweep_monitor,
        ch.sweep_danger,
        ch.sweep_ack,
    ]
    if checks != list(REQUIRED_CHANNELS):
        print("FAIL: channel template mismatch")
        print(" got:", checks)
        print("want:", list(REQUIRED_CHANNELS))
        return 1

    assert ch.dom("MNQ") == "market:dom:MNQ"
    assert ch.alert("MNQ") == "sweep:alert:MNQ"

    sample = SweepAlertPayload(
        ts_ms=1,
        symbol="MNQ",
        direction="up",
        trigger_price=20000.0,
        trigger_ticks=10.0,
        classification="sweep",
        confidence=0.7,
        danger_level=0,
        model_version="rules_v1",
    )
    assert sample.classification == "sweep"

    print("verify_contracts: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())