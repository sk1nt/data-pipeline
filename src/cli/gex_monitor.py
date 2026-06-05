"""
CLI for monitoring the GEXBot poller — live snapshots, status, and watch mode.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import click
import redis as redis_lib


SNAPSHOT_KEY_PREFIX = "gex:snapshot:"
SNAPSHOT_PUBSUB_CHANNEL = "gex:snapshot:stream"
DEFAULT_HOST = os.getenv("REDIS_HOST", "localhost")
DEFAULT_PORT = int(os.getenv("REDIS_PORT", "6379"))
DEFAULT_DB = int(os.getenv("REDIS_DB", "0"))
API_BASE = os.getenv("DATA_PIPELINE_URL", "http://localhost:8877")


def _redis() -> redis_lib.Redis:
    return redis_lib.Redis(host=DEFAULT_HOST, port=DEFAULT_PORT, db=DEFAULT_DB, decode_responses=True)


def _age_str(ts: str | None) -> str:
    if not ts:
        return "unknown"
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        return f"{age:.1f}s ago"
    except Exception:
        return ts


def _fmt_snapshot(snap: dict, symbol: str) -> str:
    spot = snap.get("spot")
    zero = snap.get("zero_gamma")
    net = snap.get("net_gex") or snap.get("sum_gex_vol")
    ts = snap.get("timestamp")
    age = _age_str(ts)

    spot_str = f"${spot:,.2f}" if isinstance(spot, (int, float)) else str(spot)
    zero_str = f"{zero:,.2f}" if isinstance(zero, (int, float)) else str(zero)
    net_str = f"{net:,.0f}" if isinstance(net, (int, float)) else str(net)

    return (
        f"{symbol:<10}  spot={spot_str:<12}  zero={zero_str:<12}  "
        f"net_gex={net_str:<14}  [{age}]"
    )


@click.group()
def gex():
    """GEXBot poller monitor."""
    pass


@gex.command()
@click.argument("symbol", default="NQ_NDX")
def snapshot(symbol: str):
    """Print the latest cached snapshot for SYMBOL."""
    r = _redis()
    raw = r.get(f"{SNAPSHOT_KEY_PREFIX}{symbol.upper()}")
    if not raw:
        click.echo(f"No snapshot found for {symbol.upper()}", err=True)
        sys.exit(1)
    snap = json.loads(raw)
    click.echo(_fmt_snapshot(snap, symbol.upper()))


@gex.command()
@click.argument("symbols", nargs=-1)
def status(symbols: tuple[str, ...]):
    """Show latest snapshot age for all cached symbols (or specific SYMBOLS)."""
    r = _redis()
    keys = r.keys(f"{SNAPSHOT_KEY_PREFIX}*")
    if not keys:
        click.echo("No GEX snapshots in Redis.")
        return

    target = {s.upper() for s in symbols} if symbols else None

    rows = []
    for key in sorted(keys):
        sym = key.removeprefix(SNAPSHOT_KEY_PREFIX)
        if target and sym not in target:
            continue
        raw = r.get(key)
        if not raw:
            continue
        snap = json.loads(raw)
        rows.append(_fmt_snapshot(snap, sym))

    click.echo("\n".join(rows) if rows else "No matching symbols.")


@gex.command()
@click.argument("symbol", default="NQ_NDX")
@click.option("--interval", "-i", default=1.0, show_default=True,
              help="Refresh interval in seconds (poll mode only).")
@click.option("--pubsub", "use_pubsub", is_flag=True, default=True,
              help="Use Redis pubsub for real-time updates (default).")
@click.option("--poll", "use_pubsub", flag_value=False,
              help="Poll Redis key instead of pubsub.")
def watch(symbol: str, interval: float, use_pubsub: bool):
    """Live-watch snapshots for SYMBOL. Ctrl-C to stop."""
    sym = symbol.upper()
    click.echo(f"Watching {sym} via {'pubsub' if use_pubsub else f'poll every {interval}s'} — Ctrl-C to stop\n")

    if use_pubsub:
        r = _redis()
        pubsub = r.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(SNAPSHOT_PUBSUB_CHANNEL)
        try:
            for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                data = message.get("data")
                if not data:
                    continue
                try:
                    snap = json.loads(data)
                except Exception:
                    continue
                if (snap.get("symbol") or "").upper() != sym:
                    continue
                ts_now = datetime.now().strftime("%H:%M:%S")
                click.echo(f"[{ts_now}]  {_fmt_snapshot(snap, sym)}")
        except KeyboardInterrupt:
            pubsub.close()
    else:
        r = _redis()
        key = f"{SNAPSHOT_KEY_PREFIX}{sym}"
        last_ts = None
        try:
            while True:
                raw = r.get(key)
                if raw:
                    snap = json.loads(raw)
                    ts = snap.get("timestamp")
                    if ts != last_ts:
                        last_ts = ts
                        ts_now = datetime.now().strftime("%H:%M:%S")
                        click.echo(f"[{ts_now}]  {_fmt_snapshot(snap, sym)}")
                time.sleep(interval)
        except KeyboardInterrupt:
            pass


@gex.command()
@click.argument("symbol", default="NQ_NDX")
def raw(symbol: str):
    """Dump the raw JSON snapshot for SYMBOL."""
    r = _redis()
    data = r.get(f"{SNAPSHOT_KEY_PREFIX}{symbol.upper()}")
    if not data:
        click.echo(f"No snapshot for {symbol.upper()}", err=True)
        sys.exit(1)
    click.echo(json.dumps(json.loads(data), indent=2))


if __name__ == "__main__":
    gex()
