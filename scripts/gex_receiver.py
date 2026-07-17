#!/usr/bin/env python3
"""GEX File Receiver — minimal FastAPI server on port 8877.

Receives webhook POSTs from GEXBot containing Azure Blob SAS URLs,
downloads the .json.gz, and saves to data/source/gexbot/{ticker}/{endpoint}/.

That's it. No DB, no import, no normalization. The JSON files are the
source of truth. build_enriched_1s.py reads them directly at build time.

Layout
------
  data/source/gexbot/NDX/gex_zero/       — raw NDX JSON downloads
  data/source/gexbot/NQ_NDX/gex_zero/    — NQ_NDX JSON downloads
  data/source/gexbot/{ticker}/gex_full/  — gex_full endpoint downloads

Usage
-----
  # Start server (foreground)
  python scripts/gex_receiver.py

  # Manually download a URL
  python scripts/gex_receiver.py --enqueue "https://nfagexbotresearch.blob.core.windows.net/..."

  # Show downloaded files
  python scripts/gex_receiver.py --status

Environment
-----------
  GEX_RECEIVER_PORT   Port to listen on (default: 8877)
  GEX_RECEIVER_HOST   Host to bind (default: 0.0.0.0)

Webhook contract (what GEXBot POSTs via monkey script)
-------------------------------------------------------
  POST /gex_history_url
  Content-Type: text/plain  (sendBeacon) or application/json (fetch fallback)

  {
    "url": "https://nfagexbotresearch.blob.core.windows.net/.../2026-06-16_NQ_NDX_classic_gex_zero.json.gz?sv=...&sig=...",
    "ticker": "NQ_NDX",        # optional — inferred from URL if absent
    "endpoint": "gex_zero"     # optional — inferred from URL if absent
  }

  Response: { "status": "downloaded", "ticker": "NQ_NDX", "endpoint": "gex_zero", "path": "..." }
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("gex_receiver")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PORT = int(os.getenv("GEX_RECEIVER_PORT", "8877"))
HOST = os.getenv("GEX_RECEIVER_HOST", "0.0.0.0")
SOURCE_DIR = ROOT / "data/source/gexbot"

# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

# Known GEXBot blob storage domains
_GEX_URL_RE = re.compile(
    r"(hist\.gex\.bot|nfagexbotresearch\.blob\.core\.windows\.net|app\.gexbot\.com/(hist|chart)/)",
    re.IGNORECASE,
)


def _is_gex_url(url: str) -> bool:
    return bool(_GEX_URL_RE.search(url))


def _infer_ticker(url: str) -> str:
    # New pattern: app.gexbot.com/hist/NDX/classic/gex_zero
    m_new = re.search(r"/(?:hist|chart)/([A-Z]+)/classic/", url)
    if m_new:
        raw = m_new.group(1).upper()
        return f"NQ_{raw}" if raw == "NDX" else raw
    # Old pattern: /2026-06-16_NQ_NDX_classic_gex_zero.json.gz
    m = re.search(r"/(\d{4}-\d{2}-\d{2})_([A-Z0-9_]+)_classic", url)
    if m:
        return m.group(2).upper()
    m2 = re.search(r"/([A-Z0-9_]{2,12})_classic", url)
    if m2:
        return m2.group(1).upper()
    return "NQ_NDX"


def _infer_endpoint(url: str) -> str:
    # New pattern: /classic/gex_zero
    m_new = re.search(r"/classic/(gex_\w+)", url)
    if m_new:
        return m_new.group(1).lower()
    # Old pattern: _gex_zero.json
    m = re.search(r"_(gex_zero|gex_one|gex_full)\.json", url)
    return m.group(1) if m else "gex_zero"


def _trade_day_from_url(url: str) -> str | None:
    m = re.search(r"/(\d{4}-\d{2}-\d{2})_", url)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download(url: str, ticker: str, endpoint: str) -> Path:
    """Download URL, save as .json.gz in the source directory.

    Azure serves .json.gz URLs as plain JSON (Content-Type: application/json),
    but we re-compress on disk to save space and ensure uniform format.
    Returns path to the saved file.
    """
    day = _trade_day_from_url(url) or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dest_dir = SOURCE_DIR / ticker / endpoint
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_name = Path(url.split("?")[0]).name or f"{day}_{ticker}_{endpoint}.json"
    gz_name = raw_name.removesuffix(".gz") + ".gz"  # always store as .json.gz
    local = dest_dir / gz_name
    if local.exists():
        LOGGER.info("  Cache hit: %s", local)
        return local
    LOGGER.info("Downloading %s → %s", url[:80], local)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.content
    # Decompress if actually gzipped (magic bytes 1f 8b), then re-compress uniformly
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    compressed = gzip.compress(data, compresslevel=6)
    local.write_bytes(compressed)
    LOGGER.info("  Saved %d bytes → %d bytes gzipped (%.1f%%)",
                len(data), len(compressed), 100 * len(compressed) / len(data))
    return local


# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------

def make_app():
    app = FastAPI(title="GEX File Receiver", version="2.0")

    # Allow gexbot.com to POST from the browser (sendBeacon / fetch)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://gexbot.com", "https://www.gexbot.com"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/gex_history_url")
    async def receive_gex_history(request: Request):
        """Accept GEXBot webhook with hist.gex.bot SAS URL.

        Downloads the file immediately and returns the local path.
        Handles both application/json and text/plain (sendBeacon sends text/plain).
        """
        raw = await request.body()
        try:
            body = json.loads(raw)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object")

        url = (body.get("url") or "").strip()
        if not url:
            raise HTTPException(status_code=422, detail="Missing url field")
        if not _is_gex_url(url):
            raise HTTPException(status_code=422, detail="URL must be from hist.gex.bot or nfagexbotresearch.blob.core.windows.net")

        ticker = None
        for key in ("ticker", "symbol", "underlying"):
            v = body.get(key)
            if isinstance(v, str) and v.strip() and not v.strip().lower().startswith("gex_"):
                ticker = v.strip().upper()
                break
        ticker = _infer_ticker(url) if not ticker else ticker

        endpoint = None
        for v in body.values():
            if isinstance(v, str) and v.strip().lower().startswith("gex_"):
                endpoint = v.strip().lower()
                break
        endpoint = endpoint or _infer_endpoint(url) or "gex_zero"

        # Handle raw_body (inline JSON payload instead of URL)
        raw_body_data = body.get("raw_body", "")
        if raw_body_data:
            day = _trade_day_from_url(url) or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            dest_dir = SOURCE_DIR / ticker / endpoint
            dest_dir.mkdir(parents=True, exist_ok=True)
            local = dest_dir / f"{day}_{ticker}_{endpoint}.json.gz"
            data = raw_body_data.encode("utf-8")
            compressed = gzip.compress(data, compresslevel=6)
            local.write_bytes(compressed)
            LOGGER.info("Saved raw body: %s (%d bytes gzipped)", local, len(compressed))
        else:
            try:
                local = download(url, ticker, endpoint)
            except Exception as e:
                LOGGER.error("Download failed: %s", e)
                raise HTTPException(status_code=502, detail=f"Download failed: {e}")

        LOGGER.info("Received webhook: ticker=%s endpoint=%s path=%s", ticker, endpoint, local)
        return {"status": "downloaded", "ticker": ticker, "endpoint": endpoint, "path": str(local)}

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def show_status() -> None:
    """Show downloaded GEX files by ticker/endpoint."""
    print("\n=== GEX File Receiver Status ===")
    print(f"  Source dir: {SOURCE_DIR}\n")
    if not SOURCE_DIR.exists():
        print("  (no files downloaded yet)")
        return
    for ticker_dir in sorted(SOURCE_DIR.iterdir()):
        if not ticker_dir.is_dir():
            continue
        print(f"  {ticker_dir.name}/")
        for endpoint_dir in sorted(ticker_dir.iterdir()):
            if not endpoint_dir.is_dir():
                continue
            files = sorted(endpoint_dir.glob("*.json.gz"))
            total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
            print(f"    {endpoint_dir.name}/  {len(files)} files, {total_mb:.1f} MB")
            for f in files[-3:]:  # show last 3
                print(f"      {f.name}")
            if len(files) > 3:
                print(f"      ... and {len(files) - 3} more")


def main() -> None:
    p = argparse.ArgumentParser(description="GEX file receiver — downloads JSON from gexbot webhooks")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--host", default=HOST)
    p.add_argument("--enqueue", metavar="URL", help="Manually download a gexbot URL")
    p.add_argument("--status", action="store_true", help="Show downloaded files")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.status:
        show_status()
        return

    if args.enqueue:
        url = args.enqueue
        ticker = _infer_ticker(url)
        endpoint = _infer_endpoint(url)
        local = download(url, ticker, endpoint)
        print(f"Downloaded: {ticker} / {endpoint} → {local}")
        return

    # Start server
    import uvicorn
    app = make_app()
    LOGGER.info("Starting GEX file receiver on %s:%d", args.host, args.port)
    LOGGER.info("Webhook endpoint: POST http://%s:%d/gex_history_url", args.host, args.port)
    LOGGER.info("Source dir: %s", SOURCE_DIR)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
