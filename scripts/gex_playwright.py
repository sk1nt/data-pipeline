#!/usr/bin/env python3
"""Persisted Chromium session for the daily GEXBot browser action.

The first run is interactive and stores cookies only in the ignored runtime
profile directory. Scheduled runs reuse that profile and never accept
credentials on the command line.

Configuration (environment variables):

    GEX_PLAYWRIGHT_URL             Page to open for the action.
    GEX_PLAYWRIGHT_LOGIN_URL      Optional login page used by ``login``.
    GEX_PLAYWRIGHT_BUTTON         CSS/text locator for the action button.
    GEX_PLAYWRIGHT_SUCCESS        Optional locator proving the action completed.
    GEX_PLAYWRIGHT_PROFILE        Optional Chromium profile directory.
    GEX_PLAYWRIGHT_WAIT_SECONDS   Post-click wait (default: 5).

Usage:

    venv/bin/python scripts/gex_playwright.py login
    venv/bin/python scripts/gex_playwright.py interactive
    venv/bin/python scripts/gex_playwright.py run

The login command intentionally leaves credential entry and MFA in the
browser. No username, password, or cookie is written to repository files.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import stat
from pathlib import Path

from playwright.async_api import BrowserContext, Page, async_playwright
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "data" / "runtime" / "playwright" / "gexbot"
LOGGER = logging.getLogger("gex_playwright")
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _profile_path() -> Path:
    profile = Path(_env("GEX_PLAYWRIGHT_PROFILE", str(DEFAULT_PROFILE))).expanduser()
    profile.mkdir(parents=True, exist_ok=True)
    profile.chmod(stat.S_IRWXU)
    return profile


async def _context(playwright: object, *, headless: bool) -> BrowserContext:
    chromium = getattr(playwright, "chromium")
    return await chromium.launch_persistent_context(
        user_data_dir=str(_profile_path()),
        headless=headless,
        accept_downloads=True,
    )


async def _page(context: BrowserContext, url: str) -> Page:
    page = context.pages[0] if context.pages else await context.new_page()
    await page.goto(url, wait_until="domcontentloaded")
    return page


async def login() -> None:
    url = _env("GEX_PLAYWRIGHT_LOGIN_URL") or _env("GEX_PLAYWRIGHT_URL")
    if not url:
        raise SystemExit("Set GEX_PLAYWRIGHT_LOGIN_URL or GEX_PLAYWRIGHT_URL first")
    async with async_playwright() as playwright:
        context = await _context(playwright, headless=False)
        try:
            page = await _page(context, url)
            LOGGER.info("Complete login/MFA in Chromium, then press Enter here")
            await asyncio.to_thread(input)
            LOGGER.info("Session retained in %s", _profile_path())
            LOGGER.info("Current page: %s", page.url)
        finally:
            await context.close()


async def run_action() -> None:
    url = _env("GEX_PLAYWRIGHT_URL")
    button = _env("GEX_PLAYWRIGHT_BUTTON")
    if not url or not button:
        raise SystemExit("Set GEX_PLAYWRIGHT_URL and GEX_PLAYWRIGHT_BUTTON first")
    wait_seconds = float(_env("GEX_PLAYWRIGHT_WAIT_SECONDS", "5"))
    success = _env("GEX_PLAYWRIGHT_SUCCESS")
    async with async_playwright() as playwright:
        context = await _context(playwright, headless=True)
        try:
            page = await _page(context, url)
            await page.locator(button).click(timeout=15_000)
            if success:
                await page.locator(success).wait_for(timeout=30_000)
            if wait_seconds > 0:
                await page.wait_for_timeout(int(wait_seconds * 1000))
            LOGGER.info("GEX browser action completed at %s", page.url)
        finally:
            await context.close()


async def interactive_action() -> None:
    """Open the page and let the operator click the target control manually."""
    url = _env("GEX_PLAYWRIGHT_URL")
    if not url:
        raise SystemExit("Set GEX_PLAYWRIGHT_URL first")
    async with async_playwright() as playwright:
        context = await _context(playwright, headless=False)
        try:
            page = await _page(context, url)
            LOGGER.info("Click the desired button in the Chromium window")
            LOGGER.info("Press Enter here after the action has completed")
            await asyncio.to_thread(input)
            LOGGER.info("Interactive action complete at %s", page.url)
        finally:
            await context.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("login", "interactive", "run"))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "login":
        action = login()
    elif args.command == "interactive":
        action = interactive_action()
    else:
        action = run_action()
    asyncio.run(action)


if __name__ == "__main__":
    main()
