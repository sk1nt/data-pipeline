"""Manage the Discord bot lifecyle as a subprocess."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)


class DiscordBotService:
    def __init__(self, script_path: Path) -> None:
        self.script_path = script_path
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.last_exit_code: Optional[int] = None
        self.last_start: Optional[str] = None
        self.last_stop: Optional[str] = None
        self.restart_count: int = 0
        self.last_restart: Optional[str] = None
        self._log_handle: Optional[Any] = None

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            LOGGER.info("Discord bot already running (pid=%s)", self.process.pid)
            return
        env = os.environ.copy()
        LOGGER.info("Starting Discord bot via %s", self.script_path)
        log_path = self.script_path.parent / "bot.log"
        self._log_handle = open(log_path, "a", buffering=1)
        self.process = subprocess.Popen(
            [sys.executable, str(self.script_path)],
            cwd=str(self.script_path.parent),
            env=env,
            stdout=self._log_handle,
            stderr=self._log_handle,
        )
        self.last_start = datetime.utcnow().isoformat()

    async def stop(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            LOGGER.info("Stopping Discord bot (pid=%s)", self.process.pid)
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, 15)
            except subprocess.TimeoutExpired:
                LOGGER.warning("Discord bot did not exit gracefully; killing")
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
        self.last_exit_code = self.process.returncode
        self.process = None
        self.last_stop = datetime.utcnow().isoformat()
        if self._log_handle:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None

    async def restart(self) -> None:
        await self.stop()
        self.restart_count += 1
        self.last_restart = datetime.utcnow().isoformat()
        self.start()

    def status(self) -> Dict[str, Any]:
        running = self.process is not None and self.process.poll() is None
        pid = self.process.pid if running else None
        uptime = None
        if running and self.last_start:
            try:
                start_time = datetime.fromisoformat(self.last_start)
                uptime = (datetime.utcnow() - start_time).total_seconds()
            except Exception:
                pass
        return {
            "running": running,
            "pid": pid,
            "last_exit_code": self.last_exit_code,
            "last_start": self.last_start,
            "last_stop": self.last_stop,
            "restart_count": self.restart_count,
            "last_restart": self.last_restart,
            "uptime_seconds": uptime,
        }
