"""Periodic cleanup of old state files, bootstrap archives, and staging dirs.

Keeps only the most recent epoch files (configurable) and removes
temporary artefacts that accumulate over time.  Runs as an asyncio
background task alongside the watchdog and management API.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from app.state_manager import StateManager

logger = logging.getLogger(__name__)

# Default interval between cleanup runs (seconds).
DEFAULT_INTERVAL_SECONDS = 3600
# How many old epochs to keep alongside the current one.
DEFAULT_KEEP_EPOCHS = 0


class Cleanup:
    """Background task that periodically removes stale files."""

    def __init__(
        self,
        state_manager: StateManager,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        keep_epochs: int = DEFAULT_KEEP_EPOCHS,
    ) -> None:
        self._state_manager = state_manager
        self._interval = interval_seconds
        self._keep_epochs = keep_epochs
        self._data_dir = state_manager.data_dir

    async def run(self, shutdown_event: asyncio.Event) -> None:
        """Run the cleanup loop until *shutdown_event* is set."""
        logger.info(
            f"Cleanup task started (interval={self._interval}s, "
            f"keep_epochs={self._keep_epochs})"
        )

        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(), timeout=self._interval
                )
                break  # shutdown requested
            except asyncio.TimeoutError:
                pass  # interval elapsed, run cleanup

            try:
                self._run_cleanup()
            except Exception as e:
                logger.error(f"Cleanup error: {e}", exc_info=True)

    def _run_cleanup(self) -> None:
        """Execute a single cleanup pass."""
        current_epoch = self._state_manager.get_local_epoch()
        if current_epoch is None:
            logger.debug("No local epoch detected, skipping cleanup")
            return

        logger.info(
            f"Running cleanup (epoch={current_epoch}, "
            f"keep={self._keep_epochs})"
        )

        self._cleanup_old_epochs(current_epoch)
        self._cleanup_bootstrap_zips()
        self._cleanup_staging()

    def _cleanup_old_epochs(self, current_epoch: int) -> None:
        """Remove state files and snapshot dirs from old epochs."""
        self._state_manager.cleanup_old_epochs(
            current_epoch, keep=self._keep_epochs
        )

    def _cleanup_bootstrap_zips(self) -> None:
        """Remove downloaded bootstrap / epoch zip archives."""
        patterns = ["ep*.zip", "epoch_files.zip", "snapshot.zip"]
        for pattern in patterns:
            for path in self._data_dir.glob(pattern):
                logger.info(f"Removing bootstrap archive: {path.name}")
                path.unlink()

    def _cleanup_staging(self) -> None:
        """Remove leftover snapshot staging directories."""
        staging = self._data_dir / ".snapshot-staging"
        if staging.is_dir():
            logger.info("Removing leftover staging directory")
            shutil.rmtree(staging, ignore_errors=True)
