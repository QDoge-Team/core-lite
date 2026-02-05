from __future__ import annotations

import asyncio
import logging
import time

from app.config import LocalSnapshotConfig
from app.node_client import NodeClient

logger = logging.getLogger(__name__)


class LocalSnapshotSaver:
    """Periodically trigger local snapshot saves (equivalent to pressing F8).

    This runs in normal mode to ensure the node's state is regularly saved
    to disk, providing recovery points in case of crashes or restarts.
    """

    def __init__(
        self,
        config: LocalSnapshotConfig,
        node_client: NodeClient,
    ) -> None:
        self._config = config
        self._node_client = node_client
        self._last_save_time: float = 0

    async def run(self, shutdown_event: asyncio.Event) -> None:
        """Main loop: periodically request snapshot saves."""
        logger.info(
            f"Local snapshot saver started "
            f"(interval={self._config.interval_seconds}s)"
        )

        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=self._config.interval_seconds,
                )
                # Shutdown requested
                return
            except asyncio.TimeoutError:
                # Interval elapsed, trigger save
                pass

            await self._trigger_save()

    async def _trigger_save(self) -> None:
        """Request the node to save a snapshot."""
        try:
            # Check if node is already saving
            tick_info = await self._node_client.get_tick_info()
            if tick_info.is_saving_snapshot:
                logger.debug("Node is already saving a snapshot, skipping")
                return

            logger.info(
                f"Triggering local snapshot save "
                f"(epoch={tick_info.epoch}, tick={tick_info.tick})"
            )

            success = await self._node_client.request_save_snapshot()
            if success:
                self._last_save_time = time.monotonic()
                logger.info("Local snapshot save requested successfully")
            else:
                logger.warning("Local snapshot save request returned non-ok status")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to trigger local snapshot save: {e}")

    @property
    def last_save_time(self) -> float:
        return self._last_save_time
