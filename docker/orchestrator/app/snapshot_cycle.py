from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.alerting import AlertManager
from app.config import SourceConfig
from app.models import NodeHealth, TickInfo
from app.node_client import NodeClient
from app.state_manager import StateManager
from app.uploaders.base import BaseUploader, RemotePackagingUploader

logger = logging.getLogger(__name__)

# Lock coordination threshold: skip cycle if another source
# uploaded a snapshot within this many ticks of our local tick.
LOCK_TICK_THRESHOLD = 100
_SNAPSHOT_ZIP_RE = re.compile(r"^ep(\d+)-t(\d+)-snap\.zip$")


class SnapshotCycle:
    """Source mode: periodically trigger, package, and upload snapshots.

    Uses lock-based coordination so multiple source nodes don't upload
    duplicate/overlapping snapshots.  Naming convention::

        Archive:  {epoch}/ep{epoch}-t{tick}-snap.zip
        Sidecar:  {epoch}/ep{epoch}-t{tick}-snap.json
        Index:    {epoch}/ep{epoch}-latest-snap.json
        Lock:     {epoch}/snap.lock
    """

    def __init__(
        self,
        config: SourceConfig,
        node_client: NodeClient,
        state_manager: StateManager,
        uploader: BaseUploader,
        alert_manager: AlertManager,
        data_dir: Path,
    ) -> None:
        self._config = config
        self._node_client = node_client
        self._state_manager = state_manager
        self._uploader = uploader
        self._alert_manager = alert_manager
        self._data_dir = data_dir
        self._last_snapshot_epoch: Optional[int] = None
        self._node_id = uuid.uuid4().hex[:12]
        self._trigger_event = asyncio.Event()
        self._cycle_running = False

    # ------------------------------------------------------------------
    # Remote key helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _archive_key(epoch: int, tick: int) -> str:
        return f"{epoch}/ep{epoch}-t{tick}-snap.zip"

    @staticmethod
    def _sidecar_key(epoch: int, tick: int) -> str:
        return f"{epoch}/ep{epoch}-t{tick}-snap.json"

    @staticmethod
    def _index_key(epoch: int) -> str:
        return f"{epoch}/ep{epoch}-latest-snap.json"

    @staticmethod
    def _lock_key(epoch: int) -> str:
        return f"{epoch}/snap.lock"

    # ------------------------------------------------------------------
    # Public trigger interface
    # ------------------------------------------------------------------
    def trigger_immediate(self) -> bool:
        """Request an immediate snapshot cycle.

        Returns True if the trigger was accepted, False if a cycle
        is already running or a trigger is already pending.
        """
        if self._cycle_running or self._trigger_event.is_set():
            return False
        self._trigger_event.set()
        return True

    @property
    def is_cycle_running(self) -> bool:
        return self._cycle_running

    @property
    def is_trigger_pending(self) -> bool:
        return self._trigger_event.is_set()

    @property
    def last_snapshot_epoch(self) -> Optional[int]:
        return self._last_snapshot_epoch

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self, shutdown_event: asyncio.Event) -> None:
        """Main snapshot cycle loop."""
        logger.info(
            f"Snapshot cycle started "
            f"(interval={self._config.snapshot_interval_seconds}s, "
            f"uploader={self._uploader.get_name()}, "
            f"node_id={self._node_id})"
        )

        while not shutdown_event.is_set():
            # Wait for shutdown, manual trigger, or interval timeout
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            trigger_task = asyncio.create_task(self._trigger_event.wait())
            timeout_task = asyncio.create_task(
                asyncio.sleep(self._config.snapshot_interval_seconds)
            )

            try:
                done, pending = await asyncio.wait(
                    {shutdown_task, trigger_task, timeout_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                for t in (shutdown_task, trigger_task, timeout_task):
                    t.cancel()
                raise

            for t in pending:
                t.cancel()

            if shutdown_task in done:
                return

            if trigger_task in done:
                self._trigger_event.clear()
                logger.info("Manual snapshot trigger received")

            try:
                self._cycle_running = True
                await self._execute_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Snapshot cycle failed: {e}", exc_info=True)
                await self._alert_manager.send_alert(
                    "error",
                    "snapshot_cycle_failed",
                    {"error": str(e)},
                )
            finally:
                self._cycle_running = False

    # ------------------------------------------------------------------
    # Single cycle
    # ------------------------------------------------------------------
    async def _execute_cycle(self) -> None:
        """Execute a single snapshot cycle with lock coordination."""

        # Step 1: Pre-check — is the node healthy?
        tick_info = await self._pre_check()
        if tick_info is None:
            return

        epoch = tick_info.epoch
        local_tick = tick_info.tick

        logger.info(
            f"Starting snapshot cycle for epoch {epoch}, tick {local_tick}"
        )

        # Detect epoch transition since last snapshot
        if (
            self._last_snapshot_epoch is not None
            and epoch != self._last_snapshot_epoch
        ):
            logger.info(
                f"Epoch changed from {self._last_snapshot_epoch} to {epoch}, "
                "resetting snapshot state"
            )

        # Step 2: Check remote lock for this epoch
        skip = await self._check_remote_lock(epoch, local_tick)
        if skip:
            return

        # Step 3: Acquire lock
        lock_acquired = await self._acquire_lock(epoch, local_tick)
        if not lock_acquired:
            logger.warning("Failed to write lock file, skipping cycle")
            return

        try:
            # Step 4: Trigger snapshot save
            logger.info("Triggering snapshot save via HTTP API")
            success = await self._node_client.request_save_snapshot()
            if not success:
                logger.warning(
                    "Snapshot save request returned non-ok status"
                )
                return

            # Step 5: Wait for save to start
            started = await self._wait_for_save_start()
            if not started:
                logger.warning(
                    "Snapshot save did not start within timeout, skipping"
                )
                return

            # Step 6: Wait for save to complete
            completed = await self._wait_for_save_complete()
            if not completed:
                logger.error("Snapshot save timed out, skipping upload")
                await self._alert_manager.send_alert(
                    "warning",
                    "snapshot_save_timeout",
                    {"epoch": epoch, "tick": local_tick},
                )
                return

            logger.info("Snapshot save completed")

            # Re-read tick info — the tick may have advanced during save
            try:
                post_save_info = await self._node_client.get_tick_info()
                snap_tick = post_save_info.tick
            except Exception:
                snap_tick = local_tick

            # Step 7: Verify snapshot files
            if not self._state_manager.has_snapshot_directory(epoch):
                logger.error(
                    f"Snapshot directory for epoch {epoch} not found "
                    "after save"
                )
                return

            # Step 8+: Package and upload
            if isinstance(self._uploader, RemotePackagingUploader):
                upload_ok = await self._rsync_upload_path(epoch, snap_tick)
            else:
                upload_ok = await self._local_package_upload_path(
                    epoch, snap_tick
                )

            # Step 9: Clean up old remote snapshots
            if upload_ok:
                await self._cleanup_old_snapshots()

        finally:
            # Always release the lock
            await self._release_lock(epoch)

    # ------------------------------------------------------------------
    # Lock coordination
    # ------------------------------------------------------------------
    async def _check_remote_lock(
        self, epoch: int, local_tick: int
    ) -> bool:
        """Check remote lock and index. Returns True if we should skip."""
        # First check the index to see if a recent-enough snapshot exists
        try:
            index_bytes = await self._uploader.get_small_file(
                self._index_key(epoch)
            )
            if index_bytes is not None:
                index = json.loads(index_bytes)
                remote_tick = index.get("tick", 0)
                if local_tick < remote_tick + LOCK_TICK_THRESHOLD:
                    logger.info(
                        f"Recent snapshot exists at tick {remote_tick} "
                        f"(local={local_tick}, threshold={LOCK_TICK_THRESHOLD}), "
                        "skipping cycle"
                    )
                    return True
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Could not read index: {e}")

        # Check if another source currently holds the lock
        try:
            lock_bytes = await self._uploader.get_small_file(
                self._lock_key(epoch)
            )
            if lock_bytes is not None:
                lock_data = json.loads(lock_bytes)
                lock_tick = lock_data.get("tick", 0)
                lock_ts = lock_data.get("timestamp", "")

                # If the lock is from a tick close to ours, another source
                # is actively building a snapshot — skip.
                if local_tick < lock_tick + LOCK_TICK_THRESHOLD:
                    logger.info(
                        f"Lock held by {lock_data.get('node_id', '?')} "
                        f"at tick {lock_tick}, skipping cycle"
                    )
                    return True

                # Lock is from a much older tick — could be stale.
                # We'll overwrite it.
                logger.info(
                    f"Stale lock at tick {lock_tick} "
                    f"(local={local_tick}), proceeding"
                )
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Could not read lock: {e}")

        return False

    async def _acquire_lock(self, epoch: int, tick: int) -> bool:
        """Write a lock file to remote storage."""
        lock_data = {
            "tick": tick,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_id": self._node_id,
        }
        return await self._uploader.put_small_file(
            self._lock_key(epoch),
            json.dumps(lock_data).encode(),
        )

    async def _release_lock(self, epoch: int) -> None:
        """Delete the lock file from remote storage."""
        try:
            await self._uploader.delete_file(self._lock_key(epoch))
        except Exception as e:
            logger.warning(f"Failed to release lock: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _pre_check(self) -> Optional[TickInfo]:
        """Check if the node is healthy enough for a snapshot."""
        try:
            tick_info = await self._node_client.get_tick_info()
        except Exception as e:
            logger.warning(f"Cannot reach node for pre-check: {e}")
            return None

        if tick_info.is_saving_snapshot:
            logger.info("Node is already saving a snapshot, skipping cycle")
            return None

        return tick_info

    async def _wait_for_save_start(self) -> bool:
        """Wait for isSavingSnapshot to become true."""
        timeout = 60
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(self._config.snapshot_poll_interval_seconds)
            try:
                tick_info = await self._node_client.get_tick_info()
                if tick_info.is_saving_snapshot:
                    return True
            except Exception:
                pass
        return False

    async def _wait_for_save_complete(self) -> bool:
        """Wait for isSavingSnapshot to become false after it was true."""
        deadline = (
            time.monotonic() + self._config.snapshot_wait_timeout_seconds
        )
        while time.monotonic() < deadline:
            await asyncio.sleep(self._config.snapshot_poll_interval_seconds)
            try:
                tick_info = await self._node_client.get_tick_info()
                if not tick_info.is_saving_snapshot:
                    return True
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------
    # Upload paths
    # ------------------------------------------------------------------
    async def _local_package_upload_path(
        self, epoch: int, snap_tick: int
    ) -> bool:
        """Standard path: local ZIP packaging + upload."""
        # Package snapshot with tick-specific name
        # Run in a thread to avoid blocking the event loop during
        # ZIP compression of potentially multi-GB state files.
        staging_dir = self._data_dir / ".snapshot-staging"
        staging_dir.mkdir(exist_ok=True)

        # Marker file to prevent cleanup task from removing staging
        in_progress_marker = staging_dir / ".upload-in-progress"

        try:
            archive_path = await asyncio.to_thread(
                self._state_manager.package_snapshot,
                epoch,
                staging_dir,
                tick=snap_tick,
                compression=self._config.package_compression,
            )
        except Exception as e:
            logger.error(f"Failed to package snapshot: {e}")
            return False

        # Upload archive
        checksum = await asyncio.to_thread(
            StateManager.compute_checksum, archive_path
        )
        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "epoch": epoch,
            "tick": snap_tick,
            "timestamp": now,
            "checksum": checksum,
            "size_bytes": archive_path.stat().st_size,
            "node_id": self._node_id,
        }
        remote_key = self._archive_key(epoch, snap_tick)

        # Mark upload as in-progress to prevent cleanup interference
        try:
            in_progress_marker.touch()
        except OSError:
            pass

        try:
            upload_ok = await self._upload_with_retries(
                archive_path, metadata, remote_key
            )

            if upload_ok:
                await self._publish_metadata(epoch, snap_tick, metadata)
            else:
                await self._alert_manager.send_alert(
                    "error",
                    "snapshot_upload_failed",
                    {"epoch": epoch, "tick": snap_tick},
                )
        finally:
            # Remove in-progress marker
            try:
                in_progress_marker.unlink(missing_ok=True)
            except OSError:
                pass

        # Cleanup staging only after successful upload
        if upload_ok:
            try:
                if archive_path.exists():
                    archive_path.unlink()
            except OSError:
                pass

        return upload_ok

    async def _rsync_upload_path(
        self, epoch: int, snap_tick: int
    ) -> bool:
        """Rsync path: sync snapshot directory + ZIP on remote server."""
        snap_dir = self._state_manager.get_snapshot_directory(epoch)
        if snap_dir is None:
            logger.error(
                f"No snapshot directory found for epoch {epoch}"
            )
            return False

        upload_ok = await self._sync_and_package_with_retries(
            snap_dir, epoch, snap_tick
        )

        if upload_ok:
            # Get checksum and size from remote
            remote_key = self._archive_key(epoch, snap_tick)
            try:
                checksum, size_bytes = (
                    await self._uploader.get_remote_checksum(remote_key)
                )
            except Exception as e:
                logger.error(f"Failed to get remote checksum: {e}")
                checksum = ""
                size_bytes = 0

            now = datetime.now(timezone.utc).isoformat()
            metadata = {
                "epoch": epoch,
                "tick": snap_tick,
                "timestamp": now,
                "checksum": checksum,
                "size_bytes": size_bytes,
                "node_id": self._node_id,
            }
            await self._publish_metadata(epoch, snap_tick, metadata)
        else:
            await self._alert_manager.send_alert(
                "error",
                "snapshot_upload_failed",
                {"epoch": epoch, "tick": snap_tick},
            )

        return upload_ok

    async def _publish_metadata(
        self, epoch: int, snap_tick: int, metadata: dict
    ) -> None:
        """Upload sidecar JSON, update index, and send alert."""
        # Sidecar JSON
        sidecar_key = self._sidecar_key(epoch, snap_tick)
        sidecar_content = json.dumps(metadata, indent=2).encode()
        await self._uploader.put_small_file(
            sidecar_key, sidecar_content
        )

        # Update index
        index_key = self._index_key(epoch)
        index_data = {
            "epoch": epoch,
            "tick": snap_tick,
            "file": f"ep{epoch}-t{snap_tick}-snap.zip",
            "sidecar": f"ep{epoch}-t{snap_tick}-snap.json",
            "timestamp": metadata["timestamp"],
            "checksum": metadata.get("checksum", ""),
            "size_bytes": metadata.get("size_bytes", 0),
        }
        await self._uploader.put_small_file(
            index_key,
            json.dumps(index_data, indent=2).encode(),
        )

        self._last_snapshot_epoch = epoch
        await self._alert_manager.send_alert(
            "info",
            "snapshot_uploaded",
            metadata,
        )

    async def _upload_with_retries(
        self,
        file_path: Path,
        metadata: dict,
        remote_key: str,
    ) -> bool:
        """Upload with retry logic."""
        for attempt in range(self._config.upload_retry_count + 1):
            result = await self._uploader.upload(
                file_path, metadata, remote_key
            )
            if result.success:
                logger.info(
                    f"Upload successful: {result.remote_url} "
                    f"({result.bytes_uploaded} bytes, "
                    f"{result.duration_seconds:.1f}s)"
                )
                return True

            logger.warning(
                f"Upload attempt {attempt + 1} failed: "
                f"{result.error_message}"
            )
            if attempt < self._config.upload_retry_count:
                await asyncio.sleep(
                    self._config.upload_retry_delay_seconds
                )

        return False

    async def _sync_and_package_with_retries(
        self,
        snap_dir: Path,
        epoch: int,
        tick: int,
    ) -> bool:
        """Rsync + remote package with retry logic."""
        for attempt in range(self._config.upload_retry_count + 1):
            result = await self._uploader.sync_and_package(
                snap_dir, epoch, tick
            )
            if result.success:
                logger.info(
                    f"Sync+package successful: {result.remote_url} "
                    f"({result.bytes_uploaded} bytes transferred, "
                    f"{result.duration_seconds:.1f}s)"
                )
                return True

            logger.warning(
                f"Sync+package attempt {attempt + 1} failed: "
                f"{result.error_message}"
            )
            if attempt < self._config.upload_retry_count:
                await asyncio.sleep(
                    self._config.upload_retry_delay_seconds
                )

        return False

    # ------------------------------------------------------------------
    # Remote snapshot cleanup
    # ------------------------------------------------------------------
    async def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots from remote, keeping only the N most recent."""
        keep = self._config.snapshot_keep_count
        if keep <= 0:
            return  # cleanup disabled

        # Collect all snapshots across all epoch directories
        all_snapshots: list[tuple[int, int]] = []  # (epoch, tick)

        try:
            entries = await self._uploader.list_remote_dir("")
        except Exception as e:
            logger.debug(f"Cannot list remote root for cleanup: {e}")
            return

        # Filter to numeric epoch directory names
        epoch_nums: list[int] = []
        for name in entries:
            try:
                epoch_nums.append(int(name))
            except ValueError:
                continue

        # For each epoch directory, find snapshot archives
        for ep in epoch_nums:
            try:
                files = await self._uploader.list_remote_dir(str(ep))
            except Exception:
                continue

            for fname in files:
                m = _SNAPSHOT_ZIP_RE.match(fname)
                if m:
                    all_snapshots.append((int(m.group(1)), int(m.group(2))))

        if len(all_snapshots) <= keep:
            return  # nothing to clean up

        # Sort by (epoch, tick) descending — most recent first
        all_snapshots.sort(reverse=True)
        to_keep = set(all_snapshots[:keep])
        to_delete = all_snapshots[keep:]

        kept_epochs = {s[0] for s in to_keep}
        deleted_count = 0

        for epoch, tick in to_delete:
            if epoch in kept_epochs:
                # Epoch still has kept snapshots — only delete this archive
                await self._uploader.delete_file(
                    self._archive_key(epoch, tick)
                )
                await self._uploader.delete_file(
                    self._sidecar_key(epoch, tick)
                )
            # Entire epoch will be removed below
            deleted_count += 1

        # Remove entire epoch directories that have no remaining snapshots
        removed_epochs = {s[0] for s in to_delete} - kept_epochs
        for epoch in removed_epochs:
            logger.info(f"Removing old epoch directory: {epoch}")
            await self._uploader.delete_remote_dir(str(epoch))

        if deleted_count > 0:
            logger.info(
                f"Cleaned up {deleted_count} old snapshot(s), "
                f"keeping {keep} most recent"
            )
