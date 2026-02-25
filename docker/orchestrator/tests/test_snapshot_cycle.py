from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.alerting import AlertManager
from app.config import AlertingConfig, SourceConfig
from app.downloaders.base import BaseDownloader
from app.models import TickInfo, UploadResult
from app.node_client import NodeClient
from app.snapshot_cycle import LOCK_TICK_THRESHOLD, SnapshotCycle
from app.state_manager import StateManager
from app.uploaders.base import BaseUploader, RemotePackagingUploader


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------
class FakeUploader(BaseUploader):
    """In-memory uploader that tracks all operations."""

    def __init__(self) -> None:
        self.uploaded: dict[str, bytes] = {}
        self.small_files: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.deleted_dirs: list[str] = []
        self.upload_calls: list[tuple[Path, dict, str]] = []
        self.upload_result = UploadResult(
            success=True,
            remote_url="scp://host/snap.zip",
            bytes_uploaded=1000,
            duration_seconds=1.0,
        )

    async def upload(
        self, file_path: Path, metadata: dict, remote_key: str
    ) -> UploadResult:
        self.upload_calls.append((file_path, metadata, remote_key))
        if file_path.exists():
            self.uploaded[remote_key] = file_path.read_bytes()
        return self.upload_result

    async def put_small_file(self, remote_key: str, content: bytes) -> bool:
        self.small_files[remote_key] = content
        return True

    async def get_small_file(self, remote_key: str) -> Optional[bytes]:
        return self.small_files.get(remote_key)

    async def delete_file(self, remote_key: str) -> bool:
        self.deleted.append(remote_key)
        if remote_key in self.small_files:
            del self.small_files[remote_key]
        return True

    async def list_remote_dir(self, remote_prefix: str) -> list[str]:
        prefix = remote_prefix.strip("/")
        entries: set[str] = set()
        for key in list(self.small_files) + list(self.uploaded):
            if prefix:
                if key.startswith(prefix + "/"):
                    rest = key[len(prefix) + 1:]
                    entries.add(rest.split("/")[0])
            else:
                entries.add(key.split("/")[0])
        return sorted(entries)

    async def delete_remote_dir(self, remote_prefix: str) -> bool:
        prefix = remote_prefix.strip("/") + "/"
        for store in (self.small_files, self.uploaded):
            for k in list(store):
                if k.startswith(prefix):
                    del store[k]
        self.deleted_dirs.append(remote_prefix)
        return True

    async def check_health(self) -> bool:
        return True

    def get_name(self) -> str:
        return "fake"


class FakeDownloader(BaseDownloader):
    async def download(self, url: str, dest_path: Path) -> Path:
        return dest_path

    async def close(self) -> None:
        pass


def _make_tick_info(
    epoch: int = 198,
    tick: int = 43101500,
    saving: bool = False,
) -> TickInfo:
    return TickInfo(
        epoch=epoch,
        tick=tick,
        initial_tick=43100000,
        aligned_votes=500,
        misaligned_votes=0,
        main_aux_status=0,
        is_saving_snapshot=saving,
    )


def _populate_snapshot_dir(data_dir: Path, epoch: int = 198) -> None:
    """Create the minimal files that StateManager.has_snapshot_directory expects."""
    ep_dir = data_dir / f"ep{epoch}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "system.snp",
        "snapshotNodeMiningState",
        "snapshotSpectrumDigest",
        "snapshotUniverseDigest",
        "snapshotComputerDigest",
        "snapshotMinerSolutionFlag",
    ]:
        (ep_dir / name).write_bytes(b"x")

    # State files for packaging
    (data_dir / "spectrum.198").write_bytes(b"spectrum")
    (data_dir / "universe.198").write_bytes(b"universe")
    (data_dir / "system").write_bytes(b"system")


@pytest.fixture
def data_dir(tmp_path) -> Path:
    d = tmp_path / "qubic"
    d.mkdir()
    return d


@pytest.fixture
def uploader() -> FakeUploader:
    return FakeUploader()


@pytest.fixture
def alert_manager() -> AlertManager:
    return AlertManager(config=AlertingConfig(enabled=False))


@pytest.fixture
def node_client() -> AsyncMock:
    client = AsyncMock(spec=NodeClient)
    return client


@pytest.fixture
def source_config() -> SourceConfig:
    return SourceConfig(
        snapshot_interval_seconds=60,
        snapshot_poll_interval_seconds=1,
        snapshot_wait_timeout_seconds=5,
    )


def _make_cycle(
    source_config: SourceConfig,
    node_client: AsyncMock,
    uploader: FakeUploader,
    alert_manager: AlertManager,
    data_dir: Path,
) -> SnapshotCycle:
    sm = StateManager(data_dir, FakeDownloader())
    return SnapshotCycle(
        config=source_config,
        node_client=node_client,
        state_manager=sm,
        uploader=uploader,
        alert_manager=alert_manager,
        data_dir=data_dir,
    )


# ---------------------------------------------------------------------------
# Key naming tests
# ---------------------------------------------------------------------------
class TestKeyNaming:
    def test_archive_key(self):
        assert (
            SnapshotCycle._archive_key(198, 43101500)
            == "198/ep198-t43101500-snap.zip"
        )

    def test_sidecar_key(self):
        assert (
            SnapshotCycle._sidecar_key(198, 43101500)
            == "198/ep198-t43101500-snap.json"
        )

    def test_index_key(self):
        assert (
            SnapshotCycle._index_key(198) == "198/ep198-latest-snap.json"
        )

    def test_lock_key(self):
        assert SnapshotCycle._lock_key(198) == "198/snap.lock"


# ---------------------------------------------------------------------------
# Lock coordination tests
# ---------------------------------------------------------------------------
class TestLockCoordination:
    @pytest.mark.asyncio
    async def test_skip_when_recent_index(
        self, source_config, node_client, uploader, alert_manager, data_dir
    ):
        """If remote index shows a recent snapshot, skip."""
        cycle = _make_cycle(
            source_config, node_client, uploader, alert_manager, data_dir
        )
        # Pre-populate index in uploader
        index_data = {"epoch": 198, "tick": 43101500}
        uploader.small_files["198/ep198-latest-snap.json"] = json.dumps(
            index_data
        ).encode()

        # Local tick is within threshold
        result = await cycle._check_remote_lock(198, 43101550)
        assert result is True  # should skip

    @pytest.mark.asyncio
    async def test_proceed_when_index_old(
        self, source_config, node_client, uploader, alert_manager, data_dir
    ):
        """If remote index snapshot is old (tick diff >= threshold), proceed."""
        cycle = _make_cycle(
            source_config, node_client, uploader, alert_manager, data_dir
        )
        index_data = {"epoch": 198, "tick": 43101000}
        uploader.small_files["198/ep198-latest-snap.json"] = json.dumps(
            index_data
        ).encode()

        result = await cycle._check_remote_lock(
            198, 43101000 + LOCK_TICK_THRESHOLD
        )
        assert result is False  # should proceed

    @pytest.mark.asyncio
    async def test_skip_when_lock_held(
        self, source_config, node_client, uploader, alert_manager, data_dir
    ):
        """If another source holds the lock at a nearby tick, skip."""
        cycle = _make_cycle(
            source_config, node_client, uploader, alert_manager, data_dir
        )
        lock_data = {
            "tick": 43101500,
            "timestamp": "2025-01-01T00:00:00Z",
            "node_id": "other_node",
        }
        uploader.small_files["198/snap.lock"] = json.dumps(
            lock_data
        ).encode()

        result = await cycle._check_remote_lock(198, 43101550)
        assert result is True  # skip — lock tick is close

    @pytest.mark.asyncio
    async def test_proceed_when_lock_stale(
        self, source_config, node_client, uploader, alert_manager, data_dir
    ):
        """If the lock is from a much older tick, treat as stale."""
        cycle = _make_cycle(
            source_config, node_client, uploader, alert_manager, data_dir
        )
        lock_data = {
            "tick": 43101000,
            "timestamp": "2025-01-01T00:00:00Z",
            "node_id": "other_node",
        }
        uploader.small_files["198/snap.lock"] = json.dumps(
            lock_data
        ).encode()

        result = await cycle._check_remote_lock(
            198, 43101000 + LOCK_TICK_THRESHOLD
        )
        assert result is False  # proceed — lock is stale

    @pytest.mark.asyncio
    async def test_proceed_when_no_lock_or_index(
        self, source_config, node_client, uploader, alert_manager, data_dir
    ):
        """If no remote index and no lock, proceed."""
        cycle = _make_cycle(
            source_config, node_client, uploader, alert_manager, data_dir
        )
        result = await cycle._check_remote_lock(198, 43101500)
        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_and_release_lock(
        self, source_config, node_client, uploader, alert_manager, data_dir
    ):
        cycle = _make_cycle(
            source_config, node_client, uploader, alert_manager, data_dir
        )
        ok = await cycle._acquire_lock(198, 43101500)
        assert ok is True
        assert "198/snap.lock" in uploader.small_files

        lock = json.loads(uploader.small_files["198/snap.lock"])
        assert lock["tick"] == 43101500
        assert "node_id" in lock

        await cycle._release_lock(198)
        assert "198/snap.lock" in uploader.deleted


# ---------------------------------------------------------------------------
# Full cycle tests
# ---------------------------------------------------------------------------
class TestExecuteCycle:
    @pytest.mark.asyncio
    async def test_full_cycle_success(
        self, source_config, node_client, uploader, alert_manager, data_dir
    ):
        """Happy-path: full snapshot cycle succeeds, uploads archive + sidecar + index."""
        _populate_snapshot_dir(data_dir, 198)

        # Mock node_client behavior
        tick_info = _make_tick_info(tick=43101500, saving=False)
        saving_info = _make_tick_info(tick=43101500, saving=True)
        done_info = _make_tick_info(tick=43101510, saving=False)

        # pre-check returns healthy, then saving starts, then completes
        node_client.get_tick_info = AsyncMock(
            side_effect=[tick_info, saving_info, done_info, done_info]
        )
        node_client.request_save_snapshot = AsyncMock(return_value=True)

        cycle = _make_cycle(
            source_config, node_client, uploader, alert_manager, data_dir
        )
        await cycle._execute_cycle()

        # Archive uploaded with tick-specific key
        assert len(uploader.upload_calls) == 1
        _, metadata, remote_key = uploader.upload_calls[0]
        assert "ep198-t43101510-snap.zip" in remote_key
        assert metadata["epoch"] == 198
        assert metadata["tick"] == 43101510

        # Sidecar uploaded
        sidecar_key = "198/ep198-t43101510-snap.json"
        assert sidecar_key in uploader.small_files

        # Index updated
        index_key = "198/ep198-latest-snap.json"
        assert index_key in uploader.small_files
        index = json.loads(uploader.small_files[index_key])
        assert index["tick"] == 43101510
        assert index["file"] == "ep198-t43101510-snap.zip"

        # Lock released
        assert "198/snap.lock" in uploader.deleted

    @pytest.mark.asyncio
    async def test_skip_when_node_already_saving(
        self, source_config, node_client, uploader, alert_manager, data_dir
    ):
        """If the node is already saving, skip the cycle."""
        node_client.get_tick_info = AsyncMock(
            return_value=_make_tick_info(saving=True)
        )

        cycle = _make_cycle(
            source_config, node_client, uploader, alert_manager, data_dir
        )
        await cycle._execute_cycle()

        assert len(uploader.upload_calls) == 0

    @pytest.mark.asyncio
    async def test_skip_when_node_unreachable(
        self, source_config, node_client, uploader, alert_manager, data_dir
    ):
        """If the node is unreachable, skip."""
        node_client.get_tick_info = AsyncMock(
            side_effect=Exception("connection refused")
        )

        cycle = _make_cycle(
            source_config, node_client, uploader, alert_manager, data_dir
        )
        await cycle._execute_cycle()

        assert len(uploader.upload_calls) == 0

    @pytest.mark.asyncio
    async def test_lock_released_on_failure(
        self, source_config, node_client, uploader, alert_manager, data_dir
    ):
        """Lock is released even when the cycle fails midway."""
        tick_info = _make_tick_info(saving=False)
        node_client.get_tick_info = AsyncMock(return_value=tick_info)
        node_client.request_save_snapshot = AsyncMock(return_value=False)

        cycle = _make_cycle(
            source_config, node_client, uploader, alert_manager, data_dir
        )
        await cycle._execute_cycle()

        # Lock was acquired then released
        assert "198/snap.lock" in uploader.deleted


# ---------------------------------------------------------------------------
# Rsync cycle tests
# ---------------------------------------------------------------------------
class FakeRsyncUploader(FakeUploader):
    """In-memory uploader that also implements RemotePackagingUploader."""

    def __init__(self) -> None:
        super().__init__()
        self.sync_calls: list[tuple] = []
        self.sync_result = UploadResult(
            success=True,
            remote_url="rsync://host/198/ep198-t43101510-snap.zip",
            bytes_uploaded=500,
            duration_seconds=2.0,
        )
        self.remote_checksum = ("abcd1234" * 8, 1000)

    async def sync_and_package(
        self,
        snap_dir: Path,
        epoch: int,
        tick: int,
    ) -> UploadResult:
        self.sync_calls.append((snap_dir, epoch, tick))
        return self.sync_result

    async def get_remote_checksum(self, remote_key: str) -> tuple:
        return self.remote_checksum

    def get_name(self) -> str:
        return "fake_rsync"


# Verify FakeRsyncUploader satisfies the protocol
assert isinstance(FakeRsyncUploader(), RemotePackagingUploader)


class TestRsyncCycle:
    @pytest.mark.asyncio
    async def test_rsync_cycle_skips_local_packaging(
        self, source_config, node_client, alert_manager, data_dir
    ):
        """Rsync path: should call sync_and_package, not upload."""
        _populate_snapshot_dir(data_dir, 198)
        uploader = FakeRsyncUploader()

        tick_info = _make_tick_info(tick=43101500, saving=False)
        saving_info = _make_tick_info(tick=43101500, saving=True)
        done_info = _make_tick_info(tick=43101510, saving=False)

        node_client.get_tick_info = AsyncMock(
            side_effect=[tick_info, saving_info, done_info, done_info]
        )
        node_client.request_save_snapshot = AsyncMock(return_value=True)

        cycle = _make_cycle(
            source_config, node_client, uploader, alert_manager, data_dir
        )
        await cycle._execute_cycle()

        # sync_and_package was called, not upload
        assert len(uploader.sync_calls) == 1
        assert len(uploader.upload_calls) == 0

        # Verify sync args
        snap_dir, epoch, tick = uploader.sync_calls[0]
        assert epoch == 198
        assert tick == 43101510
        assert snap_dir == data_dir / "ep198"

        # Sidecar and index were still written
        assert "198/ep198-t43101510-snap.json" in uploader.small_files
        assert "198/ep198-latest-snap.json" in uploader.small_files

        # Index contains correct data
        index = json.loads(uploader.small_files["198/ep198-latest-snap.json"])
        assert index["tick"] == 43101510
        assert index["file"] == "ep198-t43101510-snap.zip"
        assert index["checksum"] == "abcd1234" * 8
        assert index["size_bytes"] == 1000

        # Lock released
        assert "198/snap.lock" in uploader.deleted

    @pytest.mark.asyncio
    async def test_rsync_cycle_retries_on_failure(
        self, source_config, node_client, alert_manager, data_dir
    ):
        """Rsync path should retry on failure."""
        _populate_snapshot_dir(data_dir, 198)
        uploader = FakeRsyncUploader()

        fail_result = UploadResult(
            success=False, error_message="connection reset"
        )
        success_result = UploadResult(
            success=True,
            remote_url="rsync://host/snap.zip",
            bytes_uploaded=500,
            duration_seconds=1.0,
        )

        call_count = 0

        async def _mock_sync(snap_dir, epoch, tick):
            nonlocal call_count
            call_count += 1
            uploader.sync_calls.append((snap_dir, epoch, tick))
            if call_count == 1:
                return fail_result
            return success_result

        uploader.sync_and_package = _mock_sync

        tick_info = _make_tick_info(tick=43101500, saving=False)
        saving_info = _make_tick_info(tick=43101500, saving=True)
        done_info = _make_tick_info(tick=43101510, saving=False)

        node_client.get_tick_info = AsyncMock(
            side_effect=[tick_info, saving_info, done_info, done_info]
        )
        node_client.request_save_snapshot = AsyncMock(return_value=True)

        retry_config = SourceConfig(
            snapshot_interval_seconds=60,
            snapshot_poll_interval_seconds=1,
            snapshot_wait_timeout_seconds=5,
            upload_retry_count=3,
            upload_retry_delay_seconds=0,
        )

        cycle = _make_cycle(
            retry_config, node_client, uploader, alert_manager, data_dir
        )
        await cycle._execute_cycle()

        # 1 fail + 1 success
        assert len(uploader.sync_calls) == 2

        # Sidecar and index written after success
        assert "198/ep198-t43101510-snap.json" in uploader.small_files
        assert "198/ep198-latest-snap.json" in uploader.small_files

    @pytest.mark.asyncio
    async def test_rsync_lock_released_on_failure(
        self, source_config, node_client, alert_manager, data_dir
    ):
        """Lock is released even when rsync path fails."""
        tick_info = _make_tick_info(saving=False)
        node_client.get_tick_info = AsyncMock(return_value=tick_info)
        node_client.request_save_snapshot = AsyncMock(return_value=False)

        uploader = FakeRsyncUploader()
        cycle = _make_cycle(
            source_config, node_client, uploader, alert_manager, data_dir
        )
        await cycle._execute_cycle()

        # Lock was acquired then released
        assert "198/snap.lock" in uploader.deleted


# ---------------------------------------------------------------------------
# Snapshot cleanup tests
# ---------------------------------------------------------------------------
class TestCleanupOldSnapshots:
    @pytest.mark.asyncio
    async def test_cleanup_removes_oldest_in_same_epoch(
        self, source_config, node_client, alert_manager, data_dir
    ):
        """With 3 snapshots and keep=2, the oldest is deleted."""
        uploader = FakeUploader()
        # Populate 3 snapshots in same epoch
        for tick in [43101500, 43101510, 43101520]:
            uploader.small_files[f"198/ep198-t{tick}-snap.zip"] = b"z"
            uploader.small_files[f"198/ep198-t{tick}-snap.json"] = b"{}"

        config = SourceConfig(
            snapshot_interval_seconds=60,
            snapshot_poll_interval_seconds=1,
            snapshot_wait_timeout_seconds=5,
            snapshot_keep_count=2,
        )
        cycle = _make_cycle(
            config, node_client, uploader, alert_manager, data_dir
        )
        await cycle._cleanup_old_snapshots()

        # Oldest (43101500) should be deleted
        assert "198/ep198-t43101500-snap.zip" in uploader.deleted
        assert "198/ep198-t43101500-snap.json" in uploader.deleted
        # Newer two should still exist
        assert "198/ep198-t43101510-snap.zip" in uploader.small_files
        assert "198/ep198-t43101520-snap.zip" in uploader.small_files

    @pytest.mark.asyncio
    async def test_cleanup_removes_old_epoch_snapshot_files(
        self, source_config, node_client, alert_manager, data_dir
    ):
        """Old epoch snapshot files are removed but directory is preserved."""
        uploader = FakeUploader()
        # Epoch 197: one old snapshot + index
        uploader.small_files["197/ep197-t42900000-snap.zip"] = b"z"
        uploader.small_files["197/ep197-t42900000-snap.json"] = b"{}"
        uploader.small_files["197/ep197-latest-snap.json"] = b"{}"
        # Epoch 198: two snapshots
        uploader.small_files["198/ep198-t43101500-snap.zip"] = b"z"
        uploader.small_files["198/ep198-t43101500-snap.json"] = b"{}"
        uploader.small_files["198/ep198-t43101510-snap.zip"] = b"z"
        uploader.small_files["198/ep198-t43101510-snap.json"] = b"{}"

        config = SourceConfig(
            snapshot_interval_seconds=60,
            snapshot_poll_interval_seconds=1,
            snapshot_wait_timeout_seconds=5,
            snapshot_keep_count=2,
        )
        cycle = _make_cycle(
            config, node_client, uploader, alert_manager, data_dir
        )
        await cycle._cleanup_old_snapshots()

        # Epoch directory NOT removed (preserves epoch files)
        assert "197" not in uploader.deleted_dirs
        # Snapshot files individually deleted
        assert "197/ep197-t42900000-snap.zip" in uploader.deleted
        assert "197/ep197-t42900000-snap.json" in uploader.deleted
        # Index and lock cleaned up for epochs with no remaining snapshots
        assert "197/ep197-latest-snap.json" in uploader.deleted
        assert "197/snap.lock" in uploader.deleted
        # Epoch 198 snapshots untouched
        assert "198/ep198-t43101500-snap.zip" in uploader.small_files
        assert "198/ep198-t43101510-snap.zip" in uploader.small_files

    @pytest.mark.asyncio
    async def test_cleanup_disabled_when_keep_zero(
        self, source_config, node_client, alert_manager, data_dir
    ):
        """No cleanup when snapshot_keep_count=0."""
        uploader = FakeUploader()
        uploader.small_files["197/ep197-t42900000-snap.zip"] = b"z"
        uploader.small_files["198/ep198-t43101500-snap.zip"] = b"z"

        config = SourceConfig(
            snapshot_interval_seconds=60,
            snapshot_poll_interval_seconds=1,
            snapshot_wait_timeout_seconds=5,
            snapshot_keep_count=0,
        )
        cycle = _make_cycle(
            config, node_client, uploader, alert_manager, data_dir
        )
        await cycle._cleanup_old_snapshots()

        # Nothing deleted
        assert len(uploader.deleted) == 0
        assert len(uploader.deleted_dirs) == 0

    @pytest.mark.asyncio
    async def test_cleanup_noop_when_within_limit(
        self, source_config, node_client, alert_manager, data_dir
    ):
        """No cleanup when snapshot count <= keep."""
        uploader = FakeUploader()
        uploader.small_files["198/ep198-t43101500-snap.zip"] = b"z"
        uploader.small_files["198/ep198-t43101510-snap.zip"] = b"z"

        config = SourceConfig(
            snapshot_interval_seconds=60,
            snapshot_poll_interval_seconds=1,
            snapshot_wait_timeout_seconds=5,
            snapshot_keep_count=2,
        )
        cycle = _make_cycle(
            config, node_client, uploader, alert_manager, data_dir
        )
        await cycle._cleanup_old_snapshots()

        assert len(uploader.deleted) == 0
        assert len(uploader.deleted_dirs) == 0

    @pytest.mark.asyncio
    async def test_cleanup_across_multiple_epochs(
        self, source_config, node_client, alert_manager, data_dir
    ):
        """Keep=2 across epochs: keep newest from ep198, delete ep196 and ep197."""
        uploader = FakeUploader()
        uploader.small_files["196/ep196-t42800000-snap.zip"] = b"z"
        uploader.small_files["196/ep196-t42800000-snap.json"] = b"{}"
        uploader.small_files["197/ep197-t42900000-snap.zip"] = b"z"
        uploader.small_files["197/ep197-t42900000-snap.json"] = b"{}"
        uploader.small_files["198/ep198-t43101500-snap.zip"] = b"z"
        uploader.small_files["198/ep198-t43101500-snap.json"] = b"{}"
        uploader.small_files["198/ep198-t43101510-snap.zip"] = b"z"
        uploader.small_files["198/ep198-t43101510-snap.json"] = b"{}"

        config = SourceConfig(
            snapshot_interval_seconds=60,
            snapshot_poll_interval_seconds=1,
            snapshot_wait_timeout_seconds=5,
            snapshot_keep_count=2,
        )
        cycle = _make_cycle(
            config, node_client, uploader, alert_manager, data_dir
        )
        await cycle._cleanup_old_snapshots()

        # Epoch directories NOT removed (preserves epoch files)
        assert "196" not in uploader.deleted_dirs
        assert "197" not in uploader.deleted_dirs
        # Snapshot files individually deleted for ep196 and ep197
        assert "196/ep196-t42800000-snap.zip" in uploader.deleted
        assert "196/ep196-t42800000-snap.json" in uploader.deleted
        assert "196/ep196-latest-snap.json" in uploader.deleted
        assert "197/ep197-t42900000-snap.zip" in uploader.deleted
        assert "197/ep197-t42900000-snap.json" in uploader.deleted
        assert "197/ep197-latest-snap.json" in uploader.deleted
        # ep198 untouched
        assert "198/ep198-t43101500-snap.zip" in uploader.small_files
        assert "198/ep198-t43101510-snap.zip" in uploader.small_files
