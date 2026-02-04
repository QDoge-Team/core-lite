from __future__ import annotations

import glob
import hashlib
import logging
import os
import re
import shutil
import struct
import tempfile
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from app.downloaders.base import BaseDownloader
from app.models import SnapshotMeta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ZIP format constants
# ---------------------------------------------------------------------------
_LOCAL_FILE_HEADER_SIG = 0x04034B50
_CENTRAL_DIR_SIG = 0x02014B50
_END_CENTRAL_DIR_SIG = 0x06054B50
_ZIP64_END_SIG = 0x06064B50
_ZIP64_LOCATOR_SIG = 0x07064B50
_CHUNK_SIZE = 1024 * 1024  # 1 MB


def _compress_file_worker(
    file_path: str,
    arcname: str,
    tmp_dir: str,
) -> dict:
    """Compress a single file using raw deflate (zlib releases the GIL).

    Writes compressed data to a temp file and returns metadata needed
    to assemble the final ZIP archive.
    """
    crc = 0
    file_size = 0
    compressed_size = 0

    compressor = zlib.compressobj(
        zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15
    )

    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".zpart")
    try:
        with os.fdopen(fd, "wb") as tmp_f, open(file_path, "rb") as src:
            while True:
                chunk = src.read(_CHUNK_SIZE)
                if not chunk:
                    break
                crc = zlib.crc32(chunk, crc)
                file_size += len(chunk)
                compressed = compressor.compress(chunk)
                if compressed:
                    tmp_f.write(compressed)
                    compressed_size += len(compressed)
            # Flush remaining
            tail = compressor.flush(zlib.Z_FINISH)
            if tail:
                tmp_f.write(tail)
                compressed_size += len(tail)
    except BaseException:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return {
        "arcname": arcname,
        "tmp_path": tmp_path,
        "crc32": crc & 0xFFFFFFFF,
        "file_size": file_size,
        "compressed_size": compressed_size,
    }

# Files the Qubic node needs per epoch
CRITICAL_FILES = ["spectrum", "universe"]

# Snapshot directory files written by saveAllNodeStates()
SNAPSHOT_DIR_FILES = [
    "system.snp",
    "snapshotNodeMiningState",
    "snapshotSpectrumDigest",
    "snapshotUniverseDigest",
    "snapshotComputerDigest",
    "snapshotMinerSolutionFlag",
]


class StateManager:
    """Manages local state files for the Qubic node."""

    def __init__(self, data_dir: Path, downloader: BaseDownloader) -> None:
        self._data_dir = data_dir
        self._downloader = downloader

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def get_local_epoch(self) -> Optional[int]:
        """Detect the epoch from existing spectrum files."""
        pattern = str(self._data_dir / "spectrum.*")
        files = glob.glob(pattern)
        if not files:
            return None
        # Get the highest epoch number
        epochs = []
        for f in files:
            ext = Path(f).suffix.lstrip(".")
            try:
                epochs.append(int(ext))
            except ValueError:
                continue
        return max(epochs) if epochs else None

    def has_valid_state_files(self, epoch: int) -> bool:
        """Check if critical state files exist for the given epoch."""
        for name in CRITICAL_FILES:
            path = self._data_dir / f"{name}.{epoch}"
            if not path.exists() or path.stat().st_size == 0:
                logger.debug(f"Missing or empty: {path}")
                return False
        return True

    def get_snapshot_directory(self, epoch: int) -> Optional[Path]:
        """Return the snapshot directory path, or None if not found."""
        for name in [f"ep{epoch}", "ep"]:
            snap_dir = self._data_dir / name
            if snap_dir.is_dir():
                return snap_dir
        return None

    def has_snapshot_directory(self, epoch: int) -> bool:
        """Check if snapshot state directory exists with required files."""
        snap_dir = self.get_snapshot_directory(epoch)
        if snap_dir is None:
            return False
        for name in SNAPSHOT_DIR_FILES:
            if not (snap_dir / name).exists():
                return False
        return True

    async def download_epoch_files(
        self, epoch: int, url: str
    ) -> bool:
        """Download and extract epoch files from a URL."""
        zip_path = self._data_dir / "epoch_files.zip"
        try:
            await self._downloader.download(url, zip_path)
            self._extract_and_rename(zip_path, epoch)
            return True
        except Exception as e:
            logger.error(f"Failed to download epoch files: {e}")
            return False
        finally:
            if zip_path.exists():
                zip_path.unlink()

    async def download_snapshot(
        self, snapshot_meta: SnapshotMeta
    ) -> bool:
        """Download a snapshot archive."""
        zip_path = self._data_dir / "snapshot.zip"
        try:
            await self._downloader.download(snapshot_meta.url, zip_path)
            self._extract_and_rename(zip_path, snapshot_meta.epoch)
            return True
        except Exception as e:
            logger.error(f"Failed to download snapshot: {e}")
            return False
        finally:
            if zip_path.exists():
                zip_path.unlink()

    def _extract_and_rename(self, zip_path: Path, epoch: int) -> None:
        """Extract zip and rename files to match the epoch number."""
        logger.info(f"Extracting {zip_path} to {self._data_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self._data_dir)

        # Rename files with .000 extension (or wrong epoch) to current epoch
        for path in self._data_dir.iterdir():
            if not path.is_file():
                continue
            name = path.name
            stem = path.stem
            ext = path.suffix.lstrip(".")

            # Only process known file types
            is_state_file = stem.startswith(("spectrum", "universe", "contract"))
            if not is_state_file:
                continue

            # Skip if already correct epoch
            if ext == str(epoch):
                continue

            # Rename .000 or other epoch to current epoch
            try:
                int(ext)  # Ensure extension is numeric
                new_path = self._data_dir / f"{stem}.{epoch}"
                path.rename(new_path)
                logger.info(f"Renamed {name} -> {new_path.name}")
            except ValueError:
                continue

    def list_state_files(self, epoch: int) -> list[Path]:
        """List all state files for a given epoch."""
        files = []
        for pattern in [
            f"spectrum.{epoch}",
            f"universe.{epoch}",
            f"contract*.{epoch}",
            f"score.{epoch}",
            f"custom_mining_cache.{epoch}",
            "system",
        ]:
            files.extend(self._data_dir.glob(pattern))
        return sorted(files)

    def list_snapshot_files(self, epoch: int) -> list[Path]:
        """List all files in the snapshot directory for packaging."""
        files = []
        # Main state files
        files.extend(self.list_state_files(epoch))
        # Snapshot directory
        for snap_dir_name in [f"ep{epoch}", "ep"]:
            snap_dir = self._data_dir / snap_dir_name
            if snap_dir.is_dir():
                for f in snap_dir.rglob("*"):
                    if f.is_file():
                        files.append(f)
                break
        # Page files from SwapVirtualMemory (.pg files in subdirectories)
        for sub_dir in sorted(self._data_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            if sub_dir.name.startswith("ep"):
                continue  # already handled above
            pg_files = sorted(sub_dir.glob("*.pg"))
            if pg_files:
                files.extend(pg_files)
        return files

    @staticmethod
    def _build_zip_from_entries(
        entries: list[dict], archive_path: Path
    ) -> None:
        """Assemble a ZIP file from pre-compressed entries.

        Each entry dict has: arcname, tmp_path, crc32, file_size,
        compressed_size.  Uses ZIP64 unconditionally for simplicity.
        """
        central_dir_entries: list[bytes] = []
        with open(archive_path, "wb") as out:
            for entry in entries:
                arcname_bytes = entry["arcname"].encode("utf-8")
                offset = out.tell()

                # ZIP64 extra field for local header:
                #   Header ID (0x0001) + Data Size (28) +
                #   Original Size (8) + Compressed Size (8) + Offset (8)
                # We include offset in local extra even though it's not
                # strictly required — keeps it consistent with central dir.
                zip64_extra = struct.pack(
                    "<HH QQQ",
                    0x0001,
                    24,
                    entry["file_size"],
                    entry["compressed_size"],
                    offset,
                )

                # Local file header
                local_header = struct.pack(
                    "<I HHHHH III HH",
                    _LOCAL_FILE_HEADER_SIG,
                    45,                       # version needed (4.5 for ZIP64)
                    0,                        # general purpose bit flag
                    8,                        # compression method (deflate)
                    0,                        # last mod time
                    0,                        # last mod date
                    entry["crc32"],
                    0xFFFFFFFF,               # compressed size (ZIP64)
                    0xFFFFFFFF,               # uncompressed size (ZIP64)
                    len(arcname_bytes),
                    len(zip64_extra),
                )
                out.write(local_header)
                out.write(arcname_bytes)
                out.write(zip64_extra)

                # Copy pre-compressed data from temp file
                with open(entry["tmp_path"], "rb") as tmp_f:
                    while True:
                        chunk = tmp_f.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)

                # Central directory entry
                cd_zip64_extra = struct.pack(
                    "<HH QQQ",
                    0x0001,
                    24,
                    entry["file_size"],
                    entry["compressed_size"],
                    offset,
                )
                cd_entry = struct.pack(
                    "<I HHHH HH I II HH HHH II",
                    _CENTRAL_DIR_SIG,
                    45,                       # version made by
                    45,                       # version needed
                    0,                        # general purpose bit flag
                    8,                        # compression method
                    0,                        # last mod time
                    0,                        # last mod date
                    entry["crc32"],
                    0xFFFFFFFF,               # compressed size (ZIP64)
                    0xFFFFFFFF,               # uncompressed size (ZIP64)
                    len(arcname_bytes),
                    len(cd_zip64_extra),
                    0,                        # file comment length
                    0,                        # disk number start
                    0,                        # internal file attributes
                    0,                        # external file attributes
                    0xFFFFFFFF,               # local header offset (ZIP64)
                )
                central_dir_entries.append(
                    cd_entry + arcname_bytes + cd_zip64_extra
                )

            # Write central directory
            cd_offset = out.tell()
            for cd in central_dir_entries:
                out.write(cd)
            cd_size = out.tell() - cd_offset
            num_entries = len(central_dir_entries)

            # ZIP64 end of central directory record
            out.write(struct.pack(
                "<I Q HH II QQ QQ",
                _ZIP64_END_SIG,
                44,                           # size of remaining record
                45,                           # version made by
                45,                           # version needed
                0,                            # disk number
                0,                            # disk with central dir
                num_entries,                  # entries on this disk
                num_entries,                  # total entries
                cd_size,
                cd_offset,
            ))

            # ZIP64 end of central directory locator
            zip64_end_offset = cd_offset + cd_size
            out.write(struct.pack(
                "<I I Q I",
                _ZIP64_LOCATOR_SIG,
                0,                            # disk with ZIP64 EOCD
                zip64_end_offset,
                1,                            # total disks
            ))

            # Standard end of central directory record
            out.write(struct.pack(
                "<I HH HH II H",
                _END_CENTRAL_DIR_SIG,
                0xFFFF if num_entries > 0xFFFF else 0,
                0xFFFF if num_entries > 0xFFFF else 0,
                min(num_entries, 0xFFFF),
                min(num_entries, 0xFFFF),
                min(cd_size, 0xFFFFFFFF),
                min(cd_offset, 0xFFFFFFFF),
                0,
            ))

    def package_snapshot(
        self,
        epoch: int,
        dest: Path,
        tick: Optional[int] = None,
        compression: str = "zip",
    ) -> Path:
        """Package state files into a snapshot archive.

        Compresses files in parallel using a thread pool (zlib releases
        the GIL), then assembles the final ZIP from pre-compressed data.

        Args:
            epoch: Current epoch number.
            dest: Directory where the archive will be written.
            tick: If provided, uses tick-specific naming
                  (``ep{epoch}-t{tick}-snap.zip``).  Otherwise falls
                  back to the legacy name ``ep{epoch}-full.zip``.
            compression: Only ``"zip"`` is supported.
        """
        files = self.list_snapshot_files(epoch)
        if not files:
            raise FileNotFoundError(
                f"No state files found for epoch {epoch}"
            )

        if tick is not None:
            archive_name = f"ep{epoch}-t{tick}-snap.zip"
        else:
            archive_name = f"ep{epoch}-full.zip"

        if compression != "zip":
            raise ValueError(f"Unsupported compression: {compression}")

        archive_path = dest / archive_name
        workers = min(len(files), os.cpu_count() or 4)
        logger.info(
            f"Compressing {len(files)} files with {workers} workers"
        )

        tmp_dir = tempfile.mkdtemp(prefix="qsnap_")
        try:
            # Phase 1: Compress files in parallel
            futures = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for f in files:
                    arcname = str(f.relative_to(self._data_dir))
                    future = pool.submit(
                        _compress_file_worker, str(f), arcname, tmp_dir
                    )
                    futures[future] = arcname

                entries = []
                done_count = 0
                for future in as_completed(futures):
                    done_count += 1
                    arcname = futures[future]
                    try:
                        entry = future.result()
                        entries.append(entry)
                        logger.debug(
                            f"Compressed [{done_count}/{len(files)}] "
                            f"{arcname} "
                            f"({entry['file_size']} -> {entry['compressed_size']})"
                        )
                    except Exception as exc:
                        logger.error(
                            f"Failed to compress {arcname}: {exc}"
                        )
                        raise

            # Preserve original file order in the archive
            order = {
                str(f.relative_to(self._data_dir)): i
                for i, f in enumerate(files)
            }
            entries.sort(key=lambda e: order.get(e["arcname"], 0))

            # Phase 2: Assemble ZIP from pre-compressed data
            self._build_zip_from_entries(entries, archive_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        size = archive_path.stat().st_size
        logger.info(
            f"Packaged snapshot: {archive_path} "
            f"({size} bytes, {len(files)} files, {workers} workers)"
        )
        return archive_path

    def cleanup_old_epochs(self, current_epoch: int, keep: int = 0) -> None:
        """Remove state files, snapshot dirs, and page dirs from old epochs."""
        for path in self._data_dir.iterdir():
            if not path.is_file():
                continue
            ext = path.suffix.lstrip(".")
            try:
                file_epoch = int(ext)
            except ValueError:
                continue
            if file_epoch < current_epoch - keep:
                logger.info(f"Cleaning up old file: {path.name}")
                path.unlink()

        # Clean up old ep directories
        for d in self._data_dir.iterdir():
            if d.is_dir() and d.name.startswith("ep"):
                try:
                    dir_epoch = int(d.name[2:])
                    if dir_epoch < current_epoch - keep:
                        logger.info(f"Cleaning up old snapshot dir: {d.name}")
                        shutil.rmtree(d)
                except ValueError:
                    continue

        # Clean up old page file directories (.pg swap files)
        for d in self._data_dir.iterdir():
            if not d.is_dir() or d.name.startswith("ep"):
                continue
            if not any(d.glob("*.pg")):
                continue
            # Extract trailing epoch number (e.g. "td00data198" -> 198)
            match = re.search(r"(\d+)$", d.name)
            if match:
                dir_epoch = int(match.group(1))
                if dir_epoch < current_epoch - keep:
                    logger.info(f"Cleaning up old page dir: {d.name}")
                    shutil.rmtree(d)

    @staticmethod
    def compute_checksum(file_path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
