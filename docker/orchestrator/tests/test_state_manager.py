import zipfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.downloaders.base import BaseDownloader
from app.state_manager import StateManager


class MockDownloader(BaseDownloader):
    async def download(self, url: str, dest_path: Path) -> Path:
        # Create a mock zip file
        with zipfile.ZipFile(dest_path, "w") as zf:
            zf.writestr("spectrum.000", "spectrum data")
            zf.writestr("universe.000", "universe data")
            zf.writestr("contract0000.000", "contract data")
        return dest_path

    async def close(self) -> None:
        pass


class TestStateManager:
    def test_get_local_epoch_none(self, tmp_data_dir):
        sm = StateManager(tmp_data_dir, MockDownloader())
        assert sm.get_local_epoch() is None

    def test_get_local_epoch(self, tmp_data_dir):
        (tmp_data_dir / "spectrum.198").write_text("data")
        sm = StateManager(tmp_data_dir, MockDownloader())
        assert sm.get_local_epoch() == 198

    def test_get_local_epoch_multiple(self, tmp_data_dir):
        (tmp_data_dir / "spectrum.197").write_text("old")
        (tmp_data_dir / "spectrum.198").write_text("new")
        sm = StateManager(tmp_data_dir, MockDownloader())
        assert sm.get_local_epoch() == 198

    def test_has_valid_state_files_true(self, tmp_data_dir):
        (tmp_data_dir / "spectrum.198").write_text("data")
        (tmp_data_dir / "universe.198").write_text("data")
        sm = StateManager(tmp_data_dir, MockDownloader())
        assert sm.has_valid_state_files(198) is True

    def test_has_valid_state_files_missing(self, tmp_data_dir):
        (tmp_data_dir / "spectrum.198").write_text("data")
        sm = StateManager(tmp_data_dir, MockDownloader())
        assert sm.has_valid_state_files(198) is False

    def test_has_valid_state_files_empty(self, tmp_data_dir):
        (tmp_data_dir / "spectrum.198").write_text("")
        (tmp_data_dir / "universe.198").write_text("data")
        sm = StateManager(tmp_data_dir, MockDownloader())
        assert sm.has_valid_state_files(198) is False

    @pytest.mark.asyncio
    async def test_download_epoch_files(self, tmp_data_dir):
        sm = StateManager(tmp_data_dir, MockDownloader())
        result = await sm.download_epoch_files(
            198, "https://example.com/ep198-full.zip"
        )
        assert result is True
        assert (tmp_data_dir / "spectrum.198").exists()
        assert (tmp_data_dir / "universe.198").exists()
        assert (tmp_data_dir / "contract0000.198").exists()

    def test_list_state_files(self, tmp_data_dir):
        (tmp_data_dir / "spectrum.198").write_text("data")
        (tmp_data_dir / "universe.198").write_text("data")
        (tmp_data_dir / "contract0000.198").write_text("data")
        (tmp_data_dir / "system").write_text("data")
        sm = StateManager(tmp_data_dir, MockDownloader())
        files = sm.list_state_files(198)
        names = [f.name for f in files]
        assert "spectrum.198" in names
        assert "universe.198" in names
        assert "contract0000.198" in names
        assert "system" in names

    def test_cleanup_old_epochs(self, tmp_data_dir):
        """Only current epoch files/dirs are kept; all others removed."""
        (tmp_data_dir / "spectrum.196").write_text("old")
        (tmp_data_dir / "spectrum.197").write_text("prev")
        (tmp_data_dir / "spectrum.198").write_text("current")
        (tmp_data_dir / "system").write_text("no epoch ext")

        ep196 = tmp_data_dir / "ep196"
        ep196.mkdir()
        (ep196 / "system.snp").write_text("old snp")

        ep197 = tmp_data_dir / "ep197"
        ep197.mkdir()
        (ep197 / "system.snp").write_text("prev snp")

        ep198 = tmp_data_dir / "ep198"
        ep198.mkdir()
        (ep198 / "system.snp").write_text("cur snp")

        sm = StateManager(tmp_data_dir, MockDownloader())
        sm.cleanup_old_epochs(198)

        # Only current epoch survives
        assert not (tmp_data_dir / "spectrum.196").exists()
        assert not (tmp_data_dir / "spectrum.197").exists()
        assert (tmp_data_dir / "spectrum.198").exists()
        # Non-epoch files are untouched
        assert (tmp_data_dir / "system").exists()
        # Snapshot dirs
        assert not ep196.exists()
        assert not ep197.exists()
        assert ep198.exists()

    def test_package_snapshot_legacy_name(self, tmp_data_dir):
        (tmp_data_dir / "spectrum.198").write_text("data")
        (tmp_data_dir / "universe.198").write_text("data")
        (tmp_data_dir / "system").write_text("system")
        ep_dir = tmp_data_dir / "ep198"
        ep_dir.mkdir()
        (ep_dir / "system.snp").write_text("snp")

        sm = StateManager(tmp_data_dir, MockDownloader())
        dest = tmp_data_dir / "out"
        dest.mkdir()
        archive = sm.package_snapshot(198, dest)
        assert archive.name == "ep198-full.zip"
        assert archive.exists()

    def test_package_snapshot_tick_name(self, tmp_data_dir):
        (tmp_data_dir / "spectrum.198").write_text("data")
        (tmp_data_dir / "universe.198").write_text("data")
        (tmp_data_dir / "system").write_text("system")
        ep_dir = tmp_data_dir / "ep198"
        ep_dir.mkdir()
        (ep_dir / "system.snp").write_text("snp")

        sm = StateManager(tmp_data_dir, MockDownloader())
        dest = tmp_data_dir / "out"
        dest.mkdir()
        archive = sm.package_snapshot(198, dest, tick=43101500)
        assert archive.name == "ep198-t43101500-snap.zip"
        assert archive.exists()

        # Verify it's a valid zip
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            assert any("spectrum.198" in n for n in names)

    def test_package_snapshot_parallel_zip_valid(self, tmp_data_dir):
        """Verify parallel ZIP is valid and extractable with correct content."""
        # Create multiple files of varying sizes to exercise parallelism
        (tmp_data_dir / "spectrum.198").write_bytes(b"A" * 100_000)
        (tmp_data_dir / "universe.198").write_bytes(b"B" * 200_000)
        (tmp_data_dir / "contract0000.198").write_bytes(b"C" * 50_000)
        (tmp_data_dir / "system").write_bytes(b"D" * 10_000)
        ep_dir = tmp_data_dir / "ep198"
        ep_dir.mkdir()
        (ep_dir / "system.snp").write_bytes(b"E" * 5_000)

        sm = StateManager(tmp_data_dir, MockDownloader())
        dest = tmp_data_dir / "out"
        dest.mkdir()
        archive = sm.package_snapshot(198, dest, tick=99999)
        assert archive.exists()

        # Verify zipfile can open and validate CRCs
        with zipfile.ZipFile(archive, "r") as zf:
            bad = zf.testzip()
            assert bad is None, f"CRC mismatch in {bad}"

            names = zf.namelist()
            assert "spectrum.198" in names
            assert "universe.198" in names
            assert "contract0000.198" in names
            assert "system" in names

            # Check actual content
            assert zf.read("spectrum.198") == b"A" * 100_000
            assert zf.read("universe.198") == b"B" * 200_000
            assert zf.read("contract0000.198") == b"C" * 50_000
            assert zf.read("system") == b"D" * 10_000

        # Verify it can be extracted to a fresh directory
        extract_dir = tmp_data_dir / "extracted"
        extract_dir.mkdir()
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(extract_dir)
        assert (extract_dir / "spectrum.198").read_bytes() == b"A" * 100_000
        assert (extract_dir / "universe.198").read_bytes() == b"B" * 200_000

    def test_list_snapshot_files_includes_pg(self, tmp_data_dir):
        """Page files (.pg) in subdirectories are included in snapshot."""
        (tmp_data_dir / "spectrum.199").write_text("data")
        (tmp_data_dir / "universe.199").write_text("data")
        ep_dir = tmp_data_dir / "ep199"
        ep_dir.mkdir()
        (ep_dir / "system.snp").write_text("snp")

        # Create page file directories for current epoch
        td_dir = tmp_data_dir / "td00data199"
        td_dir.mkdir()
        (td_dir / "abcdef12.pg").write_bytes(b"\x00" * 100)
        (td_dir / "abcdef34.pg").write_bytes(b"\x00" * 100)

        tick_dir = tmp_data_dir / "tickdata199"
        tick_dir.mkdir()
        (tick_dir / "ghijkl56.pg").write_bytes(b"\x00" * 100)

        # Create page file directory from a different epoch (should be excluded)
        old_dir = tmp_data_dir / "td00data198"
        old_dir.mkdir()
        (old_dir / "oldpage.pg").write_bytes(b"\x00" * 100)

        sm = StateManager(tmp_data_dir, MockDownloader())
        files = sm.list_snapshot_files(199)
        names = [str(f.relative_to(tmp_data_dir)) for f in files]
        assert "td00data199/abcdef12.pg" in names
        assert "td00data199/abcdef34.pg" in names
        assert "tickdata199/ghijkl56.pg" in names
        assert "td00data198/oldpage.pg" not in names

    def test_list_snapshot_files_skips_non_pg_subdirs(self, tmp_data_dir):
        """Subdirectories without .pg files are not included."""
        (tmp_data_dir / "spectrum.199").write_text("data")
        (tmp_data_dir / "universe.199").write_text("data")

        other_dir = tmp_data_dir / "somedir"
        other_dir.mkdir()
        (other_dir / "random.txt").write_text("not a page file")

        sm = StateManager(tmp_data_dir, MockDownloader())
        files = sm.list_snapshot_files(199)
        names = [f.name for f in files]
        assert "random.txt" not in names

    def test_cleanup_old_pg_dirs(self, tmp_data_dir):
        """Page dirs from non-current epochs are removed (default keep=0)."""
        (tmp_data_dir / "spectrum.199").write_text("current")

        old_dir = tmp_data_dir / "td00data197"
        old_dir.mkdir()
        (old_dir / "page1.pg").write_bytes(b"\x00")

        cur_dir = tmp_data_dir / "td00data199"
        cur_dir.mkdir()
        (cur_dir / "page2.pg").write_bytes(b"\x00")

        prev_dir = tmp_data_dir / "tickdata198"
        prev_dir.mkdir()
        (prev_dir / "page3.pg").write_bytes(b"\x00")

        sm = StateManager(tmp_data_dir, MockDownloader())
        sm.cleanup_old_epochs(199)

        assert not old_dir.exists()
        assert cur_dir.exists()
        assert not prev_dir.exists()

    def test_cleanup_keep_previous_epoch(self, tmp_data_dir):
        """With keep=1, previous epoch data is preserved."""
        (tmp_data_dir / "spectrum.197").write_text("old")
        (tmp_data_dir / "spectrum.198").write_text("prev")
        (tmp_data_dir / "spectrum.199").write_text("current")

        prev_pg = tmp_data_dir / "td00data198"
        prev_pg.mkdir()
        (prev_pg / "page.pg").write_bytes(b"\x00")

        old_pg = tmp_data_dir / "td00data197"
        old_pg.mkdir()
        (old_pg / "page.pg").write_bytes(b"\x00")

        sm = StateManager(tmp_data_dir, MockDownloader())
        sm.cleanup_old_epochs(199, keep=1)

        assert not (tmp_data_dir / "spectrum.197").exists()
        assert (tmp_data_dir / "spectrum.198").exists()
        assert (tmp_data_dir / "spectrum.199").exists()
        assert not old_pg.exists()
        assert prev_pg.exists()

    def test_package_snapshot_includes_pg_files(self, tmp_data_dir):
        """Packaged snapshot archive contains .pg files with correct paths."""
        (tmp_data_dir / "spectrum.199").write_bytes(b"S" * 1000)
        (tmp_data_dir / "universe.199").write_bytes(b"U" * 1000)
        (tmp_data_dir / "system").write_bytes(b"Y" * 100)
        ep_dir = tmp_data_dir / "ep199"
        ep_dir.mkdir()
        (ep_dir / "system.snp").write_bytes(b"N" * 100)

        pg_dir = tmp_data_dir / "td00data199"
        pg_dir.mkdir()
        (pg_dir / "testpage.pg").write_bytes(b"P" * 500)

        sm = StateManager(tmp_data_dir, MockDownloader())
        dest = tmp_data_dir / "out"
        dest.mkdir()
        archive = sm.package_snapshot(199, dest, tick=44000000)

        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()
            assert "td00data199/testpage.pg" in names
            assert zf.read("td00data199/testpage.pg") == b"P" * 500

    def test_compute_checksum(self, tmp_data_dir):
        test_file = tmp_data_dir / "test.bin"
        test_file.write_bytes(b"hello world")
        checksum = StateManager.compute_checksum(test_file)
        assert len(checksum) == 64  # SHA-256 hex digest
        assert checksum == (
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        )

    def test_delete_epoch_files(self, tmp_data_dir):
        """delete_epoch_files removes all state for the given epoch only."""
        # Create files for epoch 198 and 199
        (tmp_data_dir / "spectrum.198").write_text("old")
        (tmp_data_dir / "universe.198").write_text("old")
        (tmp_data_dir / "contract0001.198").write_text("old")
        (tmp_data_dir / "spectrum.199").write_text("new")
        (tmp_data_dir / "universe.199").write_text("new")
        (tmp_data_dir / "system").write_text("system")

        ep198 = tmp_data_dir / "ep198"
        ep198.mkdir()
        (ep198 / "system.snp").write_text("snp")

        ep199 = tmp_data_dir / "ep199"
        ep199.mkdir()
        (ep199 / "system.snp").write_text("snp")

        pg198 = tmp_data_dir / "td00data198"
        pg198.mkdir()
        (pg198 / "page.pg").write_bytes(b"\x00")

        pg199 = tmp_data_dir / "td00data199"
        pg199.mkdir()
        (pg199 / "page.pg").write_bytes(b"\x00")

        sm = StateManager(tmp_data_dir, MockDownloader())
        sm.delete_epoch_files(198)

        # Epoch 198 files are gone
        assert not (tmp_data_dir / "spectrum.198").exists()
        assert not (tmp_data_dir / "universe.198").exists()
        assert not (tmp_data_dir / "contract0001.198").exists()
        assert not ep198.exists()
        assert not pg198.exists()

        # Epoch 199 and non-epoch files are untouched
        assert (tmp_data_dir / "spectrum.199").exists()
        assert (tmp_data_dir / "universe.199").exists()
        assert (tmp_data_dir / "system").exists()
        assert ep199.exists()
        assert pg199.exists()
