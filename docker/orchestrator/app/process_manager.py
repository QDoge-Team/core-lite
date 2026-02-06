from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from typing import Optional

from app.node_client import NodeClient

logger = logging.getLogger(__name__)

# Terminal escape sequences for Linux function keys.
# Matches the keyMap in src/extensions/overload.h ReadKeyStroke().
_KEY_SEQUENCES: dict[str, bytes] = {
    "f2": bytes([27, 79, 81]),
    "f3": bytes([27, 79, 82]),
    "f4": bytes([27, 79, 83]),
    "f5": bytes([27, 91, 49, 53, 126]),
    "f6": bytes([27, 91, 49, 55, 126]),
    "f7": bytes([27, 91, 49, 56, 126]),
    "f8": bytes([27, 91, 49, 57, 126]),
    "f9": bytes([27, 91, 50, 48, 126]),
    "f10": bytes([27, 91, 50, 49, 126]),
    "f11": bytes([27, 91, 50, 51, 126]),
    "f12": bytes([27, 91, 50, 52, 126]),
    "p": bytes([ord("p")]),
    "f": bytes([ord("f")]),
    "s": bytes([ord("s")]),
    "esc": bytes([27]),
}

# Human-readable descriptions for each key command.
KEY_DESCRIPTIONS: dict[str, str] = {
    "f2": "Display node status",
    "f3": "Display mining race state",
    "f4": "Drop all connections (reconnect to peers)",
    "f5": "Issue new votes",
    "f6": "Save state to disk",
    "f7": "Force epoch switching",
    "f8": "Save tick storage snapshot",
    "f9": "Decrease latestCreatedTick by 1",
    "f10": "Allow epoch transition with memory cleanup",
    "f11": "Toggle network mode (static/dynamic)",
    "f12": "Switch MAIN/aux mode",
    "p": "Cycle console logging verbosity",
    "f": "Force skip computer digest check",
    "s": "Force skip security tick check",
    "esc": "Shutdown node",
}


class ProcessManager:
    """Manages the Qubic binary subprocess lifecycle."""

    def __init__(
        self,
        binary_path: Path,
        node_client: NodeClient,
        working_dir: Path,
    ) -> None:
        self._binary_path = binary_path
        self._node_client = node_client
        self._working_dir = working_dir
        self._process: Optional[asyncio.subprocess.Process] = None
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._shutdown_requested: bool = False

    async def start(self, args: list[str]) -> asyncio.subprocess.Process:
        """Start the Qubic binary with the given arguments."""
        cmd = [str(self._binary_path)] + args
        logger.info(f"Starting Qubic: {' '.join(cmd)}")
        logger.info(f"Working directory: {self._working_dir}")

        self._shutdown_requested = False  # Reset on start
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._working_dir),
        )
        logger.info(f"Qubic process started with PID {self._process.pid}")

        # Stream stdout/stderr to logging
        self._stdout_task = asyncio.create_task(
            self._stream_output(self._process.stdout, "stdout")
        )
        self._stderr_task = asyncio.create_task(
            self._stream_output(self._process.stderr, "stderr")
        )

        return self._process

    async def _stream_output(
        self,
        stream: Optional[asyncio.StreamReader],
        name: str,
    ) -> None:
        """Forward subprocess output to the logger."""
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                logger.info(f"[qubic:{name}] {text}")

    async def stop(self, timeout: float = 120.0) -> int:
        """Gracefully stop the Qubic process.

        Escalation: /shutdown endpoint → SIGTERM → SIGKILL.
        Returns the process exit code.
        """
        self._shutdown_requested = True
        if self._process is None or self._process.returncode is not None:
            return self._process.returncode if self._process else -1

        pid = self._process.pid

        # Step 1: Try graceful shutdown via HTTP API
        logger.info(f"Requesting graceful shutdown via HTTP API (PID {pid})")
        try:
            await self._node_client.shutdown()
        except Exception as e:
            logger.warning(f"HTTP shutdown request failed: {e}")

        # Wait for process to exit
        try:
            return_code = await asyncio.wait_for(
                self._process.wait(), timeout=timeout
            )
            logger.info(
                f"Qubic process exited gracefully (code {return_code})"
            )
            await self._cleanup_tasks()
            return return_code
        except asyncio.TimeoutError:
            pass

        # Step 2: Send SIGTERM
        logger.warning(
            f"Graceful shutdown timed out after {timeout}s, sending SIGTERM"
        )
        try:
            self._process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            await self._cleanup_tasks()
            return -1

        try:
            return_code = await asyncio.wait_for(
                self._process.wait(), timeout=30.0
            )
            logger.info(f"Process exited after SIGTERM (code {return_code})")
            await self._cleanup_tasks()
            return return_code
        except asyncio.TimeoutError:
            pass

        # Step 3: SIGKILL
        logger.warning("SIGTERM timed out, sending SIGKILL")
        try:
            self._process.kill()
        except ProcessLookupError:
            pass
        return_code = await self._process.wait()
        logger.info(f"Process killed (code {return_code})")
        await self._cleanup_tasks()
        return return_code

    async def _cleanup_tasks(self) -> None:
        for task in (self._stdout_task, self._stderr_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def restart(self, args: list[str]) -> asyncio.subprocess.Process:
        """Stop the current process and start a new one."""
        await self.stop()
        return await self.start(args)

    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.returncode is None
        )

    @property
    def shutdown_requested(self) -> bool:
        """True if graceful shutdown was requested (via stop() or ESC key)."""
        return self._shutdown_requested

    def get_pid(self) -> Optional[int]:
        if self._process is not None:
            return self._process.pid
        return None

    def get_return_code(self) -> Optional[int]:
        if self._process is not None:
            return self._process.returncode
        return None

    async def send_key(self, key_name: str) -> bool:
        """Inject a key press into the Qubic process via stdin.

        The Qubic binary reads terminal escape sequences from stdin
        in non-blocking mode and maps them to function key actions.

        Returns True if the key was sent, False if the process is not
        running or the key name is unknown.
        """
        key_name = key_name.lower().strip()
        seq = _KEY_SEQUENCES.get(key_name)
        if seq is None:
            logger.warning(f"Unknown key: {key_name}")
            return False
        if not self.is_running() or self._process.stdin is None:
            logger.warning("Cannot send key: process not running")
            return False
        try:
            self._process.stdin.write(seq)
            await self._process.stdin.drain()
            desc = KEY_DESCRIPTIONS.get(key_name, key_name)
            logger.info(f"Sent key {key_name.upper()} ({desc})")
            # Mark as intentional shutdown if ESC key was sent
            if key_name == "esc":
                self._shutdown_requested = True
            return True
        except (BrokenPipeError, ConnectionResetError) as e:
            logger.warning(f"Failed to send key {key_name}: {e}")
            return False

    @staticmethod
    def available_keys() -> dict[str, str]:
        """Return a dict of available key names and their descriptions."""
        return dict(KEY_DESCRIPTIONS)
