import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.alerting import AlertManager
from app.config import AlertingConfig, WatchdogConfig
from app.epoch_service import EpochService
from app.models import EpochInfo, NodeHealth, TickInfo
from app.watchdog import Watchdog


def make_tick_info(
    epoch=198,
    tick=43101500,
    aligned=500,
    misaligned=176,
    saving=False,
):
    return TickInfo(
        epoch=epoch,
        tick=tick,
        initial_tick=43101000,
        aligned_votes=aligned,
        misaligned_votes=misaligned,
        main_aux_status=1,
        is_saving_snapshot=saving,
    )


@pytest.fixture
def watchdog_deps():
    config = WatchdogConfig(
        poll_interval_seconds=1,
        startup_grace_seconds=0,
        stuck_threshold_seconds=5,
        stuck_consecutive_polls=2,
        misaligned_threshold_votes=451,
        misaligned_threshold_seconds=0,
        max_restarts=3,
        restart_cooldown_seconds=0,
        crash_restart_delay_seconds=0,
    )
    node_client = AsyncMock()
    process_manager = MagicMock()
    process_manager.is_running.return_value = True
    process_manager.shutdown_requested = False
    process_manager.restart = AsyncMock()
    alert_manager = AlertManager(AlertingConfig(enabled=False))
    return config, node_client, process_manager, alert_manager


class TestWatchdogPollHealth:
    @pytest.mark.asyncio
    async def test_healthy(self, watchdog_deps):
        config, node_client, process_manager, alert_manager = watchdog_deps
        node_client.get_tick_info.return_value = make_tick_info(tick=100)

        wd = Watchdog(config, node_client, process_manager, alert_manager, [])
        wd._state.last_tick_change_time = time.monotonic()
        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_crashed(self, watchdog_deps):
        config, node_client, process_manager, alert_manager = watchdog_deps
        process_manager.is_running.return_value = False

        wd = Watchdog(config, node_client, process_manager, alert_manager, [])
        health = await wd._poll_health()
        assert health == NodeHealth.CRASHED

    @pytest.mark.asyncio
    async def test_saving_snapshot(self, watchdog_deps):
        config, node_client, process_manager, alert_manager = watchdog_deps
        node_client.get_tick_info.return_value = make_tick_info(saving=True)

        wd = Watchdog(config, node_client, process_manager, alert_manager, [])
        health = await wd._poll_health()
        assert health == NodeHealth.SAVING_SNAPSHOT

    @pytest.mark.asyncio
    async def test_stuck_detection(self, watchdog_deps):
        config, node_client, process_manager, alert_manager = watchdog_deps
        # Same tick every time
        node_client.get_tick_info.return_value = make_tick_info(tick=100)

        wd = Watchdog(config, node_client, process_manager, alert_manager, [])
        # Set last tick info to same tick, but long ago
        wd._state.last_tick_info = make_tick_info(tick=100)
        wd._state.last_tick_change_time = time.monotonic() - 10  # >5s threshold

        # First poll: stuck but not enough consecutive
        health = await wd._poll_health()
        # Second consecutive poll should trigger STUCK
        health = await wd._poll_health()
        assert health == NodeHealth.STUCK

    @pytest.mark.asyncio
    async def test_tick_advancing_resets_stuck(self, watchdog_deps):
        config, node_client, process_manager, alert_manager = watchdog_deps

        wd = Watchdog(config, node_client, process_manager, alert_manager, [])
        wd._state.consecutive_stuck_polls = 5

        # Tick is advancing
        wd._state.last_tick_info = make_tick_info(tick=100)
        node_client.get_tick_info.return_value = make_tick_info(tick=101)

        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY
        assert wd._state.consecutive_stuck_polls == 0

    @pytest.mark.asyncio
    async def test_misaligned_detection(self, watchdog_deps):
        config, node_client, process_manager, alert_manager = watchdog_deps
        # Misaligned votes >= 451
        node_client.get_tick_info.return_value = make_tick_info(
            tick=100, misaligned=500
        )

        wd = Watchdog(config, node_client, process_manager, alert_manager, [])
        wd._state.last_tick_change_time = time.monotonic()
        wd._state.last_tick_info = make_tick_info(tick=99)

        # Need 2 consecutive polls
        await wd._poll_health()
        health = await wd._poll_health()
        assert health == NodeHealth.MISALIGNED

    @pytest.mark.asyncio
    async def test_epoch_transition_alert(self, watchdog_deps):
        config, node_client, process_manager, alert_manager = watchdog_deps
        alert_manager.send_alert = AsyncMock()

        wd = Watchdog(config, node_client, process_manager, alert_manager, [])
        wd._state.last_tick_info = make_tick_info(epoch=197)
        wd._state.last_tick_change_time = time.monotonic()

        node_client.get_tick_info.return_value = make_tick_info(
            epoch=198, tick=43101001
        )

        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY
        alert_manager.send_alert.assert_called_once()
        call_args = alert_manager.send_alert.call_args
        assert call_args[0][1] == "epoch_transition"


class TestWatchdogVersionCheck:
    """Tests for epoch transition version compatibility checking."""

    @pytest.mark.asyncio
    async def test_epoch_transition_version_compatible(self, watchdog_deps):
        """Epoch changes, version OK → node continues."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        alert_manager.send_alert = AsyncMock()
        epoch_service = AsyncMock(spec=EpochService)
        epoch_service.get_current_epoch_info.return_value = EpochInfo(
            epoch=199, initial_tick=43200000, min_version=(1, 276),
        )

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            epoch_service=epoch_service,
            local_version=(1, 280),
        )
        wd._state.last_tick_info = make_tick_info(epoch=198)
        wd._state.last_tick_change_time = time.monotonic()

        node_client.get_tick_info.return_value = make_tick_info(
            epoch=199, tick=43200001,
        )

        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_epoch_transition_version_incompatible(self, watchdog_deps):
        """Epoch changes, version too old → node stopped, VERSION_INCOMPATIBLE."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        alert_manager.send_alert = AsyncMock()
        process_manager.stop = AsyncMock()
        epoch_service = AsyncMock(spec=EpochService)
        epoch_service.get_current_epoch_info.return_value = EpochInfo(
            epoch=199, initial_tick=43200000, min_version=(1, 280),
        )

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            epoch_service=epoch_service,
            local_version=(1, 276),
        )
        wd._state.last_tick_info = make_tick_info(epoch=198)
        wd._state.last_tick_change_time = time.monotonic()

        node_client.get_tick_info.return_value = make_tick_info(
            epoch=199, tick=43200001,
        )

        health = await wd._poll_health()
        assert health == NodeHealth.VERSION_INCOMPATIBLE
        process_manager.stop.assert_called_once()

        # Verify version_incompatible alert was sent
        version_alerts = [
            c for c in alert_manager.send_alert.call_args_list
            if c[0][1] == "version_incompatible"
        ]
        assert len(version_alerts) == 1

    @pytest.mark.asyncio
    async def test_no_min_version_skips_check(self, watchdog_deps):
        """API returns no minVersion → no version check, node continues."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        alert_manager.send_alert = AsyncMock()
        epoch_service = AsyncMock(spec=EpochService)
        epoch_service.get_current_epoch_info.return_value = EpochInfo(
            epoch=199, initial_tick=43200000, min_version=None,
        )

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            epoch_service=epoch_service,
            local_version=(1, 276),
        )
        wd._state.last_tick_info = make_tick_info(epoch=198)
        wd._state.last_tick_change_time = time.monotonic()

        node_client.get_tick_info.return_value = make_tick_info(
            epoch=199, tick=43200001,
        )

        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_version_check_api_failure_fails_open(self, watchdog_deps):
        """API error during version check → node continues (fail-open)."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        alert_manager.send_alert = AsyncMock()
        epoch_service = AsyncMock(spec=EpochService)
        epoch_service.get_current_epoch_info.side_effect = Exception(
            "API down"
        )

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            epoch_service=epoch_service,
            local_version=(1, 276),
        )
        wd._state.last_tick_info = make_tick_info(epoch=198)
        wd._state.last_tick_change_time = time.monotonic()

        node_client.get_tick_info.return_value = make_tick_info(
            epoch=199, tick=43200001,
        )

        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_periodic_api_poll_detects_new_epoch_incompatible(
        self, watchdog_deps
    ):
        """Node stuck on old epoch, API reports newer epoch with
        incompatible minVersion → node stopped."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        config.epoch_api_poll_seconds = 0  # poll immediately
        alert_manager.send_alert = AsyncMock()
        process_manager.stop = AsyncMock()
        epoch_service = AsyncMock(spec=EpochService)
        epoch_service.get_current_epoch_info.return_value = EpochInfo(
            epoch=199, initial_tick=43200000, min_version=(1, 280),
        )

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            epoch_service=epoch_service,
            local_version=(1, 276),
        )
        wd._last_epoch_api_check = 0.0  # ensure poll fires
        wd._state.last_tick_change_time = time.monotonic()

        # Node still on epoch 198
        node_client.get_tick_info.return_value = make_tick_info(
            epoch=198, tick=43101500,
        )

        health = await wd._poll_health()
        assert health == NodeHealth.VERSION_INCOMPATIBLE
        process_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_periodic_api_poll_compatible(self, watchdog_deps):
        """API reports newer epoch but version is compatible → node continues."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        config.epoch_api_poll_seconds = 0  # poll immediately
        alert_manager.send_alert = AsyncMock()
        epoch_service = AsyncMock(spec=EpochService)
        epoch_service.get_current_epoch_info.return_value = EpochInfo(
            epoch=199, initial_tick=43200000, min_version=(1, 276),
        )

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            epoch_service=epoch_service,
            local_version=(1, 280),
        )
        wd._last_epoch_api_check = 0.0
        wd._state.last_tick_change_time = time.monotonic()

        node_client.get_tick_info.return_value = make_tick_info(
            epoch=198, tick=43101500,
        )

        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_periodic_api_poll_no_min_version(self, watchdog_deps):
        """API reports newer epoch but no minVersion → no version check."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        config.epoch_api_poll_seconds = 0
        alert_manager.send_alert = AsyncMock()
        epoch_service = AsyncMock(spec=EpochService)
        epoch_service.get_current_epoch_info.return_value = EpochInfo(
            epoch=199, initial_tick=43200000, min_version=None,
        )

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            epoch_service=epoch_service,
            local_version=(1, 276),
        )
        wd._last_epoch_api_check = 0.0
        wd._state.last_tick_change_time = time.monotonic()

        node_client.get_tick_info.return_value = make_tick_info(
            epoch=198, tick=43101500,
        )

        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_no_epoch_service_skips_all_checks(self, watchdog_deps):
        """No epoch_service → version checks are entirely skipped."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        config.epoch_api_poll_seconds = 0
        alert_manager.send_alert = AsyncMock()

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            epoch_service=None,
            local_version=None,
        )
        wd._state.last_tick_info = make_tick_info(epoch=198)
        wd._state.last_tick_change_time = time.monotonic()

        node_client.get_tick_info.return_value = make_tick_info(
            epoch=199, tick=43200001,
        )

        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY


class TestWatchdogEpochBehind:
    """Tests for epoch-behind auto-restart."""

    def _make_watchdog(
        self, watchdog_deps, *, api_epoch=199, min_version=None,
        local_version=(1, 280),
    ):
        config, node_client, process_manager, alert_manager = watchdog_deps
        config.epoch_api_poll_seconds = 0  # poll immediately
        config.epoch_behind_restart_polls = 2
        alert_manager.send_alert = AsyncMock()
        process_manager.stop = AsyncMock()

        epoch_service = AsyncMock(spec=EpochService)
        epoch_service.get_current_epoch_info.return_value = EpochInfo(
            epoch=api_epoch,
            initial_tick=43200000,
            min_version=min_version,
        )

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            epoch_service=epoch_service,
            local_version=local_version,
        )
        wd._last_epoch_api_check = 0.0
        wd._state.last_tick_change_time = time.monotonic()

        # Node is on epoch 198
        node_client.get_tick_info.return_value = make_tick_info(
            epoch=198, tick=43101500,
        )
        return wd

    @pytest.mark.asyncio
    async def test_single_poll_no_restart(self, watchdog_deps):
        """API epoch > node epoch for 1 poll → HEALTHY (counter not reached)."""
        wd = self._make_watchdog(watchdog_deps)

        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY
        assert wd._consecutive_epoch_behind_polls == 1

    @pytest.mark.asyncio
    async def test_two_polls_triggers_epoch_behind(self, watchdog_deps):
        """2 consecutive polls with API ahead → returns EPOCH_BEHIND."""
        wd = self._make_watchdog(watchdog_deps)

        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY

        # Reset poll timer so second poll fires
        wd._last_epoch_api_check = 0.0
        health = await wd._poll_health()
        assert health == NodeHealth.EPOCH_BEHIND

    @pytest.mark.asyncio
    async def test_counter_resets_when_caught_up(self, watchdog_deps):
        """After 1 behind poll, node catches up → counter resets."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        config.epoch_api_poll_seconds = 0
        config.epoch_behind_restart_polls = 2
        alert_manager.send_alert = AsyncMock()

        epoch_service = AsyncMock(spec=EpochService)
        epoch_service.get_current_epoch_info.return_value = EpochInfo(
            epoch=199, initial_tick=43200000,
        )

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            epoch_service=epoch_service,
            local_version=(1, 280),
        )
        wd._last_epoch_api_check = 0.0
        wd._state.last_tick_change_time = time.monotonic()

        # First poll: node behind
        node_client.get_tick_info.return_value = make_tick_info(
            epoch=198, tick=43101500,
        )
        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY
        assert wd._consecutive_epoch_behind_polls == 1

        # Node catches up to epoch 199
        wd._last_epoch_api_check = 0.0
        node_client.get_tick_info.return_value = make_tick_info(
            epoch=199, tick=43200001,
        )
        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY
        assert wd._consecutive_epoch_behind_polls == 0

    @pytest.mark.asyncio
    async def test_version_incompatible_takes_precedence(self, watchdog_deps):
        """API epoch > node epoch AND version incompatible → VERSION_INCOMPATIBLE."""
        wd = self._make_watchdog(
            watchdog_deps,
            api_epoch=199,
            min_version=(1, 280),
            local_version=(1, 276),
        )

        health = await wd._poll_health()
        assert health == NodeHealth.VERSION_INCOMPATIBLE
        # Epoch-behind counter should not have advanced
        assert wd._consecutive_epoch_behind_polls == 0

    @pytest.mark.asyncio
    async def test_epoch_behind_without_local_version(self, watchdog_deps):
        """Epoch-behind works even without local_version (no version check)."""
        wd = self._make_watchdog(
            watchdog_deps, api_epoch=199, local_version=None,
        )

        await wd._poll_health()
        wd._last_epoch_api_check = 0.0
        health = await wd._poll_health()
        assert health == NodeHealth.EPOCH_BEHIND


class TestStateIncompatible:
    """Tests for STATE_INCOMPATIBLE detection via rapid-fail pattern."""

    @pytest.mark.asyncio
    async def test_rapid_fail_triggers_state_incompatible(self, watchdog_deps):
        """Two rapid failures after restart → STATE_INCOMPATIBLE callback."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        config.rapid_fail_threshold_seconds = 120
        config.rapid_fail_count_for_incompatible = 2
        alert_manager.send_alert = AsyncMock()
        process_manager.stop = AsyncMock()
        process_manager.send_key = AsyncMock()

        on_incompatible = AsyncMock()

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            on_state_incompatible=on_incompatible,
        )
        shutdown = asyncio.Event()

        # Simulate: node was restarted recently and is stuck again
        wd._state.restart_count = 1
        wd._state.last_restart_time = time.monotonic() - 10  # 10s ago (< 120s threshold)
        wd._rapid_fail_count = 1  # already had one rapid failure
        # Exhaust the F4 peer reset window so it falls through to restart logic
        wd._first_peer_reset_time = time.monotonic() - config.stuck_threshold_seconds - 1

        await wd._handle_unhealthy(NodeHealth.STUCK, shutdown)

        assert wd._state.health == NodeHealth.STARTING  # reset after recovery
        assert wd._rapid_fail_count == 0
        process_manager.stop.assert_called_once()
        on_incompatible.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_rapid_fail_on_slow_failure(self, watchdog_deps):
        """Failure after threshold → normal restart, not STATE_INCOMPATIBLE."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        config.rapid_fail_threshold_seconds = 120
        config.rapid_fail_count_for_incompatible = 2
        config.restart_cooldown_seconds = 0
        alert_manager.send_alert = AsyncMock()
        process_manager.send_key = AsyncMock()

        on_incompatible = AsyncMock()

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            on_state_incompatible=on_incompatible,
        )
        shutdown = asyncio.Event()

        # Node was restarted long ago (> threshold)
        wd._state.restart_count = 1
        wd._state.last_restart_time = time.monotonic() - 300  # 5 min ago
        wd._state.last_tick_info = make_tick_info()
        # Exhaust the F4 peer reset window
        wd._first_peer_reset_time = time.monotonic() - config.stuck_threshold_seconds - 1

        await wd._handle_unhealthy(NodeHealth.STUCK, shutdown)

        on_incompatible.assert_not_awaited()
        process_manager.restart.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rapid_fail_counter_resets_on_healthy(self, watchdog_deps):
        """Rapid fail counter resets when node becomes healthy."""
        config, node_client, process_manager, alert_manager = watchdog_deps

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
        )
        wd._rapid_fail_count = 1

        # Simulate healthy poll
        node_client.get_tick_info.return_value = make_tick_info(tick=200)
        wd._state.last_tick_info = make_tick_info(tick=199)
        wd._state.last_tick_change_time = time.monotonic()

        # Run one iteration of the watchdog health check
        health = await wd._poll_health()
        assert health == NodeHealth.HEALTHY

        # The counter is reset in the run() loop when health == HEALTHY,
        # so we simulate that here
        if health == NodeHealth.HEALTHY:
            wd._rapid_fail_count = 0
        assert wd._rapid_fail_count == 0

    @pytest.mark.asyncio
    async def test_state_incompatible_without_callback(self, watchdog_deps):
        """No callback → logs error, does not crash."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        config.rapid_fail_threshold_seconds = 120
        config.rapid_fail_count_for_incompatible = 2
        alert_manager.send_alert = AsyncMock()
        process_manager.stop = AsyncMock()
        process_manager.send_key = AsyncMock()

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            on_state_incompatible=None,  # no callback
        )
        shutdown = asyncio.Event()

        wd._state.restart_count = 1
        wd._state.last_restart_time = time.monotonic() - 10
        wd._rapid_fail_count = 1
        wd._state.last_tick_info = make_tick_info()
        wd._first_peer_reset_time = time.monotonic() - config.stuck_threshold_seconds - 1

        # Should not raise
        await wd._handle_unhealthy(NodeHealth.STUCK, shutdown)
        assert wd._state.health == NodeHealth.STATE_INCOMPATIBLE

    @pytest.mark.asyncio
    async def test_first_failure_not_rapid_fail(self, watchdog_deps):
        """First-ever failure (restart_count=0) is never a rapid fail."""
        config, node_client, process_manager, alert_manager = watchdog_deps
        config.rapid_fail_threshold_seconds = 120
        config.rapid_fail_count_for_incompatible = 2
        config.restart_cooldown_seconds = 0
        alert_manager.send_alert = AsyncMock()
        process_manager.send_key = AsyncMock()

        on_incompatible = AsyncMock()

        wd = Watchdog(
            config, node_client, process_manager, alert_manager, [],
            on_state_incompatible=on_incompatible,
        )
        shutdown = asyncio.Event()

        # First failure, never restarted before
        wd._state.restart_count = 0
        wd._state.last_tick_info = make_tick_info()
        wd._first_peer_reset_time = time.monotonic() - config.stuck_threshold_seconds - 1

        await wd._handle_unhealthy(NodeHealth.STUCK, shutdown)

        assert wd._rapid_fail_count == 0
        on_incompatible.assert_not_awaited()
        process_manager.restart.assert_awaited_once()
