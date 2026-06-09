"""Tests for emergo.alerting and emergo.logging_config."""

from __future__ import annotations

import io
import json
import logging

import numpy as np
import pytest

from emergo import Graph, make_initial_phi
from emergo.alerting import (
    AlertEvent,
    AlertLevel,
    AlertManager,
    authority_monopoly_rule,
    convergence_stall_rule,
    high_rejection_rate_rule,
    phi_loss_spike_rule,
)
from emergo.logging_config import JsonFormatter, configure_structured_logging, get_emergo_logger
from emergo.types import Authority, CoordinationEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ce() -> CoordinationEvent:
    return CoordinationEvent(
        event_type="add_edge",
        participants=("A", "B"),
        params=frozenset([("weight", 0.5)]),
    )


def _state_with_authority(scores: dict) -> tuple:
    """Build a minimal State tuple with the given authority scores."""
    n = len(scores)
    agent_ids = tuple(scores.keys())
    adj = np.zeros((n, n))
    caps = np.ones((n, 2))
    G = Graph(agent_ids=agent_ids, adjacency=adj, capabilities=caps)
    phi = make_initial_phi(d_latent=4, d_features=8, d_ce=4, seed=0)
    A = Authority(scores=dict(scores), baseline=0.5)
    return G, phi, A, []


def _default_state() -> tuple:
    return _state_with_authority({"A": 0.5, "B": 0.5, "C": 0.5})


# ---------------------------------------------------------------------------
# TestAlertEvent
# ---------------------------------------------------------------------------


class TestAlertEvent:
    def test_frozen_dataclass(self):
        evt = AlertEvent(
            rule_name="test_rule",
            level=AlertLevel.WARNING,
            message="test message",
            iteration=42,
            details={"key": "value"},
        )
        with pytest.raises((AttributeError, TypeError)):
            evt.rule_name = "mutated"  # type: ignore[misc]

    def test_fields_accessible(self):
        evt = AlertEvent(
            rule_name="my_rule",
            level=AlertLevel.CRITICAL,
            message="something happened",
            iteration=7,
            details={"x": 1},
        )
        assert evt.rule_name == "my_rule"
        assert evt.level == AlertLevel.CRITICAL
        assert evt.message == "something happened"
        assert evt.iteration == 7
        assert evt.details == {"x": 1}


# ---------------------------------------------------------------------------
# TestAlertManager
# ---------------------------------------------------------------------------


class TestAlertManager:
    def test_no_rules_fires_nothing(self):
        fired = []
        mgr = AlertManager(rules=[], on_alert=fired.append)
        state = _default_state()
        mgr.on_iteration_start(0, state)
        assert fired == []

    def test_fires_callback_when_rule_fires(self):
        fired = []
        state = _state_with_authority({"A": 0.9})  # exceeds default threshold 0.75
        rule = authority_monopoly_rule(threshold=0.75)
        mgr = AlertManager(rules=[rule], on_alert=fired.append)
        mgr.on_iteration_start(0, state)
        assert len(fired) == 1
        assert fired[0].rule_name == rule.name
        assert fired[0].iteration == 0

    def test_cooldown_prevents_repeated_alerts(self):
        fired = []
        state = _state_with_authority({"A": 0.9})
        rule = authority_monopoly_rule(threshold=0.75)
        mgr = AlertManager(rules=[rule], on_alert=fired.append, cooldown=10)
        # Fire at t=0
        mgr.on_iteration_start(0, state)
        # Try again at t=5 — within cooldown window
        mgr.on_iteration_start(5, state)
        assert len(fired) == 1  # cooldown suppressed second alert

    def test_cooldown_allows_alert_after_window(self):
        fired = []
        state = _state_with_authority({"A": 0.9})
        rule = authority_monopoly_rule(threshold=0.75)
        mgr = AlertManager(rules=[rule], on_alert=fired.append, cooldown=10)
        mgr.on_iteration_start(0, state)
        # Now t=10 — exactly at cooldown boundary, should fire again
        mgr.on_iteration_start(10, state)
        assert len(fired) == 2

    def test_alert_history_accumulates(self):
        fired = []
        state = _state_with_authority({"A": 0.9})
        rule = authority_monopoly_rule(threshold=0.75)
        mgr = AlertManager(rules=[rule], on_alert=fired.append, cooldown=1)
        mgr.on_iteration_start(0, state)
        mgr.on_iteration_start(1, state)
        mgr.on_iteration_start(2, state)
        assert len(mgr.alert_history) == 3

    def test_reset_clears_history(self):
        fired = []
        state = _state_with_authority({"A": 0.9})
        rule = authority_monopoly_rule(threshold=0.75)
        mgr = AlertManager(rules=[rule], on_alert=fired.append, cooldown=1)
        mgr.on_iteration_start(0, state)
        assert len(mgr.alert_history) == 1
        mgr.reset()
        assert mgr.alert_history == []

    def test_alert_history_is_copy(self):
        """alert_history should return a new list each time."""
        state = _state_with_authority({"A": 0.9})
        rule = authority_monopoly_rule(threshold=0.75)
        mgr = AlertManager(rules=[rule], on_alert=lambda e: None, cooldown=1)
        mgr.on_iteration_start(0, state)
        h1 = mgr.alert_history
        h2 = mgr.alert_history
        assert h1 is not h2

    def test_on_ce_result_stored(self):
        """AlertManager stores CE result for use in next on_iteration_start."""
        accepted_flags = []

        class _TrackingRule:
            name = "tracking_rule"
            level = AlertLevel.INFO

            def check(
                self, iteration, state, *, last_ce_accepted=False, last_errors=None, phi_loss=None
            ):
                accepted_flags.append(last_ce_accepted)
                return None

        mgr = AlertManager(rules=[_TrackingRule()], on_alert=lambda e: None)
        ce = _make_ce()
        state = _default_state()
        mgr.on_ce_result(0, ce, True, None)
        mgr.on_iteration_start(1, state)
        assert accepted_flags == [True]

    def test_on_phi_updated_stored(self):
        """AlertManager stores phi_loss for use in next on_iteration_start."""
        phi_losses_seen = []

        class _TrackingRule:
            name = "phi_tracking"
            level = AlertLevel.INFO

            def check(
                self, iteration, state, *, last_ce_accepted=False, last_errors=None, phi_loss=None
            ):
                phi_losses_seen.append(phi_loss)
                return None

        mgr = AlertManager(rules=[_TrackingRule()], on_alert=lambda e: None)
        state = _default_state()
        mgr.on_phi_updated(5, 0.42)
        mgr.on_iteration_start(6, state)
        assert phi_losses_seen == [pytest.approx(0.42)]

    def test_default_callback_logs_warning(self, caplog):
        """Default on_alert logs at WARNING level."""
        state = _state_with_authority({"A": 0.9})
        rule = authority_monopoly_rule(threshold=0.75)
        mgr = AlertManager(rules=[rule])  # no on_alert → uses default
        with caplog.at_level(logging.WARNING, logger="emergo.alerting"):
            mgr.on_iteration_start(0, state)
        assert any("ALERT" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# TestAuthorityMonopolyRule
# ---------------------------------------------------------------------------


class TestAuthorityMonopolyRule:
    def test_fires_when_authority_exceeds_threshold(self):
        rule = authority_monopoly_rule(threshold=0.75)
        state = _state_with_authority({"A": 0.8, "B": 0.5})
        event = rule.check(10, state)
        assert event is not None
        assert event.iteration == 10
        assert "A" in event.message or "0.8" in event.message

    def test_no_fire_when_below_threshold(self):
        rule = authority_monopoly_rule(threshold=0.75)
        state = _state_with_authority({"A": 0.7, "B": 0.5})
        event = rule.check(10, state)
        assert event is None

    def test_no_fire_when_exactly_at_threshold(self):
        rule = authority_monopoly_rule(threshold=0.75)
        state = _state_with_authority({"A": 0.75, "B": 0.5})
        # threshold is strict > so exactly at threshold should not fire
        event = rule.check(10, state)
        assert event is None

    def test_custom_threshold(self):
        rule = authority_monopoly_rule(threshold=0.6)
        state = _state_with_authority({"A": 0.65})
        event = rule.check(5, state)
        assert event is not None

    def test_custom_level(self):
        rule = authority_monopoly_rule(threshold=0.5, level=AlertLevel.CRITICAL)
        state = _state_with_authority({"A": 0.6})
        event = rule.check(0, state)
        assert event is not None
        assert event.level == AlertLevel.CRITICAL

    def test_details_contains_violators(self):
        rule = authority_monopoly_rule(threshold=0.75)
        state = _state_with_authority({"A": 0.9, "B": 0.5})
        event = rule.check(1, state)
        assert event is not None
        assert "violators" in event.details
        assert "A" in event.details["violators"]


# ---------------------------------------------------------------------------
# TestConvergenceStallRule
# ---------------------------------------------------------------------------


class TestConvergenceStallRule:
    def test_fires_after_stall_iterations(self):
        rule = convergence_stall_rule(stall_iterations=5)
        state = _default_state()
        # Feed 5 rejections
        event = None
        for t in range(5):
            event = rule.check(t, state, last_ce_accepted=False)
        # Should fire on iteration 4 (5th call with window full)
        assert event is not None

    def test_no_fire_when_ce_accepted(self):
        rule = convergence_stall_rule(stall_iterations=5)
        state = _default_state()
        # 4 rejections then 1 acceptance
        for t in range(4):
            rule.check(t, state, last_ce_accepted=False)
        event = rule.check(4, state, last_ce_accepted=True)
        assert event is None

    def test_no_fire_before_window_full(self):
        rule = convergence_stall_rule(stall_iterations=10)
        state = _default_state()
        # Only 5 rejections — window not full
        event = None
        for t in range(5):
            event = rule.check(t, state, last_ce_accepted=False)
        assert event is None

    def test_recovers_after_acceptance(self):
        rule = convergence_stall_rule(stall_iterations=3)
        state = _default_state()
        # Fill window with rejections — fires
        for t in range(3):
            rule.check(t, state, last_ce_accepted=False)
        # After an acceptance, window includes True so should not fire
        rule.check(3, state, last_ce_accepted=True)
        event = rule.check(4, state, last_ce_accepted=False)
        assert event is None  # window has one True in it now

    def test_stall_iterations_in_details(self):
        rule = convergence_stall_rule(stall_iterations=3)
        state = _default_state()
        event = None
        for t in range(3):
            event = rule.check(t, state, last_ce_accepted=False)
        assert event is not None
        assert event.details["stall_iterations"] == 3


# ---------------------------------------------------------------------------
# TestHighRejectionRateRule
# ---------------------------------------------------------------------------


class TestHighRejectionRateRule:
    def test_fires_when_rejection_rate_high(self):
        rule = high_rejection_rate_rule(window=10, threshold=0.8)
        state = _default_state()
        # 10 rejections → rate = 1.0 > 0.8
        event = None
        for t in range(10):
            event = rule.check(t, state, last_ce_accepted=False)
        assert event is not None

    def test_no_fire_when_acceptance_normal(self):
        rule = high_rejection_rate_rule(window=10, threshold=0.8)
        state = _default_state()
        # Alternate accepted/rejected → rate = 0.5 < 0.8
        for t in range(10):
            rule.check(t, state, last_ce_accepted=(t % 2 == 0))
        event = rule.check(10, state, last_ce_accepted=False)
        assert event is None

    def test_no_fire_before_window_full(self):
        rule = high_rejection_rate_rule(window=10, threshold=0.8)
        state = _default_state()
        # Only 5 items — window not full
        event = None
        for t in range(5):
            event = rule.check(t, state, last_ce_accepted=False)
        assert event is None

    def test_rejection_rate_in_details(self):
        rule = high_rejection_rate_rule(window=5, threshold=0.8)
        state = _default_state()
        event = None
        for t in range(5):
            event = rule.check(t, state, last_ce_accepted=False)
        assert event is not None
        assert event.details["rejection_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestPhiLossSpikeRule
# ---------------------------------------------------------------------------


class TestPhiLossSpikeRule:
    def test_fires_when_loss_spikes(self):
        rule = phi_loss_spike_rule(spike_factor=5.0)
        state = _default_state()
        # Establish baseline with modest loss
        rule.check(0, state, phi_loss=0.1)
        # Spike massively
        event = rule.check(1, state, phi_loss=100.0)
        assert event is not None
        assert event.level == AlertLevel.CRITICAL

    def test_no_fire_on_first_call(self):
        rule = phi_loss_spike_rule(spike_factor=5.0)
        state = _default_state()
        # First call — no baseline yet
        event = rule.check(0, state, phi_loss=999.0)
        assert event is None

    def test_no_fire_when_phi_loss_is_none(self):
        rule = phi_loss_spike_rule(spike_factor=5.0)
        state = _default_state()
        event = rule.check(0, state, phi_loss=None)
        assert event is None

    def test_no_fire_on_moderate_change(self):
        rule = phi_loss_spike_rule(spike_factor=10.0)
        state = _default_state()
        rule.check(0, state, phi_loss=1.0)
        # 5x the baseline — below spike_factor=10
        event = rule.check(1, state, phi_loss=5.0)
        assert event is None

    def test_spike_factor_in_details(self):
        rule = phi_loss_spike_rule(spike_factor=3.0)
        state = _default_state()
        rule.check(0, state, phi_loss=1.0)
        event = rule.check(1, state, phi_loss=100.0)
        assert event is not None
        assert event.details["spike_factor"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# TestJsonFormatter
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    def _make_record(
        self,
        msg: str = "hello",
        level: int = logging.INFO,
        logger_name: str = "test.logger",
        extra: dict | None = None,
    ) -> logging.LogRecord:
        record = logging.LogRecord(
            name=logger_name,
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        if extra:
            for k, v in extra.items():
                setattr(record, k, v)
        return record

    def test_emits_valid_json(self):
        formatter = JsonFormatter()
        record = self._make_record("test message")
        output = formatter.format(record)
        parsed = json.loads(output)  # must not raise
        assert isinstance(parsed, dict)

    def test_required_fields_present(self):
        formatter = JsonFormatter()
        record = self._make_record("check fields", logger_name="emergo.kernel")
        parsed = json.loads(formatter.format(record))
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed

    def test_timestamp_format(self):
        """Timestamp should be ISO-8601 UTC ending in Z."""
        formatter = JsonFormatter()
        record = self._make_record("ts test")
        parsed = json.loads(formatter.format(record))
        ts = parsed["timestamp"]
        # Should end with Z
        assert ts.endswith("Z")
        # Should contain T separator
        assert "T" in ts

    def test_level_is_string(self):
        formatter = JsonFormatter()
        record = self._make_record("level test", level=logging.WARNING)
        parsed = json.loads(formatter.format(record))
        assert parsed["level"] == "WARNING"

    def test_extra_fields_included(self):
        formatter = JsonFormatter()
        record = self._make_record("extra test", extra={"run_id": "exp_001", "n_agents": 50})
        parsed = json.loads(formatter.format(record))
        assert parsed["run_id"] == "exp_001"
        assert parsed["n_agents"] == 50

    def test_formatter_extra_fields_baked_in(self):
        """Fields passed to JsonFormatter.__init__ appear in every record."""
        formatter = JsonFormatter(extra_fields={"service": "emergo", "env": "test"})
        record = self._make_record("baked fields")
        parsed = json.loads(formatter.format(record))
        assert parsed["service"] == "emergo"
        assert parsed["env"] == "test"

    def test_message_field_matches_msg(self):
        formatter = JsonFormatter()
        record = self._make_record("hello world")
        parsed = json.loads(formatter.format(record))
        assert parsed["message"] == "hello world"

    def test_single_line_output(self):
        """Each record should be a single line (no embedded newlines)."""
        formatter = JsonFormatter()
        record = self._make_record("single line check")
        output = formatter.format(record)
        assert "\n" not in output


# ---------------------------------------------------------------------------
# TestConfigureStructuredLogging
# ---------------------------------------------------------------------------


class TestConfigureStructuredLogging:
    def test_returns_handler(self):
        stream = io.StringIO()
        handler = configure_structured_logging(stream=stream, include_emergo_context=False)
        assert isinstance(handler, logging.Handler)

    def test_handler_uses_json_formatter(self):
        stream = io.StringIO()
        handler = configure_structured_logging(stream=stream, include_emergo_context=False)
        assert isinstance(handler.formatter, JsonFormatter)

    def test_handler_emits_json(self):
        stream = io.StringIO()
        handler = configure_structured_logging(
            level="DEBUG", stream=stream, include_emergo_context=False
        )

        # Create an isolated test logger (not the emergo hierarchy)
        test_logger = logging.getLogger("test_configure_" + str(id(stream)))
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        test_logger.propagate = False

        test_logger.info("structured test message")

        output = stream.getvalue().strip()
        assert output  # non-empty
        parsed = json.loads(output)
        assert parsed["message"] == "structured test message"

    def test_include_emergo_context_adds_handler(self):
        """With include_emergo_context=True, the emergo logger should have our handler."""
        stream = io.StringIO()
        handler = configure_structured_logging(stream=stream, include_emergo_context=True)
        emergo_logger = logging.getLogger("emergo")
        assert handler in emergo_logger.handlers
        # Cleanup
        emergo_logger.removeHandler(handler)

    def test_extra_fields_propagate_to_records(self):
        stream = io.StringIO()
        handler = configure_structured_logging(
            stream=stream,
            include_emergo_context=False,
            extra_fields={"deployment": "prod"},
        )
        test_logger = logging.getLogger("test_extra_fields_" + str(id(stream)))
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        test_logger.propagate = False

        test_logger.info("with extra")
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["deployment"] == "prod"

    def test_no_duplicate_handlers_on_repeated_calls(self):
        """Calling configure_structured_logging twice with same stream shouldn't duplicate."""
        stream = io.StringIO()
        configure_structured_logging(stream=stream, include_emergo_context=True)
        configure_structured_logging(stream=stream, include_emergo_context=True)
        emergo_logger = logging.getLogger("emergo")
        # Count handlers pointing to this stream
        matching = [
            h
            for h in emergo_logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream is stream
        ]
        # Cleanup before asserting
        for h in matching:
            emergo_logger.removeHandler(h)
        assert len(matching) == 1


# ---------------------------------------------------------------------------
# TestGetEmergoLogger
# ---------------------------------------------------------------------------


class TestGetEmergoLogger:
    def test_returns_logger_adapter(self):
        adapter = get_emergo_logger("emergo.test_module", run_id="abc")
        assert isinstance(adapter, logging.LoggerAdapter)

    def test_context_fields_appear_in_output(self):
        stream = io.StringIO()
        handler = configure_structured_logging(stream=stream, include_emergo_context=False)
        test_logger_name = "emergo.test_get_logger_" + str(id(stream))
        base = logging.getLogger(test_logger_name)
        base.addHandler(handler)
        base.setLevel(logging.DEBUG)
        base.propagate = False

        adapter = get_emergo_logger(test_logger_name, run_id="exp_007", n_agents=10)
        adapter.info("adapter message")

        output = stream.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["run_id"] == "exp_007"
        assert parsed["n_agents"] == 10
        assert parsed["message"] == "adapter message"
