"""Unit tests for PSE broker_worker.py.

No RabbitMQ required — PSEBrokerWorker's async loop is never started.
BrokerGp is exercised by injecting a mock broker whose publish_run_trial
side-effect immediately resolves the trial.

Tests:
  _unpack_result   — float, dict, failure, unexpected format
  do_measurement   — success, failure, timeout, channel returned on timeout
  _filter_discrete_points — placeholder returns all points and logs warning
  PSEBrokerWorker  — register_trial / pop_result isolation
"""

import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pse.broker_worker import BrokerGp, PSEBrokerWorker


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EXP_PAR = pd.DataFrame([
    {
        "name": "DOPC", "type": "compound", "value": 0.0,
        "lower_opt": 0.0, "upper_opt": 0.3, "step_opt": 0.1, "optimize": True,
    },
])


def make_entry():
    return {
        "parameter names": ["DOPC"],
        "position": [0.1],
        "itlabel": 0,
        "value": None,
        "variance": None,
    }


def make_mock_broker(protocol_result=None, success=True, set_event=True):
    """Return a mock PSEBrokerWorker that resolves trials synchronously.

    publish_run_trial stores the result and sets the threading.Event created
    by register_trial, so do_measurement unblocks immediately.
    """
    broker = MagicMock(spec=PSEBrokerWorker)
    _events: dict[str, threading.Event] = {}
    _results: dict[str, dict] = {}

    if success:
        payload = {
            "result": {"protocol_result": protocol_result},
            "channel": 0,
        }
    else:
        payload = {"error": "method failed", "channel": 0}

    def fake_register(trial_id):
        event = threading.Event()
        _events[trial_id] = event
        return event

    def fake_publish(trial_id, **kwargs):
        _results[trial_id] = {"success": success, "payload": payload}
        if set_event:
            _events[trial_id].set()

    broker.register_trial.side_effect = fake_register
    broker.publish_run_trial.side_effect = fake_publish
    broker.pop_result.side_effect = lambda tid: _results.pop(tid, {})

    return broker


def make_broker_gp(broker, tmp_path, channels=None, trial_timeout=60.0):
    return BrokerGp(
        broker_worker=broker,
        protocol_id="test-proto",
        channels=channels,
        trial_timeout=trial_timeout,
        exp_par=EXP_PAR,
        storage_path=str(tmp_path),
        optimizer="gpcam",
        resume=False,
    )


# ---------------------------------------------------------------------------
# _unpack_result
# ---------------------------------------------------------------------------

class TestUnpackResult:

    def _make_gp_with_result(self, tmp_path, trial_id, stored):
        broker = MagicMock(spec=PSEBrokerWorker)
        broker.pop_result.return_value = stored
        gp = make_broker_gp(broker, tmp_path)
        return gp

    def test_plain_float(self, tmp_path):
        gp = self._make_gp_with_result(tmp_path, "tid", {
            "success": True,
            "payload": {"result": {"protocol_result": -12.5}},
        })
        value, variance = gp._unpack_result("tid", 0)
        assert value == -12.5
        assert variance is None

    def test_dict_with_value_and_variance(self, tmp_path):
        gp = self._make_gp_with_result(tmp_path, "tid", {
            "success": True,
            "payload": {"result": {"protocol_result": {"value": -8.3, "variance": 1.5}}},
        })
        value, variance = gp._unpack_result("tid", 0)
        assert value == pytest.approx(-8.3)
        assert variance == pytest.approx(1.5)

    def test_failure_returns_none(self, tmp_path):
        gp = self._make_gp_with_result(tmp_path, "tid", {
            "success": False,
            "payload": {"error": "formulation_infeasible"},
        })
        value, variance = gp._unpack_result("tid", 0)
        assert value is None
        assert variance is None

    def test_unexpected_format_returns_none(self, tmp_path):
        gp = self._make_gp_with_result(tmp_path, "tid", {
            "success": True,
            "payload": {"result": {"protocol_result": "oops a string"}},
        })
        value, variance = gp._unpack_result("tid", 0)
        assert value is None
        assert variance is None


# ---------------------------------------------------------------------------
# do_measurement
# ---------------------------------------------------------------------------

class TestDoMeasurement:

    def test_success_float(self, tmp_path):
        broker = make_mock_broker(protocol_result=-10.0)
        gp = make_broker_gp(broker, tmp_path)
        q = Queue()
        entry = make_entry()

        value, variance = gp.do_measurement({"DOPC": 0.1}, 0, entry, q)

        assert value == pytest.approx(-10.0)
        assert variance is None
        assert entry["value"] == pytest.approx(-10.0)
        assert not q.empty()

    def test_success_dict(self, tmp_path):
        broker = make_mock_broker(protocol_result={"value": -5.0, "variance": 2.0})
        gp = make_broker_gp(broker, tmp_path)
        q = Queue()

        value, variance = gp.do_measurement({"DOPC": 0.1}, 0, make_entry(), q)

        assert value == pytest.approx(-5.0)
        assert variance == pytest.approx(2.0)

    def test_failure_returns_none(self, tmp_path):
        broker = make_mock_broker(success=False)
        gp = make_broker_gp(broker, tmp_path)
        q = Queue()
        entry = make_entry()

        value, variance = gp.do_measurement({"DOPC": 0.1}, 0, entry, q)

        assert value is None
        assert variance is None
        assert entry["value"] is None
        assert not q.empty()  # entry still put on queue even on failure

    def test_timeout_returns_none(self, tmp_path):
        broker = make_mock_broker(set_event=False)  # event never set
        gp = make_broker_gp(broker, tmp_path, trial_timeout=0.05)
        q = Queue()

        value, variance = gp.do_measurement({"DOPC": 0.1}, 0, make_entry(), q)

        assert value is None
        assert variance is None

    def test_channel_returned_after_timeout(self, tmp_path):
        broker = make_mock_broker(set_event=False)
        gp = make_broker_gp(broker, tmp_path, channels=[0], trial_timeout=0.05)

        gp.do_measurement({"DOPC": 0.1}, 0, make_entry(), Queue())

        # Channel must be back in the pool after do_measurement returns,
        # regardless of whether it timed out.
        assert gp._channel_pool.qsize() == 1

    def test_publish_run_trial_called_with_correct_fields(self, tmp_path):
        broker = make_mock_broker(protocol_result=-1.0)
        gp = make_broker_gp(broker, tmp_path)

        gp.do_measurement({"DOPC": 0.2}, 0, make_entry(), Queue())

        call_kwargs = broker.publish_run_trial.call_args
        assert call_kwargs.kwargs["parameters"] == {"DOPC": 0.2}
        assert call_kwargs.kwargs["protocol_id"] == "test-proto"
        assert call_kwargs.kwargs["channel"] == 0

    def test_telemetry_published_on_success(self, tmp_path):
        broker = make_mock_broker(protocol_result=-1.0)
        gp = make_broker_gp(broker, tmp_path)

        gp.do_measurement({"DOPC": 0.1}, 0, make_entry(), Queue())

        routing_keys = [c.args[0] for c in broker.publish_telemetry.call_args_list]
        from roadmap_broker_client.topics import (
            PSE_EXPLORATION_POINT_DISPATCHED,
            PSE_EXPLORATION_POINT_COMPLETED,
        )
        assert PSE_EXPLORATION_POINT_DISPATCHED in routing_keys
        assert PSE_EXPLORATION_POINT_COMPLETED in routing_keys

    def test_point_completed_not_published_on_failure(self, tmp_path):
        broker = make_mock_broker(success=False)
        gp = make_broker_gp(broker, tmp_path)

        gp.do_measurement({"DOPC": 0.1}, 0, make_entry(), Queue())

        routing_keys = [c.args[0] for c in broker.publish_telemetry.call_args_list]
        from roadmap_broker_client.topics import PSE_EXPLORATION_POINT_COMPLETED
        assert PSE_EXPLORATION_POINT_COMPLETED not in routing_keys


# ---------------------------------------------------------------------------
# _filter_discrete_points placeholder
# ---------------------------------------------------------------------------

class TestFilterDiscretePoints:

    def test_returns_all_points_unchanged(self, tmp_path):
        broker = MagicMock(spec=PSEBrokerWorker)
        gp = make_broker_gp(broker, tmp_path)
        original = list(gp.gp_discrete_points)

        result = gp._filter_discrete_points()

        assert len(result) == len(original)

    def test_logs_warning(self, tmp_path):
        broker = MagicMock(spec=PSEBrokerWorker)
        gp = make_broker_gp(broker, tmp_path)

        with patch("pse.broker_worker.logger") as mock_logger:
            gp._filter_discrete_points()
            mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# PSEBrokerWorker — threading bridge in isolation (no async loop)
# ---------------------------------------------------------------------------

class TestPSEBrokerWorker:

    def test_register_and_pop(self):
        worker = PSEBrokerWorker()
        event = worker.register_trial("tid-1")

        assert isinstance(event, threading.Event)
        assert not event.is_set()

        # Simulate the broker thread delivering a result
        with worker._lock:
            worker._results["tid-1"] = {"success": True, "payload": {"x": 1}}
        event.set()

        assert event.is_set()
        result = worker.pop_result("tid-1")
        assert result["success"] is True
        assert result["payload"]["x"] == 1

    def test_pop_cleans_up_pending_and_results(self):
        worker = PSEBrokerWorker()
        worker.register_trial("tid-2")
        with worker._lock:
            worker._results["tid-2"] = {"success": False, "payload": {}}

        worker.pop_result("tid-2")

        with worker._lock:
            assert "tid-2" not in worker._pending
            assert "tid-2" not in worker._results

    def test_pop_unknown_trial_returns_empty(self):
        worker = PSEBrokerWorker()
        result = worker.pop_result("nonexistent")
        assert result == {}

    @pytest.mark.asyncio
    async def test_on_trial_result_sets_event(self):
        """_on_trial_result sets the threading.Event for a registered trial."""
        import uuid
        from roadmap_broker_client.envelope import Envelope
        from roadmap_broker_client.topics import TRIAL_COMPLETED

        trial_id = str(uuid.uuid4())
        worker = PSEBrokerWorker()
        event = worker.register_trial(trial_id)

        envelope = Envelope(
            task_id=trial_id,
            execution_policy="repeatable",
            payload={"trial_id": trial_id, "result": {"protocol_result": 5.0}, "channel": 0},
        )
        message = MagicMock()
        message.routing_key = TRIAL_COMPLETED

        await worker._on_trial_result(envelope, message)

        assert event.is_set()
        result = worker.pop_result(trial_id)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_on_trial_result_unknown_id_does_not_raise(self):
        import uuid
        from roadmap_broker_client.envelope import Envelope
        from roadmap_broker_client.topics import TRIAL_FAILED

        worker = PSEBrokerWorker()
        envelope = Envelope(
            task_id=uuid.uuid4(),
            execution_policy="repeatable",
            payload={"trial_id": str(uuid.uuid4()), "error": "oops"},
        )
        message = MagicMock()
        message.routing_key = TRIAL_FAILED

        # Should log a warning and return cleanly — no exception.
        await worker._on_trial_result(envelope, message)


# ---------------------------------------------------------------------------
# BrokerGp defaults
# ---------------------------------------------------------------------------

class TestBrokerGpDefaults:

    def test_default_channels_is_single_channel_zero(self, tmp_path):
        broker = MagicMock(spec=PSEBrokerWorker)
        gp = BrokerGp(
            broker_worker=broker,
            protocol_id="p",
            exp_par=EXP_PAR,
            storage_path=str(tmp_path),
            optimizer="gpcam",
            resume=False,
        )
        assert gp.parallel_measurements == 1
        assert gp._channel_pool.qsize() == 1
        assert gp._channel_pool.get() == 0

    def test_multichannel_sets_parallel_measurements(self, tmp_path):
        broker = MagicMock(spec=PSEBrokerWorker)
        gp = make_broker_gp(broker, tmp_path, channels=[0, 1, 2])
        assert gp.parallel_measurements == 3
        assert gp._channel_pool.qsize() == 3
