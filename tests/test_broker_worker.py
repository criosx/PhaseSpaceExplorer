"""Unit tests for PSE broker_worker.py — Phase 2 Step C.

No RabbitMQ required.  PSEBrokerWorker's async loop is never started.
PSEPointService GP internals (GPOptimizer, gpCAMstream, exp_par) are mocked
so no gpcam initialisation is needed.

Tests:
  PSEPointService       — request_point, submit_result, cancel_trial,
                          notify_in_flight, phantom-tell, duplicate prevention
  PSEBrokerWorker       — command dispatch (_handle_request_point,
                          _handle_submit_result, _handle_cancel_trial,
                          _handle_notify_in_flight), set_service guard,
                          paused flag, emit shape
  _filter_discrete_points — placeholder returns all points and logs warning
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pse.broker_worker import PSEBrokerWorker, PSEPointService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXP_PAR = pd.DataFrame([
    {
        "name": "DOPC", "type": "compound", "value": 0.0,
        "lower_opt": 0.0, "upper_opt": 0.3, "step_opt": 0.1, "optimize": True,
    },
    {
        "name": "DPPC", "type": "compound", "value": 0.0,
        "lower_opt": 0.0, "upper_opt": 0.3, "step_opt": 0.1, "optimize": True,
    },
])


def make_service() -> PSEPointService:
    """Return a PSEPointService with GP internals mocked out.

    The mock GPOptimizer returns deterministic values so request_point()
    exercises the real in-flight / phantom-tell logic without touching gpcam.
    """
    service = PSEPointService.__new__(PSEPointService)
    service._lock = threading.Lock()
    service._paused = False
    service._in_flight: dict = {}
    service.gpiteration = 0
    service.exp_par = EXP_PAR
    service.acq_func = MagicMock()
    service.gp_discrete_points = [[0.1, 0.0], [0.2, 0.1], [0.3, 0.2]]
    service.signal_estimate = 1.0
    service.gpcam_init_dataset_size = 5
    service.train_global_every = 10
    service.gpcam_iterations = 100

    # Mock methods that touch disk / real GP
    service.gpcam_init_ae = MagicMock()
    service.gpcam_train = MagicMock()
    service.gpcam_prediction = MagicMock()
    service.plot_results = MagicMock()
    service.results_io = MagicMock()
    service.iterations_inprogress_save_to_file = MagicMock()
    service.gp_hardware_shutdown = MagicMock()

    # Mock GPOptimizer (my_ae)
    ae = MagicMock()
    ae.ask.return_value = {"x": [np.array([0.15, 0.05])]}
    ae.posterior_mean.return_value = {"m(x)": np.array([0.5])}
    ae.posterior_covariance.return_value = {"v(x)": np.array([0.01])}
    ae.tell = MagicMock()
    service.my_ae = ae

    # Mock gpCAMstream as an append-able list-like
    service.gpCAMstream = MagicMock()
    service.gpCAMstream.__len__ = MagicMock(return_value=2)

    service.task_dict = {"cancelled": False, "paused": False, "progress": "0%", "status": "running"}
    service.optimizer = "gpcam"
    service._acq_func_name = "variance_target"

    return service


# ---------------------------------------------------------------------------
# PSEPointService — request_point
# ---------------------------------------------------------------------------

class TestRequestPoint:

    def test_returns_trial_id_and_params(self):
        service = make_service()
        trial_id, params, pse_ctx = service.request_point()

        assert isinstance(trial_id, str) and len(trial_id) == 36  # UUID
        assert set(params.keys()) == {"DOPC", "DPPC"}
        assert all(isinstance(v, float) for v in params.values())
        assert "acq_func" in pse_ctx and "training_set_size" in pse_ctx

    def test_registers_in_flight(self):
        service = make_service()
        trial_id, _, _ctx = service.request_point()
        assert trial_id in service._in_flight

    def test_increments_gpiteration(self):
        service = make_service()
        service.request_point()
        assert service.gpiteration == 1

    def test_phantom_tell_called(self):
        service = make_service()
        service.request_point()
        assert service.my_ae.tell.called

    def test_two_concurrent_requests_call_ask_twice(self):
        """Each request_point acquires the lock, so two sequential calls both ask."""
        service = make_service()
        tid1, _, _ctx1 = service.request_point()
        tid2, _, _ctx2 = service.request_point()
        assert tid1 != tid2
        assert service.my_ae.ask.call_count == 2

    def test_raises_if_not_initialized(self):
        service = make_service()
        service.my_ae = None  # simulate uninitialized
        with pytest.raises(RuntimeError, match="initialize"):
            service.request_point()


# ---------------------------------------------------------------------------
# PSEPointService — submit_result
# ---------------------------------------------------------------------------

class TestSubmitResult:

    def test_removes_from_in_flight(self):
        service = make_service()
        trial_id, _, _ctx = service.request_point()
        service.submit_result(trial_id, -12.5, None)
        assert trial_id not in service._in_flight

    def test_updates_gpCAMstream(self):
        service = make_service()
        trial_id, _, _ctx = service.request_point()
        service.submit_result(trial_id, -8.0, 0.5)
        service.gpCAMstream.loc.__setitem__.assert_called_once()

    def test_uses_default_variance_when_none(self):
        service = make_service()
        trial_id, _, _ctx = service.request_point()
        service.submit_result(trial_id, 1.0, None)
        # signal_estimate is 1.0 → default variance is 1e-6
        _, call_kwargs = service.gpCAMstream.loc.__setitem__.call_args
        # variance field in the row
        row = service.gpCAMstream.loc.__setitem__.call_args[0][1]
        assert row["variance"] == pytest.approx(1e-6)

    def test_ignores_unknown_trial_id(self):
        service = make_service()
        service.submit_result("nonexistent", 1.0, None)
        service.gpCAMstream.loc.__setitem__.assert_not_called()

    def test_reinitialises_ae_after_submit(self):
        service = make_service()
        trial_id, _, _ctx = service.request_point()
        service.submit_result(trial_id, 2.0, None)
        assert service.gpcam_init_ae.called


# ---------------------------------------------------------------------------
# PSEPointService — cancel_trial
# ---------------------------------------------------------------------------

class TestCancelTrial:

    def test_removes_from_in_flight(self):
        service = make_service()
        trial_id, _, _ctx = service.request_point()
        service.cancel_trial(trial_id)
        assert trial_id not in service._in_flight

    def test_rebuilds_phantom_tells(self):
        service = make_service()
        tid1, _, _ctx = service.request_point()
        service.request_point()  # second in-flight
        service.cancel_trial(tid1)
        # gpcam_init_ae called once on cancel (to rebuild without cancelled point)
        assert service.gpcam_init_ae.call_count >= 1

    def test_ignores_unknown_trial_id(self):
        service = make_service()
        # Should not raise
        service.cancel_trial("nonexistent")


# ---------------------------------------------------------------------------
# PSEPointService — notify_in_flight
# ---------------------------------------------------------------------------

class TestNotifyInFlight:

    def test_repopulates_in_flight(self):
        service = make_service()
        service.notify_in_flight("old-trial", {"DOPC": 0.1, "DPPC": 0.05})
        assert "old-trial" in service._in_flight

    def test_position_matches_params(self):
        service = make_service()
        service.notify_in_flight("t1", {"DOPC": 0.2, "DPPC": 0.1})
        pos = service._in_flight["t1"]
        np.testing.assert_array_almost_equal(pos, [0.2, 0.1])

    def test_rebuilds_phantom_tells(self):
        service = make_service()
        service.notify_in_flight("t1", {"DOPC": 0.1, "DPPC": 0.0})
        assert service.gpcam_init_ae.called


# ---------------------------------------------------------------------------
# _filter_discrete_points placeholder
# ---------------------------------------------------------------------------

class TestFilterDiscretePoints:

    def test_returns_all_points_unchanged(self):
        service = make_service()
        original = list(service.gp_discrete_points)
        result = service._filter_discrete_points()
        assert result == original

    def test_logs_warning(self):
        service = make_service()
        with patch("pse.broker_worker.logger") as mock_logger:
            service._filter_discrete_points()
            mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# PSEBrokerWorker — command dispatch (no async loop, no RabbitMQ)
# ---------------------------------------------------------------------------

def make_broker_worker_with_service(paused=False):
    """Return (worker, mock_service) with the loop NOT started."""
    worker = PSEBrokerWorker.__new__(PSEBrokerWorker)
    worker._loop = None
    worker._exchange = MagicMock()
    worker._thread = None
    worker._ready = threading.Event()
    worker._ready.set()  # pretend connected
    worker._lock = threading.Lock()
    worker._service = None

    service = MagicMock(spec=PSEPointService)
    service._paused = paused
    service.request_point.return_value = ("trial-uuid", {"DOPC": 0.1}, {"acq_func": "variance_target", "training_set_size": 5, "uncertainty_at_suggestion": 0.02})
    worker._service = service

    # Replace _emit with a no-op coroutine so handlers can be awaited
    worker._emit = AsyncMock()

    return worker, service


class TestPSEBrokerWorkerDispatch:

    @pytest.mark.asyncio
    async def test_request_point_calls_service(self):
        worker, service = make_broker_worker_with_service()
        await worker._handle_request_point({"campaign_id": "c1", "channel": 0})
        service.request_point.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_point_emits_point_suggested(self):
        from roadmap_broker_client.topics import PSE_POINT_SUGGESTED
        worker, service = make_broker_worker_with_service()
        await worker._handle_request_point({"campaign_id": "c1", "channel": 2})
        worker._emit.assert_awaited_once()
        rk = worker._emit.call_args[0][0]
        assert rk == PSE_POINT_SUGGESTED

    @pytest.mark.asyncio
    async def test_request_point_echoes_caller_context(self):
        worker, service = make_broker_worker_with_service()
        await worker._handle_request_point({"campaign_id": "c1", "channel": 2})
        payload = worker._emit.call_args[0][1]
        assert payload.get("campaign_id") == "c1"
        assert payload.get("channel") == 2

    @pytest.mark.asyncio
    async def test_request_point_skipped_when_paused(self):
        worker, service = make_broker_worker_with_service(paused=True)
        await worker._handle_request_point({})
        service.request_point.assert_not_called()
        worker._emit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_request_point_skipped_when_no_service(self):
        worker, _ = make_broker_worker_with_service()
        worker._service = None
        await worker._handle_request_point({})
        worker._emit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_submit_result_calls_service(self):
        worker, service = make_broker_worker_with_service()
        await worker._handle_submit_result({
            "trial_id": "t1", "value": -5.0, "variance": 0.1
        })
        service.submit_result.assert_called_once_with("t1", -5.0, 0.1)

    @pytest.mark.asyncio
    async def test_submit_result_rejects_missing_fields(self):
        worker, service = make_broker_worker_with_service()
        await worker._handle_submit_result({"trial_id": "t1"})  # no value
        service.submit_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_trial_calls_service(self):
        worker, service = make_broker_worker_with_service()
        await worker._handle_cancel_trial({"trial_id": "t2"})
        service.cancel_trial.assert_called_once_with("t2")

    @pytest.mark.asyncio
    async def test_cancel_trial_ignores_missing_id(self):
        worker, service = make_broker_worker_with_service()
        await worker._handle_cancel_trial({})
        service.cancel_trial.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_in_flight_calls_service(self):
        worker, service = make_broker_worker_with_service()
        await worker._handle_notify_in_flight({
            "trial_id": "t3", "parameters": {"DOPC": 0.2}
        })
        service.notify_in_flight.assert_called_once_with("t3", {"DOPC": 0.2})

    @pytest.mark.asyncio
    async def test_notify_in_flight_ignores_empty_params(self):
        worker, service = make_broker_worker_with_service()
        await worker._handle_notify_in_flight({"trial_id": "t3", "parameters": {}})
        service.notify_in_flight.assert_not_called()


# ---------------------------------------------------------------------------
# PSEBrokerWorker — set_service
# ---------------------------------------------------------------------------

class TestSetService:

    def test_set_service_registers_service(self):
        worker = PSEBrokerWorker.__new__(PSEBrokerWorker)
        worker._lock = threading.Lock()
        worker._service = None
        worker._ready = threading.Event()
        worker._loop = None

        mock_service = MagicMock(spec=PSEPointService)
        worker.set_service(mock_service)

        with worker._lock:
            assert worker._service is mock_service

    def test_set_service_none_clears_service(self):
        worker = PSEBrokerWorker.__new__(PSEBrokerWorker)
        worker._lock = threading.Lock()
        worker._service = MagicMock()
        worker._ready = threading.Event()
        worker._loop = None

        worker.set_service(None)

        with worker._lock:
            assert worker._service is None
