"""Broker integration for PhaseSpaceExplorer.

Runs an asyncio event loop in a dedicated daemon thread alongside the
synchronous GP optimization loop.  Provides:

  Inbound (exchange.protocol):
    trial.completed   — Protocol Studio signals a trial finished successfully
    trial.failed      — Protocol Studio signals a trial failed

  Outbound (exchange.protocol):
    command.protocol_studio.run_trial  — dispatch a trial to Protocol Studio
    pse.exploration.*                  — telemetry events

Thread model
------------
The broker event loop runs in a daemon thread.  BrokerGp measurement threads
call register_trial() + publish_run_trial(), then block on threading.Event
until the broker thread sets it on receipt of trial.completed / trial.failed.
"""

import asyncio
import logging
import queue
import threading
from typing import Dict, List, Optional
from uuid import uuid4

import aio_pika
import numpy as np
from gpcam import GPOptimizer

from roadmap_broker_client.connection import get_connection
from roadmap_broker_client.consumer import consume
from roadmap_broker_client.envelope import Envelope, build
from roadmap_broker_client.publisher import publish
from roadmap_broker_client.topology import declare_topology
from roadmap_broker_client.topics import (
    CMD_RUN_TRIAL,
    PROTOCOL_EXCHANGE,
    PSE_EXPLORATION_COMPLETED,
    PSE_EXPLORATION_POINT_COMPLETED,
    PSE_EXPLORATION_POINT_DISPATCHED,
    PSE_EXPLORATION_STARTED,
    PSE_EXPLORATION_STOPPED,
    TRIAL_COMPLETED,
    TRIAL_FAILED,
    command_key,
)

from pse.gp import Gp

logger = logging.getLogger(__name__)

_READY_TIMEOUT = 30.0    # seconds to wait for initial broker connection
_TRIAL_TIMEOUT = 7200.0  # default per-trial timeout (2 hours)


# ---------------------------------------------------------------------------
# Acquisition function
# ---------------------------------------------------------------------------

def acq_variance_target(x: np.ndarray, gpoptimizer: GPOptimizer) -> np.ndarray:
    """Variance / (mean + offset)^2 — balances exploration against regions of
    high expected signal. Tolerance term prevents division by zero near zero mean."""
    tolerance = 5
    var = np.array(gpoptimizer.posterior_covariance(x, variance_only=True)["v(x)"])
    mean = np.array(gpoptimizer.posterior_mean(x)["f(x)"])
    return var / ((mean + 25) ** 2 + tolerance ** 2)


# ---------------------------------------------------------------------------
# PSEBrokerWorker
# ---------------------------------------------------------------------------

class PSEBrokerWorker:
    """Asyncio broker consumer/publisher running in a daemon thread.

    Provides a thread-safe API for BrokerGp measurement threads to register
    pending trials, publish run_trial commands, and retrieve results.

    Wire in gp_server.py:
        self._broker = PSEBrokerWorker()
        self._broker.start()
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._exchange: Optional[aio_pika.abc.AbstractExchange] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

        self._lock = threading.Lock()
        # trial_id → threading.Event, set when the result arrives
        self._pending: Dict[str, threading.Event] = {}
        # trial_id → result dict (success flag + raw payload)
        self._results: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="pse-broker"
        )
        self._thread.start()
        if not self._ready.wait(timeout=_READY_TIMEOUT):
            raise TimeoutError(
                f"PSE broker worker did not connect to RabbitMQ within {_READY_TIMEOUT}s."
            )

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        except Exception:
            logger.exception("PSE broker worker loop exited with error.")

    async def _run(self) -> None:
        connection = await get_connection()
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=10)
            await declare_topology(channel)

            self._exchange = await channel.get_exchange(PROTOCOL_EXCHANGE)
            self._ready.set()

            # Non-durable — auto-deletes on disconnect so stale trial results
            # from a previous session do not replay on restart.
            result_queue = await channel.declare_queue(
                "pse.trial_results",
                durable=False,
                auto_delete=True,
                arguments={"x-dead-letter-exchange": "exchange.dead_letter"},
            )
            await result_queue.bind(self._exchange, routing_key=TRIAL_COMPLETED)
            await result_queue.bind(self._exchange, routing_key=TRIAL_FAILED)

            logger.info("PSE broker worker running.")
            await consume(result_queue, self._on_trial_result)

    # ------------------------------------------------------------------
    # Inbound
    # ------------------------------------------------------------------

    async def _on_trial_result(
        self, envelope: Envelope, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        payload = envelope.payload or {}
        trial_id = payload.get("trial_id")
        if not trial_id:
            logger.warning("Trial result received with no trial_id — discarding.")
            return

        success = message.routing_key == TRIAL_COMPLETED

        with self._lock:
            self._results[trial_id] = {"success": success, "payload": payload}
            event = self._pending.get(trial_id)

        if event:
            event.set()
        else:
            logger.warning(
                "Trial result for unregistered trial_id %s — discarding.", trial_id
            )

    # ------------------------------------------------------------------
    # Thread-safe API for BrokerGp
    # ------------------------------------------------------------------

    def register_trial(self, trial_id: str) -> threading.Event:
        """Register a pending trial before publishing run_trial.
        Must be called before publish_run_trial to avoid a race where the
        result arrives before the event is registered."""
        event = threading.Event()
        with self._lock:
            self._pending[trial_id] = event
        return event

    def pop_result(self, trial_id: str) -> dict:
        """Retrieve and remove the stored result for a completed trial."""
        with self._lock:
            self._pending.pop(trial_id, None)
            return self._results.pop(trial_id, {})

    def publish_run_trial(
        self,
        trial_id: str,
        parameters: dict,
        channel: int,
        protocol_id: str,
        version_num=None,
    ) -> None:
        """Publish command.protocol_studio.run_trial. Blocks until sent so
        do_measurement can safely block on the event immediately after."""
        self._ready.wait()
        future = asyncio.run_coroutine_threadsafe(
            self._emit_run_trial(trial_id, parameters, channel, protocol_id, version_num),
            self._loop,
        )
        future.result()

    async def _emit_run_trial(
        self,
        trial_id: str,
        parameters: dict,
        channel: int,
        protocol_id: str,
        version_num,
    ) -> None:
        rk = command_key("protocol_studio", CMD_RUN_TRIAL)
        env = build(
            device_id="pse",
            routing_key=rk,
            task_id=trial_id,
            payload={
                "trial_id": trial_id,
                "parameters": parameters,
                "channel": channel,
                "protocol_id": protocol_id,
                "version_num": version_num,
            },
        )
        await publish(self._exchange, rk, env)

    def publish_telemetry(self, routing_key: str, payload: dict) -> None:
        """Fire-and-forget telemetry publish. Silently drops if not yet connected."""
        if not self._ready.is_set() or self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._emit(routing_key, payload), self._loop
        )

    async def _emit(self, routing_key: str, payload: dict) -> None:
        env = build(device_id="pse", routing_key=routing_key, payload=payload)
        await publish(self._exchange, routing_key, env)


# ---------------------------------------------------------------------------
# BrokerGp
# ---------------------------------------------------------------------------

class BrokerGp(Gp):
    """Gp subclass that dispatches measurements via the broker instead of
    calling lh_manager directly.

    Overrides:
      do_measurement()             — publish run_trial, block on threading.Event
      gp_hardware_intitialzation() — filter discrete points; init protocol placeholder
      gp_hardware_shutdown()       — shutdown protocol placeholder
    """

    def __init__(
        self,
        broker_worker: PSEBrokerWorker,
        protocol_id: str,
        channels: Optional[List[int]] = None,
        trial_timeout: float = _TRIAL_TIMEOUT,
        version_num=None,
        **gp_kwargs,
    ) -> None:
        self._broker = broker_worker
        self._protocol_id = protocol_id
        self._version_num = version_num
        self._trial_timeout = trial_timeout

        if channels is None:
            channels = [0]

        # Thread-safe channel pool. Each do_measurement thread claims a channel
        # for the duration of the trial, then returns it. Naturally blocks when
        # all channels are occupied, replacing the busy-flag pattern.
        self._channel_pool: queue.Queue = queue.Queue()
        for ch in channels:
            self._channel_pool.put(ch)

        # Ensure parallel_measurements matches the channel count regardless of
        # what the caller passed.
        gp_kwargs.pop("parallel_measurements", None)
        super().__init__(parallel_measurements=len(channels), **gp_kwargs)

        self.acq_func = acq_variance_target

    # ------------------------------------------------------------------
    # Constraint filtering
    # ------------------------------------------------------------------

    def _filter_discrete_points(self) -> list:
        """Return the feasible subset of gp_discrete_points.

        Placeholder: returns all points unchanged. Implement by building A (k×d)
        and b (k,) from the protocol's constraint schema, then filtering with:

            X = np.array(self.gp_discrete_points)
            mask = np.all(X @ A.T <= b, axis=1)
            return [self.gp_discrete_points[i] for i in np.where(mask)[0]]
        """
        logger.warning(
            "No constraint filter configured — using full discrete point set (%d points).",
            len(self.gp_discrete_points),
        )
        return self.gp_discrete_points

    # ------------------------------------------------------------------
    # Hardware lifecycle
    # ------------------------------------------------------------------

    def gp_hardware_intitialzation(self) -> bool:  # noqa: N802 — preserves base class spelling
        self.gp_discrete_points = self._filter_discrete_points()
        # TODO: send run_trial with an init protocol_id for each channel once
        # init protocols are defined in Protocol Studio.
        logger.info(
            "BrokerGp initialised: %d channels, %d candidate points after filtering.",
            self._channel_pool.qsize(),
            len(self.gp_discrete_points),
        )
        return True

    def gp_hardware_shutdown(self) -> bool:  # noqa: N802
        # TODO: send run_trial with a shutdown protocol_id for each channel.
        logger.info("BrokerGp shutdown.")
        return True

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def do_measurement(self, optpars: dict, it_label, entry: dict, q) -> tuple:
        """Publish run_trial, block until result arrives, feed scalar to GP queue."""
        channel = self._channel_pool.get()
        trial_id = str(uuid4())

        try:
            # Register before publishing to eliminate the race where the result
            # arrives before the event exists.
            event = self._broker.register_trial(trial_id)

            self._broker.publish_telemetry(PSE_EXPLORATION_POINT_DISPATCHED, {
                "trial_id": trial_id,
                "iteration": it_label,
                "parameters": optpars,
                "channel": channel,
            })

            self._broker.publish_run_trial(
                trial_id=trial_id,
                parameters=optpars,
                channel=channel,
                protocol_id=self._protocol_id,
                version_num=self._version_num,
            )

            if not event.wait(timeout=self._trial_timeout):
                logger.error(
                    "Trial %s (iteration %s) timed out after %ss.",
                    trial_id, it_label, self._trial_timeout,
                )
                value, variance = None, None
            else:
                value, variance = self._unpack_result(trial_id, it_label)

            if value is not None:
                self._broker.publish_telemetry(PSE_EXPLORATION_POINT_COMPLETED, {
                    "trial_id": trial_id,
                    "iteration": it_label,
                    "result": value,
                    "variance": variance,
                    "channel": channel,
                })

        finally:
            self._channel_pool.put(channel)

        entry["value"] = value
        entry["variance"] = variance
        q.put(entry)
        return value, variance

    def _unpack_result(self, trial_id: str, it_label) -> tuple:
        """Extract (value, variance) from the stored trial result payload.

        Protocol Studio publishes trial.completed with:
            {"trial_id": ..., "result": {"protocol_result": <value>}, "channel": ...}

        protocol_result may be:
          - A plain float             → value=float, variance=None
          - {"value": float,
             "variance": float}      → both unpacked directly
        """
        stored = self._broker.pop_result(trial_id)

        if not stored.get("success"):
            error = stored.get("payload", {}).get("error", "unknown")
            logger.warning(
                "Trial %s (iteration %s) failed: %s", trial_id, it_label, error
            )
            return None, None

        protocol_result = stored.get("payload", {}).get("result", {}).get("protocol_result")

        if isinstance(protocol_result, dict):
            return protocol_result.get("value"), protocol_result.get("variance")

        if isinstance(protocol_result, (int, float)):
            return float(protocol_result), None

        logger.warning(
            "Trial %s: unexpected protocol_result format %r — recording None.",
            trial_id, protocol_result,
        )
        return None, None
