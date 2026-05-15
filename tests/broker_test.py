"""
Broker integration test for PhaseSpaceExplorer.

Starts a PSEBrokerWorker and a fake Protocol Studio subscriber in-process,
then exercises BrokerGp.do_measurement() end-to-end over a real RabbitMQ
connection.

Tests:
  - Round-trip success (float result)
  - Round-trip success (dict result with variance)
  - Round-trip failure (trial.failed → do_measurement returns None)
  - Multi-channel: two concurrent measurements use different channels

Requires RabbitMQ running at localhost:5672:
  docker-compose up -d rabbitmq

Usage:
  cd PhaseSpaceExplorer
  python tests/broker_test.py
"""

import asyncio
import logging
import sys
import tempfile
import threading
from pathlib import Path
from queue import Queue

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "roadmap-broker-client"))

from roadmap_broker_client.connection import get_connection
from roadmap_broker_client.envelope import Envelope, build
from roadmap_broker_client.publisher import publish
from roadmap_broker_client.topology import declare_topology
from roadmap_broker_client.topics import (
    CMD_RUN_TRIAL,
    PROTOCOL_EXCHANGE,
    TRIAL_COMPLETED,
    TRIAL_FAILED,
    command_key,
)

from pse.broker_worker import BrokerGp, PSEBrokerWorker

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

APP_START_WAIT = 2.0

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_results: list[tuple[str, bool]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail and not condition else ""
    print(f"  [{status}] {name}{suffix}")
    _results.append((name, condition))


# ---------------------------------------------------------------------------
# Minimal parameter space
# ---------------------------------------------------------------------------

EXP_PAR = pd.DataFrame([
    {
        "name": "DOPC", "type": "compound", "value": 0.0,
        "lower_opt": 0.0, "upper_opt": 0.3, "step_opt": 0.1, "optimize": True,
    },
])


def make_entry(channel=0):
    return {
        "parameter names": ["DOPC"],
        "position": [0.1],
        "itlabel": channel,
        "value": None,
        "variance": None,
    }


# ---------------------------------------------------------------------------
# Fake Protocol Studio
# ---------------------------------------------------------------------------

class FakeProtocolStudio:
    """Subscribes to run_trial commands and immediately replies.

    reply_with controls what is published back:
      "success_float"  → trial.completed, protocol_result = -42.0
      "success_dict"   → trial.completed, protocol_result = {"value": -7.5, "variance": 1.2}
      "failure"        → trial.failed
    """

    def __init__(self, exchange, reply_with: str = "success_float") -> None:
        self._exchange = exchange
        self._reply_with = reply_with

    async def run(self, queue) -> None:
        async with queue.iterator() as msgs:
            async for msg in msgs:
                try:
                    env = Envelope.model_validate_json(msg.body)
                    payload = env.payload or {}
                    trial_id = payload.get("trial_id", str(env.task_id))
                    channel = payload.get("channel", 0)

                    if self._reply_with == "failure":
                        rk = TRIAL_FAILED
                        reply_payload = {
                            "trial_id": trial_id,
                            "error": "formulation_infeasible",
                            "channel": channel,
                        }
                    else:
                        rk = TRIAL_COMPLETED
                        if self._reply_with == "success_dict":
                            protocol_result = {"value": -7.5, "variance": 1.2}
                        else:
                            protocol_result = -42.0
                        reply_payload = {
                            "trial_id": trial_id,
                            "result": {"protocol_result": protocol_result},
                            "channel": channel,
                        }

                    reply = build(
                        device_id="fake_protocol_studio",
                        routing_key=rk,
                        task_id=trial_id,
                        payload=reply_payload,
                    )
                    await publish(self._exchange, rk, reply)
                    await msg.ack()

                except Exception:
                    logging.exception("[FakePS] error processing message")
                    await msg.nack(requeue=False)


async def setup_fake_ps(connection, reply_with="success_float") -> tuple:
    """Create a dedicated AMQP channel, declare a unique fake PS queue, and
    start its consumer task.  Each test gets its own channel and queue name so
    that cancelling one test's consumer never invalidates the next test's
    channel or re-uses an auto-deleted queue."""
    import uuid as _uuid
    ch = await connection.channel()
    await ch.set_qos(prefetch_count=10)
    exchange = await ch.get_exchange(PROTOCOL_EXCHANGE)
    queue_name = f"test.pse.fake_ps.{_uuid.uuid4().hex[:8]}"
    cmd_queue = await ch.declare_queue(queue_name, durable=False, auto_delete=True)
    await cmd_queue.bind(
        exchange, routing_key=command_key("protocol_studio", CMD_RUN_TRIAL)
    )
    fps = FakeProtocolStudio(exchange, reply_with=reply_with)
    task = asyncio.create_task(fps.run(cmd_queue))
    return fps, task, ch


async def teardown_fake_ps(task, ch) -> None:
    """Cancel the consumer task and close its dedicated channel."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await ch.close()


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

async def test_round_trip_float(broker: PSEBrokerWorker, tmp_dir: str, connection) -> None:
    print("\n[test_round_trip_float] do_measurement → trial.completed (float)")
    fps, task, ch = await setup_fake_ps(connection, reply_with="success_float")

    gp = BrokerGp(
        broker_worker=broker,
        protocol_id="test-proto",
        channels=[0],
        trial_timeout=15.0,
        exp_par=EXP_PAR,
        storage_path=tmp_dir,
        optimizer="gpcam",
        resume=False,
    )

    q = Queue()
    entry = make_entry()
    value, variance = await asyncio.to_thread(
        gp.do_measurement, {"DOPC": 0.1}, 0, entry, q
    )

    check("value == -42.0", value == -42.0, f"got {value!r}")
    check("variance is None", variance is None, f"got {variance!r}")
    check("entry updated", entry["value"] == -42.0)
    check("result put on queue", not q.empty())

    await teardown_fake_ps(task, ch)


async def test_round_trip_dict(broker: PSEBrokerWorker, tmp_dir: str, connection) -> None:
    print("\n[test_round_trip_dict] do_measurement → trial.completed (dict with variance)")
    fps, task, ch = await setup_fake_ps(connection, reply_with="success_dict")

    gp = BrokerGp(
        broker_worker=broker,
        protocol_id="test-proto",
        channels=[0],
        trial_timeout=15.0,
        exp_par=EXP_PAR,
        storage_path=tmp_dir,
        optimizer="gpcam",
        resume=False,
    )

    value, variance = await asyncio.to_thread(
        gp.do_measurement, {"DOPC": 0.1}, 0, make_entry(), Queue()
    )

    check("value == -7.5", value is not None and abs(value - (-7.5)) < 1e-6, f"got {value!r}")
    check("variance == 1.2", variance is not None and abs(variance - 1.2) < 1e-6, f"got {variance!r}")

    await teardown_fake_ps(task, ch)


async def test_round_trip_failure(broker: PSEBrokerWorker, tmp_dir: str, connection) -> None:
    print("\n[test_round_trip_failure] do_measurement → trial.failed → (None, None)")
    fps, task, ch = await setup_fake_ps(connection, reply_with="failure")

    gp = BrokerGp(
        broker_worker=broker,
        protocol_id="test-proto",
        channels=[0],
        trial_timeout=15.0,
        exp_par=EXP_PAR,
        storage_path=tmp_dir,
        optimizer="gpcam",
        resume=False,
    )

    value, variance = await asyncio.to_thread(
        gp.do_measurement, {"DOPC": 0.1}, 0, make_entry(), Queue()
    )

    check("value is None on failure", value is None, f"got {value!r}")
    check("variance is None on failure", variance is None, f"got {variance!r}")

    await teardown_fake_ps(task, ch)


async def test_multichannel(broker: PSEBrokerWorker, tmp_dir: str, connection) -> None:
    print("\n[test_multichannel] two concurrent measurements use different channels")
    fps, task, ch = await setup_fake_ps(connection, reply_with="success_float")

    channels_used: list[int] = []
    original_publish = broker.publish_run_trial

    # Wrap publish_run_trial to record which channel each trial uses.
    def recording_publish(trial_id, parameters, channel, protocol_id, version_num=None):
        channels_used.append(channel)
        original_publish(
            trial_id=trial_id,
            parameters=parameters,
            channel=channel,
            protocol_id=protocol_id,
            version_num=version_num,
        )

    broker.publish_run_trial = recording_publish

    gp = BrokerGp(
        broker_worker=broker,
        protocol_id="test-proto",
        channels=[0, 1],
        trial_timeout=15.0,
        exp_par=EXP_PAR,
        storage_path=tmp_dir,
        optimizer="gpcam",
        resume=False,
    )

    # Launch both measurements concurrently in threads.
    t0 = threading.Thread(
        target=gp.do_measurement, args=({"DOPC": 0.1}, 0, make_entry(0), Queue())
    )
    t1 = threading.Thread(
        target=gp.do_measurement, args=({"DOPC": 0.2}, 1, make_entry(1), Queue())
    )
    t0.start()
    t1.start()
    await asyncio.to_thread(t0.join, 20.0)
    await asyncio.to_thread(t1.join, 20.0)

    check("both measurements completed", not t0.is_alive() and not t1.is_alive())
    check("two distinct channels used", len(set(channels_used)) == 2,
          f"channels used: {channels_used}")
    check("channels are 0 and 1", set(channels_used) == {0, 1},
          f"got {channels_used}")
    check("channel pool fully returned", gp._channel_pool.qsize() == 2)

    broker.publish_run_trial = original_publish
    await teardown_fake_ps(task, ch)


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

async def run_harness() -> None:
    print("Starting PSE broker worker ...")
    broker = PSEBrokerWorker()
    broker.start()
    print(f"Waiting {APP_START_WAIT}s for broker to connect ...")
    await asyncio.sleep(APP_START_WAIT)

    connection = await get_connection()
    async with connection:
        # Declare topology once on a throwaway channel; each test creates its
        # own channel for the fake PS so teardowns don't invalidate each other.
        init_ch = await connection.channel()
        await declare_topology(init_ch)
        await init_ch.close()

        with tempfile.TemporaryDirectory() as tmp_dir:
            print("\n" + "=" * 60)
            print("Running PhaseSpaceExplorer broker tests")
            print("=" * 60)

            try:
                await test_round_trip_float(broker, tmp_dir, connection)
                await test_round_trip_dict(broker, tmp_dir, connection)
                await test_round_trip_failure(broker, tmp_dir, connection)
                await test_multichannel(broker, tmp_dir, connection)
            except Exception:
                logging.exception("Unhandled exception in test run")

    passed = sum(1 for _, ok in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print("Failed:")
        for name, ok in _results:
            if not ok:
                print(f"  - {name}")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(name)-20s  %(levelname)s  %(message)s",
    )
    success = asyncio.run(run_harness())
    sys.exit(0 if success else 1)
