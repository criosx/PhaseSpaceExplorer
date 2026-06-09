# PhaseSpaceExplorer — Bayesian Optimization Engine

PhaseSpaceExplorer (PSE) is a Gaussian Process (gpCAM) optimization engine that
adaptively selects the next experiment parameters to maximize information gain. It acts as
a **passive point-generation service**: Protocol Studio drives the campaign loop and
requests points from PSE on demand; PSE does not drive the loop itself.

PSE builds a feasible parameter space from linear constraints, trains a GP model on
incoming trial results, and suggests the next measurement point when asked.

---

## Role in the System

PSE is a **Tier 1 service** on `exchange.protocol`. It is purely computational — it has
no knowledge of channels, device methods, or physical hardware.

```
Protocol Studio (campaign coordinator)
    │ command.pse.configure       → PSE initialises GP; publishes pse.ready
    │ command.pse.request_point   → PSE publishes pse.point_suggested
    │ command.pse.submit_result   → PSE updates GP model
    │ command.pse.stop_campaign   → PSE finalises, deregisters
    ▼
PhaseSpaceExplorer (PSEPointService)
    │ pse.ready
    │ pse.point_suggested
    │ pse.exploration.*  (telemetry)
    ▼
exchange.protocol
```

Protocol Studio reads `pse.point_suggested`, dispatches a trial, collects the result, and
sends it back via `command.pse.submit_result`. PSE never publishes a trial command
directly; it only responds to requests.

---

## Optimization Loop Rules

**Passive model.** PSE responds to `request_point` commands; it does not initiate trials.
The campaign coordinator in Protocol Studio owns the loop, channel allocation, and
concurrency. PSE is unaware of how many channels are active.

**Control measurement cycling.** Every `control_cycle` results per channel (default 4), a
new buffer baseline is collected and stored. All subsequent QCMD results for that channel
are normalized against the most recent control. This state is persisted to disk in
`results.json` and must survive restarts — `resume=True` rehydrates it on init.

**GP state is serialized to disk.** Hyperparameters, training dataset, and iteration count
are written to `storage_path/` after each update. On restart the GP rehydrates this state.
The broker `notify_in_flight` command re-populates in-flight trials so PSE does not
double-count results that arrive after a restart.

**Feasible parameter space.** If the configured Protocol has linear constraints (e.g.,
"sum of composition fractions ≤ 1"), PSE builds the feasible discrete grid using
union-find + group coarsening before the campaign starts. Both `gp_discrete_points`
(candidate points) and `gp_evaluation_points` (prediction grid) are restricted to the
feasible set. `feasible_point_count` is reported in `pse.ready` and shown in the
Protocol Studio GUI.

**Large data stays on disk.** GP model state, full results JSON, and posterior plots live
in `storage_path/`. The broker carries only scalar summaries per point — never model
binaries or raw QCMD arrays.

---

## Broker Interface

PSE binds a durable queue to `command.pse.#` on `exchange.protocol`.

### Subscribes to
| Command verb | Payload | Action |
|---|---|---|
| `configure` | `{campaign_id, exp_par, constraints, downsample_seed}` | Initialise `PSEPointService`; build feasible space; publish `pse.ready` |
| `request_point` | `{campaign_id, channel}` | Select next GP point; publish `pse.point_suggested` |
| `submit_result` | `{trial_id, result, channel}` | Update GP model with completed trial result |
| `cancel_trial` | `{trial_id}` | Discard in-flight trial; no model update |
| `notify_in_flight` | `{trials: [...]}` | Re-populate in-flight list after PSE restart |
| `stop_campaign` | `{campaign_id}` | Finalise campaign; deregister service |
| `announce_request` | `{}` | Re-publish `pse.ready` (used by PS after reconnect) |

All commands use `command_key("pse", verb)` → `command.pse.<verb>`.

### Publishes
| Routing key | When | Payload |
|---|---|---|
| `pse.ready` | After `configure` completes, and on `announce_request` | `{has_service: true, feasible_point_count: N, campaign_id: ...}` |
| `pse.point_suggested` | After `request_point` | `{trial_id, parameters, campaign_id, channel}` |
| `pse.exploration.started` | Campaign begins | `{campaign_id}` |
| `pse.exploration.point_dispatched` | Point sent for evaluation | `{campaign_id, trial_id, parameters}` |
| `pse.exploration.point_completed` | Result incorporated | `{campaign_id, trial_id, result}` |
| `pse.exploration.model_updated` | GP retrained | `{campaign_id, iteration}` |
| `pse.exploration.completed` | All iterations done | `{campaign_id}` |
| `pse.exploration.paused/resumed/stopped` | Operator control | `{campaign_id}` |

---

## GP Server

PSE runs as a Flask server (`GpServer`) that Protocol Studio starts automatically at
campaign launch. The server manages the GP lifecycle and exposes a `PSEPointService` that
handles broker commands.

The Flask server is started by the Streamlit autocontrol GUI. PSE does not start its own
campaign — it waits for a `configure` command from Protocol Studio.

**GP state files** (in `storage_path/`):

| File | Contents |
|---|---|
| `results.json` | All iteration parameters, results, controls per channel |
| `gp_state.pkl` | Serialised gpCAM hyperparameters and training data |
| `plots/` | Posterior contour and 1D optimum plots (PNG/PDF) |

---

## Getting Started

```bash
# PSE starts automatically when Protocol Studio launches a campaign.
# To start the Flask GP server manually:
python -m pse.gp_server
```

PSE requires RabbitMQ to be running. The `PSEPointService` is only active while a
campaign is configured — between campaigns the server is idle.

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `AMQP_URL` | `amqp://roadbot:roadbot_dev@localhost/` | RabbitMQ connection |
| `PSE_PORT` | `5050` | Flask GP server port |
| `PSE_STORAGE_PATH` | — | Root directory for GP state files and results |
