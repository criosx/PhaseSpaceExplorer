# PhaseSpaceExplorer — Refactor Context for AI Assistant

## 1. Core Responsibility
`PhaseSpaceExplorer` is a Bayesian Gaussian Process (gpCAM) optimization engine that adaptively selects the next experiment parameters to maximize information gain: a Flask `GpServer` manages the GP lifecycle via HTTP, a background thread runs the optimization loop, and `ROADMAP_Gp.do_measurement()` dispatches physical sample prep + QCMD measurement jobs to `lh_manager` via the synchronous `ManagerClient`, blocks until results return, reduces QCMD frequency data to a scalar, and feeds the result back to the GP to propose the next point.

---

## 2. Physical Constraints (Optimization Loop Rules)

- **Multi-channel parallel measurements — channel is first-available, not sticky.** `ROADMAP_Gp.n_channels` channels (set by `parallel_measurements`) run concurrently. `do_measurement()` picks the first non-busy channel. Channel affinity enforcement is downstream in autocontrol — PSE just picks which one is free and records a per-channel count for control cycling.
- **Control measurement cycling.** Every `control_cycle` measurements per channel (default 4), a new buffer baseline is collected and stored in `self.controls[channel]`. All subsequent QCMD results for that channel are normalized against the most recent control. This state is persisted to `results.json` and must survive restarts (resume logic reads `load_controls()` on init).
- **Measurement results are UNCERTAIN error domain.** `ROADMAP_QCMD_MakeBilayer` physically deposits lipids on a sensor. A `do_measurement` failure (None result) means sample state is unknown — the GP records None and moves on, but hardware must not be auto-retried. Human intervention is required to clear the channel.
- **The GP blocks on measurement completion.** `manager.wait_for_result()` is a synchronous polling call on the lh_manager REST API. After the broker refactor, this blocking wait becomes a subscription to `Task.Completed` events. The GP thread cannot proceed to the next iteration until the result is received.
- **GP state is serialized to disk for resume.** Hyperparameters, training dataset, and iteration count are pickled/JSON-saved to `storage_path/`. On restart with `resume=True`, the GP rehydrates this state. The broker refactor must not break the resume path.
- **Tight import coupling to lh_manager.** `roadmap.py` imports `lh_manager` Python classes directly (`ManagerClient`, method classes, `Composition`, `SoluteFormulation`, etc.). This is the primary coupling to replace with broker publish messages — after the refactor, PSE publishes `Command.SubmitSample` and subscribes to `Task.Completed` instead of calling `ManagerClient` directly.

---

## 3. Deprecated REST Endpoints

### GpServer Flask endpoints (inbound — will become broker commands)
- `GET /` — health check
- `GET /get_status` — returns `{progress, cancelled, paused}` polled by GUI
- `POST /start_pse` — starts a new GP exploration; body: Gp init kwargs (`exp_par`, `storage_path`, `gpcam_iterations`, `parallel_measurements`, `client`, etc.)
- `GET /stop_pse` — stops the exploration and shuts down hardware
- `GET /pause_pse` — pauses (joins the background thread)
- `POST /resume_pse` — resumes a paused exploration; body: updated Gp init kwargs

### Outbound (PSE → lh_manager — will become broker publish)
- `ManagerClient.new_sample()` → `POST /GUI/AddSample/`
- `ManagerClient.update_sample()` → `POST /GUI/UpdateSample/`
- `ManagerClient.run_sample()` → `POST /GUI/RunSample/`
- `ManagerClient.wait_for_result()` → polls `GET /GUI/GetSampleStatus/` (blocking)
- `ManagerClient.get_layout()` → `GET /GUI/GetLayout/`

---

## 4. Future Telemetry Needs

### State change events (publish on change)
- `Exploration.Started` — GP study initiated; payload: `{study_id, n_iterations, n_channels, parameter_names}`
- `Exploration.PointDispatched` — next point submitted to lh_manager; payload: `{study_id, iteration, parameters, channel}`
- `Exploration.PointCompleted` — measurement result fed to GP; payload: `{study_id, iteration, result, variance, channel}`
- `Exploration.ModelUpdated` — GP hyperparameters retrained; payload: `{study_id, iteration}`
- `Exploration.Completed` — all iterations finished; payload: `{study_id, best_parameters, best_value}`
- `Exploration.Paused` / `Exploration.Resumed` / `Exploration.Stopped`

### Inbound broker subscriptions (replace outbound HTTP)
- Subscribe to `Task.Completed` / `Task.Failed` from lh_manager → replaces `manager.wait_for_result()` blocking poll

### Large data payloads (Claim Check — do not carry in broker message body)
- **GP model state** — hyperparameters + full training dataset; pickled to `storage_path/`; reference by `study_id`. Never in broker message.
- **Full results JSON** — all iteration parameters, raw QCMD data, reduced scalars; saved to `storage_path/results/results.json`. Broker carries only scalar summary per point.
- **GP posterior plots** — PNG/PDF contour and 1D plots in `storage_path/plots/`; never in broker.
