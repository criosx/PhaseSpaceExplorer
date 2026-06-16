"""Broker integration for PhaseSpaceExplorer — Phase 2 Step C.

PSE is now a passive point-generator service.  Protocol Studio drives the
optimization loop; PSE responds to inbound commands and maintains the GP model.

Inbound (exchange.protocol, queue: pse.commands — durable):
  command.pse.configure        — PS sends parameter space + optimizer config; starts service
  command.pse.request_point    — PS asks for the next measurement point
  command.pse.submit_result    — PS reports a completed trial result
  command.pse.cancel_trial     — PS cancels an in-flight trial (no model update)
  command.pse.notify_in_flight — PS re-populates in-flight list after PSE restart

Outbound (exchange.protocol):
  pse.ready           — published on startup and whenever set_service() is called
  pse.point_suggested — the suggested next point, with echoed caller context
  pse.exploration.*   — telemetry events (unchanged)

Threading model
---------------
PSEBrokerWorker runs an asyncio event loop in a daemon thread.  CPU-heavy GP
operations (ask, tell, train) are offloaded to asyncio.to_thread() so they
never block the event loop.  PSEPointService uses a threading.Lock to serialize
access to the GPOptimizer.
"""

import asyncio
import itertools
import logging
import os
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aio_pika
import numpy as np
import pandas as pd
from gpcam import GPOptimizer

from roadmap_broker_client.connection import get_connection
from roadmap_broker_client.consumer import consume
from roadmap_broker_client.envelope import Envelope, build
from roadmap_broker_client.publisher import publish
from roadmap_broker_client.topology import declare_topology
from roadmap_broker_client.topics import (
    CMD_PSE_ANNOUNCE_REQUEST,
    CMD_PSE_CANCEL_TRIAL,
    CMD_PSE_CONFIGURE,
    CMD_PSE_NOTIFY_IN_FLIGHT,
    CMD_PSE_REQUEST_POINT,
    CMD_PSE_STOP_CAMPAIGN,
    CMD_PSE_SUBMIT_RESULT,
    PROTOCOL_EXCHANGE,
    PSE_EXPLORATION_COMPLETED,
    PSE_EXPLORATION_POINT_COMPLETED,
    PSE_EXPLORATION_POINT_DISPATCHED,
    PSE_EXPLORATION_STARTED,
    PSE_EXPLORATION_STOPPED,
    PSE_POINT_SUGGESTED,
    PSE_READY,
    command_key,
    command_subscription_pattern,
)

from pse.gp import Gp

logger = logging.getLogger(__name__)

_READY_TIMEOUT = 30.0  # seconds to wait for initial broker connection


# ---------------------------------------------------------------------------
# Acquisition function
# ---------------------------------------------------------------------------

def acq_variance_target(x: np.ndarray, gpoptimizer: GPOptimizer) -> np.ndarray:
    """Variance / (mean + offset)^2 — balances exploration against regions of
    high expected signal.  Tolerance term prevents division by zero near zero mean."""
    tolerance = 5
    var = np.array(gpoptimizer.posterior_covariance(x, variance_only=True)["v(x)"])
    mean = np.array(gpoptimizer.posterior_mean(x)["m(x)"])
    return var / ((mean + 25) ** 2 + tolerance ** 2)


# ---------------------------------------------------------------------------
# Acquisition function registry
#
# Maps the string name sent in command.pse.configure (acq_func field) to the
# value that GPOptimizer.ask(acquisition_function=...) expects.
# Values are either a Python callable (custom) or a plain string (gpCAM built-in).
#
# TODO: refactor to accept parameters (e.g. kappa for ucb/lcb, target for
# target_probability).  The configure payload should carry an optional
# acq_func_params dict forwarded to GPOptimizer.ask().  The registry would
# become a registry of factory functions rather than bare callables.
# ---------------------------------------------------------------------------

ACQUISITION_FUNCTIONS: Dict[str, Any] = {
    "variance_target":                   acq_variance_target,
    "variance":                          "variance",
    "ucb":                               "ucb",
    "lcb":                               "lcb",
    "maximum":                           "maximum",
    "minimum":                           "minimum",
    "gradient":                          "gradient",
    "total correlation":                 "total correlation",
    "expected improvement":              "expected improvement",
    "probability of improvement":        "probability of improvement",
    "relative information entropy":      "relative information entropy",
    "relative information entropy set":  "relative information entropy set",
    "target probability":                "target probability",
}


# ---------------------------------------------------------------------------
# PSEPointService
# ---------------------------------------------------------------------------

class PSEPointService(Gp):
    """Passive GP service.  Responds to inbound broker commands instead of
    driving the optimization loop internally.

    Lifecycle:
        service = PSEPointService(**gp_kwargs)
        service.initialize()        # builds GPOptimizer, loads prior data
        broker.set_service(service) # makes PSE available to callers

    Thread safety: all public methods acquire self._lock before touching
    GPOptimizer state, so concurrent broker command handlers are safe.
    """

    def __init__(self, constraints: Optional[List[Dict]] = None,
                 downsample_seed: Optional[int] = None, **gp_kwargs) -> None:
        self._in_flight: Dict[str, np.ndarray] = {}  # trial_id → position vector
        self._lock = threading.Lock()
        self._paused: bool = False
        self._constraints: List[Dict] = constraints or []
        self._downsample_seed: Optional[int] = downsample_seed
        self.feasible_point_count: Optional[int] = None
        super().__init__(**gp_kwargs)
        # Translate the acq_func string stored by Gp.__init__ into the callable or
        # built-in string that GPOptimizer.ask() expects.  Unknown names fall back
        # to acq_variance_target so existing callers that omit acq_func are unaffected.
        self._acq_func_name: str = self.acq_func  # preserve string name for pse_context
        self.acq_func = ACQUISITION_FUNCTIONS.get(self.acq_func, acq_variance_target)
        # task_dict used by gp_server for status reporting
        self.task_dict = {"cancelled": False, "paused": False,
                          "progress": "0%", "status": "running"}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize the service.  For gpcam, builds the GPOptimizer.
        For grid, the axes are already set up by Gp.__init__; nothing extra needed.
        Call once after construction, before registering with the broker worker.
        """
        if self.optimizer == "gpcam":
            if self._constraints:
                feasible = self._build_feasible_space()
                self.gp_discrete_points = feasible
                self.feasible_point_count = len(feasible) if feasible is not None else 0
                if feasible is not None:
                    # gp_evaluation_points is built by Gp.__init__ from the full unconstrained
                    # meshgrid. Overwrite it so prediction and plotting stay within the
                    # feasible space (mirrors what Gp.__init__ does when discrete points
                    # are passed at construction time).
                    self.gp_evaluation_points = np.array(feasible)
            elif self.gp_discrete_points is not None:
                self.gp_discrete_points = self._filter_discrete_points()
            else:
                # No constraints, no discrete points — full unconstrained meshgrid.
                # Downsample if it exceeds the GP evaluation point limit.
                n_total = len(self.gp_evaluation_points)
                if n_total > self._MAX_GROUP_POINTS:
                    rng = np.random.default_rng(self._downsample_seed)
                    idx = rng.choice(n_total, self._MAX_GROUP_POINTS, replace=False)
                    self.gp_evaluation_points = self.gp_evaluation_points[idx]
                    logger.info(
                        "PSEPointService: unconstrained grid downsampled from %d to %d points.",
                        n_total, len(self.gp_evaluation_points),
                    )
                self.feasible_point_count = len(self.gp_evaluation_points)
            self.gpcam_init_ae()
        logger.info(
            "PSEPointService initialized: optimizer=%s, n_params=%d, feasible=%s",
            self.optimizer,
            len(self.exp_par),
            self.feasible_point_count,
        )

    # ------------------------------------------------------------------
    # Constraint filtering
    # ------------------------------------------------------------------

    _MAX_GROUP_POINTS = 40_000

    def _filter_discrete_points(self) -> list:
        """Return gp_discrete_points unchanged (legacy path, no constraints)."""
        return self.gp_discrete_points

    def _build_feasible_space(self) -> Optional[list]:
        """Build the feasible discrete point space from linear constraints.

        Algorithm:
        1. Parse self._constraints → A (k×n_opt), b (k,) where op '>=','==' are normalised to '<='.
        2. Union-find over parameter names in constraint terms → connected components (groups).
        3. Per group: generate sub-grid from exp_par bounds, coarsen (halve step counts)
           until product < _MAX_GROUP_POINTS, then filter by group-relevant constraints.
        4. Cartesian product across groups → if total > _MAX_GROUP_POINTS, subsample
           each group proportionally using self._downsample_seed for reproducibility.
        5. Return list of 1-D numpy arrays in exp_par (optimizable) parameter order.
        """
        # Gp.__init__ already filters exp_par to optimizable parameters only —
        # the 'optimize' column is absent by this point.  Use all rows as-is.
        opt_rows = list(self.exp_par.itertuples())
        n_opt = len(opt_rows)
        if n_opt == 0 or not self._constraints:
            return None

        name_to_idx = {row.name: i for i, row in enumerate(opt_rows)}

        # --- Build A, b ---
        A_rows: List[np.ndarray] = []
        b_vals: List[float] = []
        for c in self._constraints:
            terms = c.get("terms", {})
            op = c.get("op", "<=")
            rhs = float(c.get("rhs", 0.0))
            coeffs = np.zeros(n_opt)
            for name, coeff in terms.items():
                if name in name_to_idx:
                    coeffs[name_to_idx[name]] = float(coeff)
            if op == "<=":
                A_rows.append(coeffs)
                b_vals.append(rhs)
            elif op == ">=":
                A_rows.append(-coeffs)
                b_vals.append(-rhs)
            elif op == "==":
                A_rows.append(coeffs.copy())
                b_vals.append(rhs)
                A_rows.append(-coeffs.copy())
                b_vals.append(-rhs)

        if not A_rows:
            return None

        A = np.array(A_rows)   # (k, n_opt)
        b = np.array(b_vals)   # (k,)

        # --- Union-find grouping ---
        parent = list(range(n_opt))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            px, py = _find(x), _find(y)
            if px != py:
                parent[px] = py

        for row_vec in A_rows:
            nonzero = [i for i in range(n_opt) if abs(row_vec[i]) > 1e-12]
            for j in nonzero[1:]:
                _union(nonzero[0], j)

        groups_map: Dict[int, List[int]] = defaultdict(list)
        for i in range(n_opt):
            groups_map[_find(i)].append(i)
        group_list = list(groups_map.values())

        # Map A rows to their group root (or -1 for constant rows)
        row_group_root = []
        for row_vec in A_rows:
            nonzero = [i for i in range(n_opt) if abs(row_vec[i]) > 1e-12]
            row_group_root.append(_find(nonzero[0]) if nonzero else -1)

        # --- Per-group grid + filter ---
        feasible_groups: List[tuple] = []
        for group_indices in group_list:
            root = _find(group_indices[0])
            relevant = [
                ri for ri, g in enumerate(row_group_root)
                if g == root or g == -1
            ]
            A_g = A[np.ix_(relevant, group_indices)] if relevant else np.empty((0, len(group_indices)))
            b_g = b[relevant] if relevant else np.empty(0)

            step_counts, lows, highs = [], [], []
            for idx in group_indices:
                row = opt_rows[idx]
                lo = float(row.lower_opt)
                hi = float(row.upper_opt)
                step = float(row.step_opt)
                n = max(2, round((hi - lo) / step) + 1) if hi > lo and step > 0 else 2
                step_counts.append(n)
                lows.append(lo)
                highs.append(hi)

            while int(np.prod(step_counts)) > self._MAX_GROUP_POINTS:
                step_counts = [max(2, (n + 1) // 2) for n in step_counts]

            axes = [np.linspace(lo, hi, n) for lo, hi, n in zip(lows, highs, step_counts)]
            grid = np.array(list(itertools.product(*axes)))   # (M, group_dim)

            if A_g.shape[0] > 0 and grid.size > 0:
                mask = np.all(grid @ A_g.T <= b_g, axis=1)
                grid = grid[mask]

            if len(grid) == 0:
                logger.warning(
                    "_build_feasible_space: group %s has no feasible points — constraints may be infeasible.",
                    group_indices,
                )

            feasible_groups.append((group_indices, grid))

        # --- Cross-group Cartesian product ---
        group_grids = [fg[1] for fg in feasible_groups]
        group_idx_lists = [fg[0] for fg in feasible_groups]

        sizes = [len(g) for g in group_grids]
        n_total = int(np.prod(sizes)) if sizes else 0

        if n_total > self._MAX_GROUP_POINTS:
            rng = np.random.default_rng(self._downsample_seed)
            target = max(2, int(np.floor(self._MAX_GROUP_POINTS ** (1.0 / max(len(group_grids), 1)))))
            subsampled = []
            for grid in group_grids:
                n_s = min(len(grid), target)
                idx = rng.choice(len(grid), n_s, replace=False)
                subsampled.append(grid[idx])
            group_grids = subsampled

        points: List[np.ndarray] = []
        for combo in itertools.product(*[range(len(g)) for g in group_grids]):
            full_vec = np.zeros(n_opt)
            for g_pos, pt_idx in enumerate(combo):
                for col, param_idx in enumerate(group_idx_lists[g_pos]):
                    full_vec[param_idx] = group_grids[g_pos][pt_idx, col] if group_grids[g_pos].ndim > 1 else group_grids[g_pos][pt_idx]
            points.append(full_vec)

        logger.info(
            "_build_feasible_space: %d feasible points from %d groups (%d params).",
            len(points), len(group_list), n_opt,
        )
        return points if points else None

    # ------------------------------------------------------------------
    # Public GP interface (called from PSEBrokerWorker via asyncio.to_thread)
    # ------------------------------------------------------------------

    def request_point(self) -> tuple:
        """Ask for the next measurement point.

        For gpcam: uses the GP acquisition function (phantom-tell to block the
        point for concurrent callers).
        For grid: steps sequentially through the Cartesian product of axes,
        skipping points already in-flight.

        Returns:
            (trial_id, params_dict) where params_dict maps parameter name → value.

        Raises:
            RuntimeError: if initialize() has not been called.
        """
        with self._lock:
            trial_id = str(uuid4())

            if self.optimizer == "grid":
                grid = list(itertools.product(*self.axes))
                in_flight_set = set(
                    tuple(v.tolist()) for v in self._in_flight.values()
                )
                # Find the next grid point not already in-flight.
                start = self.gpiteration % len(grid)
                next_point = None
                for offset in range(len(grid)):
                    candidate = np.array(grid[(start + offset) % len(grid)])
                    if tuple(candidate.tolist()) not in in_flight_set:
                        next_point = candidate
                        break
                if next_point is None:
                    next_point = np.array(grid[start % len(grid)])

            else:
                if self.my_ae is None:
                    raise RuntimeError(
                        "PSEPointService.initialize() must be called before request_point()."
                    )

                if len(self.gpCAMstream) < self.gpcam_init_dataset_size:
                    # Initial dataset phase: GP has no tell() yet, so ask() would fail.
                    # Pick randomly from the feasible set (gp_discrete_points if constraints
                    # were applied, otherwise gp_evaluation_points), avoiding in-flight points.
                    init_pool = (
                        self.gp_discrete_points
                        if self.gp_discrete_points is not None
                        else self.gp_evaluation_points
                    )
                    in_flight_set = set(
                        tuple(v.tolist()) for v in self._in_flight.values()
                    )
                    next_point = None
                    for _ in range(20):
                        idx = np.random.randint(len(init_pool))
                        candidate = np.array(init_pool[idx])
                        if tuple(candidate.tolist()) not in in_flight_set:
                            next_point = candidate
                            break
                    if next_point is None:
                        next_point = np.array(
                            init_pool[np.random.randint(len(init_pool))]
                        )
                    # No phantom-tell here: the GP is uninitialised and posterior_mean
                    # would also fail.  In-flight tracking via _in_flight is enough.
                else:
                    if self.gp_discrete_points is not None:
                        input_set = self.gp_discrete_points
                    else:
                        input_set = np.array(
                            [(row.lower_opt, row.upper_opt)
                             for row in self.exp_par.itertuples()]
                        )

                    ask_result = self.my_ae.ask(
                        input_set=input_set,
                        n=1,
                        method="global",
                        acquisition_function=self.acq_func,
                        info=True,
                    )
                    next_point = np.array(ask_result["x"][0])
                    self._phantom_tell(next_point)

            params = {
                row.name: float(next_point[i])
                for i, row in enumerate(self.exp_par.itertuples())
            }

            self._in_flight[trial_id] = next_point
            self.gpiteration += 1
            self.iterations_inprogress_save_to_file()

            # Build pse_context for dm_worker trial records.
            optimizer = getattr(self, "optimizer", "gpcam")
            n_training = len(self.gpCAMstream) if optimizer == "gpcam" else 0
            uncertainty: Optional[float] = None
            if (optimizer == "gpcam"
                    and getattr(self, "my_ae", None) is not None
                    and n_training >= self.gpcam_init_dataset_size):
                try:
                    v = self.my_ae.posterior_covariance(
                        next_point.reshape(1, -1), variance_only=True
                    )["v(x)"]
                    uncertainty = float(np.atleast_1d(v)[0])
                except Exception:
                    pass
            pse_context = {
                "acq_func": getattr(self, "_acq_func_name", "unknown"),
                "training_set_size": n_training,
                "uncertainty_at_suggestion": uncertainty,
            }

            logger.info("request_point → trial %s  params=%s", trial_id, params)
            return trial_id, params, pse_context

    def submit_result(
        self, trial_id: str, value: float, variance: Optional[float]
    ) -> None:
        """Record a completed trial result and update the model.

        For gpcam: updates the GP, re-applies phantom tells, trains if enough data.
        For grid: writes value into self.results at the grid position; no GP update.

        Args:
            trial_id:  UUID returned by a prior request_point call.
            value:     Scalar measurement result.
            variance:  Measurement variance (None → use a small default).
        """
        with self._lock:
            position = self._in_flight.pop(trial_id, None)
            if position is None:
                logger.warning(
                    "submit_result for unknown trial_id %s — ignoring.", trial_id
                )
                return

            variance = (
                float(variance)
                if variance is not None
                else self.signal_estimate * 1e-6
            )

            if self.optimizer == "grid":
                # Find the grid index for this position and store the result.
                grid = list(itertools.product(*self.axes))
                pos_tuple = tuple(float(v) for v in position)
                try:
                    idx = next(
                        i for i, pt in enumerate(grid)
                        if all(abs(a - b) < 1e-9 for a, b in zip(pt, pos_tuple))
                    )
                    nd_idx = list(np.ndindex(*self.steplist))[idx]
                    self.results[nd_idx] = float(value)
                    self.variances[nd_idx] = variance
                    self.n_iter[nd_idx] += 1
                except StopIteration:
                    logger.warning("submit_result: position %s not found in grid.", position)
                n = int(np.sum(~np.isnan(self.results)))
                progress = min(n / max(self.results.size, 1), 1.0)
                self.task_dict["progress"] = f"{progress * 100:.2f}%"
                self.results_io()
                self.iterations_inprogress_save_to_file()
                logger.info("submit_result (grid): trial %s  value=%.4g  n=%d", trial_id, value, n)
                return

            # gpcam path
            self.gpCAMstream.loc[len(self.gpCAMstream)] = {
                "parameter names": self.exp_par["name"].to_list(),
                "position": position,
                "value": float(value),
                "variance": variance,
                "mutual information": None,
                "itlabel": len(self.gpCAMstream),
            }

            # Re-initialize with only real data (removes phantom points).
            self.gpcam_init_ae(just_gpcamstream=True)

            # Re-add phantoms for any remaining in-flight trials.
            for pos in self._in_flight.values():
                self._phantom_tell(pos)

            n = len(self.gpCAMstream)
            if n >= self.gpcam_init_dataset_size:
                try:
                    train_every = self.train_global_every or 1
                    method = "global" if n % train_every == 0 else "local"
                    self.task_dict["status"] = "retraining"
                    self.gpcam_train(method=method)
                    self.gpcam_prediction()
                    hypars = self.my_ae.get_hyperparameters().tolist()
                    for i, val in enumerate(hypars):
                        self.gpCAMstream.at[self.gpCAMstream.index[-1], f"hy{i}"] = val
                    self.gpCAMstream.at[self.gpCAMstream.index[-1], "mutual information"] = self.mutual_information_gpcam
                    self.plot_results()
                except Exception:
                    logger.exception(
                        "submit_result: training/plotting failed for trial %s (data still saved)",
                        trial_id,
                    )
                finally:
                    self.task_dict["status"] = "running"

            progress = min(n / max(self.gpcam_iterations, 1), 1.0)
            self.task_dict["progress"] = f"{progress * 100:.2f}%"

            self.results_io()
            self.iterations_inprogress_save_to_file()
            logger.info(
                "submit_result: trial %s  value=%.4g  n=%d", trial_id, value, n
            )

    def cancel_trial(self, trial_id: str) -> None:
        """Discard an in-flight trial without updating the GP model."""
        with self._lock:
            position = self._in_flight.pop(trial_id, None)
            if position is None:
                logger.debug("cancel_trial for unknown trial_id %s — ignoring.", trial_id)
                return
            # Rebuild phantom tells without this point.
            self.gpcam_init_ae(just_gpcamstream=True)
            for pos in self._in_flight.values():
                self._phantom_tell(pos)
            self.iterations_inprogress_save_to_file()
            logger.info("cancel_trial: trial %s discarded.", trial_id)

    def notify_in_flight(self, trial_id: str, params: dict) -> None:
        """Re-populate an in-flight entry after PSE restart.

        PS calls this for every trial that was in-flight when PSE crashed.
        PSE does not re-ask the GP; it just registers the position so
        submit_result / cancel_trial can find it later.
        """
        with self._lock:
            position = np.array(
                [params[row.name] for row in self.exp_par.itertuples()]
            )
            self._in_flight[trial_id] = position
            if self.optimizer == "gpcam":
                # Rebuild phantom tells to include the newly registered position.
                self.gpcam_init_ae(just_gpcamstream=True)
                for pos in self._in_flight.values():
                    self._phantom_tell(pos)
            logger.info(
                "notify_in_flight: trial %s re-registered (%d total in-flight).",
                trial_id, len(self._in_flight),
            )

    def retrain(self, data_points: list) -> None:
        """Replace the entire GP training set with the provided data points.

        Used when PS excludes a trial from the dataset: PS sends the full set
        of included completed trials so PSE can rebuild without the bad point.

        Each entry in data_points must have:
            parameters: dict[name, value]  — parameter values keyed by name
            value:      float              — scalar result
            variance:   float | None       — measurement variance

        In-flight state (_in_flight) is preserved so concurrent trials are
        not disrupted.  Phantom tells are rebuilt after the GP is reset.
        Grid optimizer has no GP to retrain — this is a no-op for it.
        """
        if self.optimizer != "gpcam":
            logger.debug("retrain: no-op for optimizer=%r", self.optimizer)
            return

        with self._lock:
            in_flight = dict(self._in_flight)

            # Rebuild gpCAMstream from the provided included set.
            self.gpCAMstream = self.gpCAMstream.iloc[0:0].copy()
            for dp in data_points:
                params = dp.get("parameters", {})
                position = np.array(
                    [params.get(row.name, 0.0) for row in self.exp_par.itertuples()]
                )
                variance = float(dp.get("variance") or self.signal_estimate * 1e-6)
                self.gpCAMstream.loc[len(self.gpCAMstream)] = {
                    "parameter names": self.exp_par["name"].to_list(),
                    "position": position,
                    "value": float(dp["value"]),
                    "variance": variance,
                    "mutual information": None,
                    "itlabel": len(self.gpCAMstream),
                }

            # Re-initialize GP on real data only (removes any prior phantom points).
            self.gpcam_init_ae(just_gpcamstream=True)

            # Restore in-flight state and rebuild phantom tells.
            self._in_flight = in_flight
            for pos in self._in_flight.values():
                self._phantom_tell(pos)

            n = len(self.gpCAMstream)
            if n >= self.gpcam_init_dataset_size:
                try:
                    self.task_dict["status"] = "retraining"
                    self.gpcam_train(method="global")
                    self.gpcam_prediction()
                    self.plot_results()
                except Exception:
                    logger.exception("retrain: training/plotting failed (data saved)")
                finally:
                    self.task_dict["status"] = "running"

            self.results_io()
            self.iterations_inprogress_save_to_file()
            logger.info(
                "retrain: GP rebuilt from %d included data points (%d in-flight).",
                n, len(in_flight),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def iterations_inprogress_save_to_file(self) -> None:
        """Override: build the in-progress pickle from _in_flight instead of
        measurement_inprogress (which is only used by the old loop path)."""
        import pickle
        from os import path
        rows = []
        for trial_id, position in self._in_flight.items():
            row = {row.name: float(position[i]) for i, row in enumerate(self.exp_par.itertuples())}
            row["trial_id"] = trial_id
            rows.append(row)
        output_df = pd.DataFrame(rows)
        with open(path.join(self.spath, "results", "current_iterations.pkl"), "wb") as f:
            pickle.dump(output_df, f)

    def _gp_has_data(self) -> bool:
        """True once the GP has been tell()-initialised with at least one real measurement."""
        return len(self.gpCAMstream) >= 1

    def _phantom_tell(self, point: np.ndarray) -> None:
        """Tell the optimizer about an in-flight point using its posterior mean
        as a placeholder result.  This prevents duplicate suggestions when
        multiple channels request points concurrently.

        No-op during the initial dataset phase when the GP has not yet been
        tell()-initialised (posterior_mean would raise AttributeError).
        """
        if not self._gp_has_data():
            return
        pts = point.reshape(1, -1)
        pred_mean = np.atleast_1d(self.my_ae.posterior_mean(pts)["m(x)"])
        pred_var = np.atleast_1d([self.signal_estimate * 1e-7])
        self.my_ae.tell(pts, pred_mean, pred_var, append=True)


# ---------------------------------------------------------------------------
# PSEBrokerWorker
# ---------------------------------------------------------------------------

class PSEBrokerWorker:
    """Asyncio broker consumer/publisher running in a daemon thread.

    Subscribes to command.pse.# and dispatches to PSEPointService methods.
    Publishes pse.ready on startup and whenever set_service() is called.

    Wire in gp_server.py:
        self._broker = PSEBrokerWorker()
        self._broker.start()
        ...
        self._broker.set_service(service)  # when a PSE run begins
        self._broker.set_service(None)     # when a PSE run ends
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._exchange: Optional[aio_pika.abc.AbstractExchange] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._service: Optional[PSEPointService] = None
        self._on_service_changed = None  # Optional[Callable[[Optional[PSEPointService]], None]]

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
            await channel.set_qos(prefetch_count=1)
            await declare_topology(channel)

            self._exchange = await channel.get_exchange(PROTOCOL_EXCHANGE)

            # Publish pse.ready so PS knows PSE is available (but no service yet).
            await self._emit(PSE_READY, {"has_service": False, **self._capabilities_payload()})
            self._ready.set()

            # Durable command queue — survives PSE restarts; in-flight commands
            # are redelivered.
            cmd_queue = await channel.declare_queue(
                "pse.commands",
                durable=True,
                arguments={"x-dead-letter-exchange": "exchange.dead_letter"},
            )
            await cmd_queue.bind(
                self._exchange,
                routing_key=command_subscription_pattern("pse"),
            )

            logger.info("PSE broker worker running.")
            await consume(cmd_queue, self._on_command)

    # ------------------------------------------------------------------
    # Service registration
    # ------------------------------------------------------------------

    def set_service(self, service: Optional[PSEPointService]) -> None:
        """Register (or deregister) the active PSEPointService.

        Publishing pse.ready on registration prompts Protocol Studio to replay
        notify_in_flight for any trials that were in-flight when PSE restarted.
        """
        with self._lock:
            self._service = service
        if self._on_service_changed is not None:
            self._on_service_changed(service)
        if self._ready.is_set() and self._loop is not None:
            asyncio.run_coroutine_threadsafe(
                self._emit(PSE_READY, {"has_service": service is not None,
                                       **self._capabilities_payload()}),
                self._loop,
            )

    # ------------------------------------------------------------------
    # Inbound command dispatch
    # ------------------------------------------------------------------

    async def _on_command(
        self, envelope: Envelope, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        rk = message.routing_key or ""
        payload = envelope.payload or {}

        if rk == command_key("pse", CMD_PSE_CONFIGURE):
            await self._handle_configure(payload)
        elif rk == command_key("pse", CMD_PSE_REQUEST_POINT):
            await self._handle_request_point(payload)
        elif rk == command_key("pse", CMD_PSE_SUBMIT_RESULT):
            await self._handle_submit_result(payload)
        elif rk == command_key("pse", CMD_PSE_CANCEL_TRIAL):
            await self._handle_cancel_trial(payload)
        elif rk == command_key("pse", CMD_PSE_NOTIFY_IN_FLIGHT):
            await self._handle_notify_in_flight(payload)
        elif rk == command_key("pse", CMD_PSE_STOP_CAMPAIGN):
            await self._handle_stop_campaign(payload)
        elif rk == command_key("pse", CMD_PSE_ANNOUNCE_REQUEST):
            await self._handle_announce_request(payload)
        else:
            logger.debug("Ignoring unknown PSE command: %s", rk)

    async def _handle_stop_campaign(self, payload: dict) -> None:
        """PS signals that the campaign is over (max_steps reached or manual stop).

        Deregisters the service immediately (so the monitor flips to idle and PS
        stops sending request_point), then flushes data and plots in the background.
        set_service(None) publishes pse.ready(has_service=False).  PSE's disk files
        are a write-through cache only — PS's DB is the authoritative record.
        """
        campaign_id = payload.get("campaign_id", "")
        final_status = payload.get("final_status", "stopped")

        with self._lock:
            service = self._service

        if service is None:
            logger.info("stop_campaign for %s: no active service to finalize.", campaign_id)
            return

        logger.info("Stopping PSE service for campaign %s (%s).", campaign_id, final_status)

        # Deregister first so the monitor shows idle and PS stops sending commands.
        self.set_service(None)

        def _finalize() -> None:
            service.results_io()
            service.iterations_inprogress_save_to_file()
            if service.optimizer == "gpcam":
                service.plot_results()

        try:
            await asyncio.to_thread(_finalize)
        except Exception:
            logger.exception("PSE finalization failed for campaign %s.", campaign_id)

        await self._emit(PSE_EXPLORATION_STOPPED, {
            "campaign_id": campaign_id,
            "final_status": final_status,
        })
        logger.info("PSE service finalized for campaign %s.", campaign_id)

    async def _handle_configure(self, payload: dict) -> None:
        """Construct and initialize a new PSEPointService from the broker payload.

        Called when PS sends command.pse.configure in response to start_campaign.
        Replaces any existing service.  Publishes pse.ready (has_service=True)
        on success so PS knows it can start sending request_point commands.
        """
        parameter_space = payload.get("parameter_space")
        if not parameter_space:
            logger.warning("configure received with empty parameter_space — ignoring.")
            return

        campaign_id = payload.get("campaign_id", "")

        # Defensively deregister any existing service before creating a new one.
        # Under normal operation the prior campaign's stop_campaign is always
        # processed first (queue is sequential), so this is a safety net only.
        with self._lock:
            existing = self._service
        if existing is not None:
            logger.warning(
                "configure for campaign %s: replacing active service without prior stop_campaign.",
                campaign_id,
            )
            self.set_service(None)
        optimizer = payload.get("optimizer", "gpcam")
        storage_path = payload.get("storage_path") or f"pse_runs/{campaign_id}"
        os.makedirs(storage_path, exist_ok=True)

        acq_func = payload.get("acq_func", "variance_target")
        if acq_func not in ACQUISITION_FUNCTIONS:
            logger.warning(
                "configure: unknown acq_func %r — falling back to 'variance_target'.", acq_func
            )
            acq_func = "variance_target"

        exp_par = pd.DataFrame(parameter_space)

        constraints: List[Dict] = payload.get("constraints") or []
        downsample_seed: Optional[int] = payload.get("downsample_seed")

        service = PSEPointService(
            constraints=constraints,
            downsample_seed=downsample_seed,
            exp_par=exp_par,
            optimizer=optimizer,
            acq_func=acq_func,
            storage_path=storage_path,
            project_name=payload.get("project_name") or campaign_id,
            gpcam_iterations=int(payload.get("gpcam_iterations", 50)),
            gpcam_init_dataset_size=int(payload.get("gpcam_init_dataset_size", 10)),
            train_global_every=payload.get("train_global_every"),
            signal_estimate=float(payload.get("signal_estimate", 10.0)),
            resume=bool(payload.get("resume", True)),
        )
        try:
            await asyncio.to_thread(service.initialize)
            await asyncio.to_thread(service.iterations_inprogress_save_to_file)
        except Exception:
            logger.exception(
                "PSEPointService.initialize() failed for campaign %s.", campaign_id
            )
            return

        with self._lock:
            self._service = service
        if self._on_service_changed is not None:
            self._on_service_changed(service)

        ready_payload: dict = {
            "has_service": True,
            "campaign_id": campaign_id,
            **self._capabilities_payload(),
        }
        if service.feasible_point_count is not None:
            ready_payload["feasible_point_count"] = service.feasible_point_count

        await self._emit(PSE_READY, ready_payload)
        logger.info(
            "PSE configured for campaign %s (optimizer=%s, acq_func=%s, params=%d, feasible=%s).",
            campaign_id, optimizer, acq_func, len(parameter_space), service.feasible_point_count,
        )

    async def _handle_request_point(self, payload: dict) -> None:
        with self._lock:
            service = self._service
        if service is None:
            logger.warning("request_point received but no PSE service is running.")
            return
        if service._paused:
            logger.info("request_point received but PSE is paused — ignoring.")
            return

        try:
            trial_id, params, pse_context = await asyncio.to_thread(service.request_point)
        except Exception:
            logger.exception("PSEPointService.request_point() failed.")
            return

        # Echo back all caller-supplied context (campaign_id, channel, etc.)
        # so PS can correlate the suggestion with its campaign state.
        await self._emit(PSE_POINT_SUGGESTED, {
            **payload,
            "trial_id": trial_id,
            "parameters": params,
            "pse_context": pse_context,
        })

    async def _handle_submit_result(self, payload: dict) -> None:
        # Full-dataset rebuild path: PS sends {"full_dataset": true, "data_points": [...]}
        # to replace the entire training set (e.g. after excluding a bad trial).
        if payload.get("full_dataset"):
            with self._lock:
                service = self._service
            if service is None:
                logger.warning("submit_result(full_dataset) received but no PSE service is running.")
                return
            data_points = payload.get("data_points", [])
            campaign_id = payload.get("campaign_id", "?")
            logger.info(
                "submit_result(full_dataset): rebuilding from %d points for campaign %s.",
                len(data_points), campaign_id,
            )
            try:
                await asyncio.to_thread(service.retrain, data_points)
            except Exception:
                logger.exception(
                    "PSEPointService.retrain() failed for campaign %s.", campaign_id
                )
            return

        trial_id = payload.get("trial_id")
        value = payload.get("value")
        variance = payload.get("variance")

        if trial_id is None or value is None:
            logger.warning(
                "submit_result missing required fields (trial_id=%s, value=%s).",
                trial_id, value,
            )
            return

        with self._lock:
            service = self._service
        if service is None:
            logger.warning("submit_result received but no PSE service is running.")
            return

        try:
            await asyncio.to_thread(
                service.submit_result, trial_id, float(value), variance
            )
        except Exception:
            logger.exception("PSEPointService.submit_result() failed for trial %s.", trial_id)

    async def _handle_cancel_trial(self, payload: dict) -> None:
        trial_id = payload.get("trial_id")
        if not trial_id:
            return
        with self._lock:
            service = self._service
        if service is None:
            logger.warning("cancel_trial received but no PSE service is running.")
            return
        try:
            await asyncio.to_thread(service.cancel_trial, trial_id)
        except Exception:
            logger.exception("PSEPointService.cancel_trial() failed for trial %s.", trial_id)

    async def _handle_notify_in_flight(self, payload: dict) -> None:
        trial_id = payload.get("trial_id")
        params = payload.get("parameters", {})
        if not trial_id or not params:
            return
        with self._lock:
            service = self._service
        if service is None:
            logger.warning("notify_in_flight received but no PSE service is running.")
            return
        try:
            await asyncio.to_thread(service.notify_in_flight, trial_id, params)
        except Exception:
            logger.exception(
                "PSEPointService.notify_in_flight() failed for trial %s.", trial_id
            )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def _capabilities_payload(self) -> dict:
        """Return the acquisition_functions list for inclusion in pse.ready payloads."""
        return {"acquisition_functions": list(ACQUISITION_FUNCTIONS.keys())}

    async def _handle_announce_request(self, payload: dict) -> None:
        """Re-emit pse.ready with current state + capabilities on demand.

        Mirrors the device.announce_request pattern so Protocol Studio can learn
        PSE's capabilities even when PS starts after PSE's initial pse.ready.
        """
        with self._lock:
            has_service = self._service is not None
        await self._emit(PSE_READY, {"has_service": has_service, **self._capabilities_payload()})
        logger.debug("announce_request: re-emitted pse.ready (has_service=%s).", has_service)

    # ------------------------------------------------------------------
    # Outbound helpers
    # ------------------------------------------------------------------

    async def _emit(self, routing_key: str, payload: dict) -> None:
        env = build(device_id="pse", routing_key=routing_key, payload=payload)
        await publish(self._exchange, routing_key, env)

    def publish_telemetry(self, routing_key: str, payload: dict) -> None:
        """Fire-and-forget telemetry publish.  Silently drops if not yet connected."""
        if not self._ready.is_set() or self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._emit(routing_key, payload), self._loop
        )
