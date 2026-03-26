"""
Multi-Objective Parametric Optimizer.

Finds Pareto-optimal designs across competing objectives (e.g. maximum
field flux vs. minimum material volume vs. minimum power consumption).

Uses scipy differential evolution for global search and a fast PINN
surrogate when available, falling back to direct evaluation otherwise.

Dependencies: numpy, scipy (standard); optuna (optional, for Bayesian search)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import differential_evolution, minimize

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParameterBound:
    """A single named design parameter with bounds."""
    name: str
    low: float
    high: float
    integer: bool = False       # round to nearest int during evaluation

    def sample(self) -> float:
        v = np.random.uniform(self.low, self.high)
        return round(v) if self.integer else v

    def clip(self, v: float) -> float:
        v = float(np.clip(v, self.low, self.high))
        return round(v) if self.integer else v


@dataclass
class Objective:
    """A named scalar objective with a direction."""
    name: str
    fn: Callable[[dict], float]     # maps param dict → scalar
    minimize: bool = True           # True = lower is better

    def evaluate(self, params: dict) -> float:
        raw = self.fn(params)
        return raw if self.minimize else -raw   # internally always minimise


@dataclass
class DesignPoint:
    """One evaluated design in the search space."""
    params: dict
    objective_values: dict          # raw (un-negated) values per objective name
    dominated: bool = False

    def __repr__(self):
        vals = ", ".join(f"{k}={v:.4g}" for k, v in self.objective_values.items())
        return f"DesignPoint({vals})"


@dataclass
class OptimizationResult:
    """Full result of a multi-objective optimization run."""
    all_points: list[DesignPoint]
    pareto_front: list[DesignPoint]
    best_single: Optional[DesignPoint] = None   # best on first objective
    elapsed_s: float = 0.0
    method: str = ""

    def summary(self) -> str:
        lines = [
            f"Method: {self.method}  |  {len(self.all_points)} designs evaluated  "
            f"|  {len(self.pareto_front)} on Pareto front  |  {self.elapsed_s:.1f}s",
            "",
            "Pareto Front:",
        ]
        for i, pt in enumerate(self.pareto_front):
            vals = "  ".join(f"{k}={v:.4g}" for k, v in pt.objective_values.items())
            lines.append(f"  [{i+1:2d}]  {vals}")
            pstr = "  ".join(f"{k}={v:.4g}" for k, v in pt.params.items())
            lines.append(f"        params: {pstr}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------

def is_dominated(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if point `a` is dominated by point `b` (b ≤ a in all dims, < in one)."""
    return bool(np.all(b <= a) and np.any(b < a))


def compute_pareto_front(points: list[DesignPoint], objective_names: list[str]) -> list[DesignPoint]:
    """
    Mark dominated points and return the Pareto-non-dominated subset.

    Args:
        points:           All evaluated design points.
        objective_names:  Objective names used for comparison (minimisation assumed).

    Returns:
        Non-dominated subset of points.
    """
    n = len(points)
    costs = np.array([
        [pt.objective_values[name] for name in objective_names]
        for pt in points
    ])

    dominated_flags = [False] * n
    for i in range(n):
        for j in range(n):
            if i != j and not dominated_flags[j]:
                if is_dominated(costs[i], costs[j]):
                    dominated_flags[i] = True
                    break

    for pt, flag in zip(points, dominated_flags):
        pt.dominated = flag

    return [pt for pt, flag in zip(points, dominated_flags) if not flag]


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class ParametricOptimizer:
    """
    Multi-objective parametric optimizer.

    Supports three backends:
    - ``"differential_evolution"`` — scipy global search (default, no extra deps)
    - ``"bayesian"`` — Optuna TPE sampler (install: ``pip install optuna``)
    - ``"latin_hypercube"`` — space-filling random sampling (fastest, least precise)

    Example::

        from utils.optimizer import ParametricOptimizer, ParameterBound, Objective
        from cad.magnetic_analyzer import MagneticAnalyzer, solenoid_path

        analyzer = MagneticAnalyzer()

        def mean_flux(params):
            coil = solenoid_path(
                radius_m=params["radius"] * 1e-3,
                length_m=params["length"] * 1e-3,
                turns=int(params["turns"]),
                current=params["current"],
            )
            pts = analyzer.axial_line(-0.05, 0.05, n=50)
            result = analyzer.compute_field(coil, pts)
            return -result.mean_field()   # negate: we want to maximise

        def wire_mass(params):
            import numpy as np
            wire_r = 0.5e-3   # 1 mm wire, fixed
            length = 2 * np.pi * params["radius"] * 1e-3 * int(params["turns"])
            return length * np.pi * wire_r**2 * 8960.0  # copper kg

        opt = ParametricOptimizer(
            bounds=[
                ParameterBound("radius",  10.0, 100.0),
                ParameterBound("length",  20.0, 200.0),
                ParameterBound("turns",    10,   500, integer=True),
                ParameterBound("current",  0.5,  10.0),
            ],
            objectives=[
                Objective("mean_flux_T", mean_flux, minimize=False),
                Objective("wire_mass_kg", wire_mass, minimize=True),
            ],
        )
        result = opt.run(n_iterations=200, method="differential_evolution")
        print(result.summary())
    """

    def __init__(
        self,
        bounds: list[ParameterBound],
        objectives: list[Objective],
        pinn_surrogate=None,        # optional: trained PINNModel for fast evaluation
        surrogate_input_fn: Optional[Callable[[dict], np.ndarray]] = None,
        verbose: bool = True,
    ):
        """
        Args:
            bounds:              List of ParameterBound (design variables).
            objectives:          List of Objective (what to optimise).
            pinn_surrogate:      Optional PINNModel. When provided, used for fast
                                 field estimates instead of full FEM/Biot-Savart.
            surrogate_input_fn:  Maps param dict → (1, input_dim) tensor for PINN.
            verbose:             Print progress.
        """
        self.bounds = bounds
        self.objectives = objectives
        self.pinn = pinn_surrogate
        self.surrogate_input_fn = surrogate_input_fn
        self.verbose = verbose
        self._all_points: list[DesignPoint] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        n_iterations: int = 200,
        method: str = "differential_evolution",
        seed: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Run multi-objective optimization.

        Args:
            n_iterations: Total function evaluations (approx).
            method:       "differential_evolution", "bayesian", or "latin_hypercube".
            seed:         Random seed for reproducibility.

        Returns:
            OptimizationResult with Pareto front and all evaluated points.
        """
        if seed is not None:
            np.random.seed(seed)

        self._all_points = []
        t0 = time.perf_counter()

        if method == "differential_evolution":
            self._run_de(n_iterations)
        elif method == "bayesian":
            if not _OPTUNA_AVAILABLE:
                warnings.warn("Optuna not installed. Falling back to differential_evolution. "
                              "Install with: pip install optuna")
                self._run_de(n_iterations)
                method = "differential_evolution (fallback)"
            else:
                self._run_bayesian(n_iterations)
        elif method == "latin_hypercube":
            self._run_lhs(n_iterations)
        else:
            raise ValueError(f"Unknown method: {method}")

        pareto = compute_pareto_front(
            self._all_points, [o.name for o in self.objectives]
        )

        # Best on first objective (raw value)
        first_obj = self.objectives[0].name
        best = min(self._all_points, key=lambda p: p.objective_values[first_obj],
                   default=None)

        elapsed = time.perf_counter() - t0
        result = OptimizationResult(
            all_points=list(self._all_points),
            pareto_front=pareto,
            best_single=best,
            elapsed_s=elapsed,
            method=method,
        )
        if self.verbose:
            print(result.summary())
        return result

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _evaluate(self, param_array: np.ndarray) -> dict[str, float]:
        """Convert raw numpy array to param dict, evaluate all objectives."""
        params = {
            b.name: b.clip(v)
            for b, v in zip(self.bounds, param_array)
        }
        obj_values = {}
        for obj in self.objectives:
            try:
                obj_values[obj.name] = float(obj.fn(params))
            except Exception as exc:
                obj_values[obj.name] = float("inf")
                if self.verbose:
                    print(f"  [Optimizer] Objective '{obj.name}' failed: {exc}")

        point = DesignPoint(params=params, objective_values=obj_values)
        self._all_points.append(point)
        return obj_values

    def _scalarize(self, param_array: np.ndarray) -> float:
        """Weighted sum scalarization for single-objective backends."""
        vals = self._evaluate(param_array)
        # Equal-weight sum of internally-minimised objectives
        total = 0.0
        for obj in self.objectives:
            v = vals[obj.name]
            total += v if obj.minimize else -v
        return total

    def _run_de(self, n_iterations: int):
        """Differential evolution global search."""
        scipy_bounds = [(b.low, b.high) for b in self.bounds]
        popsize = max(5, n_iterations // 15)

        if self.verbose:
            print(f"[Optimizer] Differential evolution | pop={popsize} | "
                  f"max_iter={n_iterations // popsize}")

        differential_evolution(
            self._scalarize,
            bounds=scipy_bounds,
            maxiter=max(1, n_iterations // popsize),
            popsize=popsize,
            seed=None,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=True,
            updating="deferred",
        )

    def _run_bayesian(self, n_iterations: int):
        """Optuna TPE-based Bayesian optimization."""
        if self.verbose:
            print(f"[Optimizer] Bayesian (Optuna) | n_trials={n_iterations}")

        def optuna_objective(trial):
            param_array = []
            for b in self.bounds:
                if b.integer:
                    v = trial.suggest_int(b.name, int(b.low), int(b.high))
                else:
                    v = trial.suggest_float(b.name, b.low, b.high)
                param_array.append(v)
            return self._scalarize(np.array(param_array))

        study = optuna.create_study(direction="minimize")
        study.optimize(optuna_objective, n_trials=n_iterations, show_progress_bar=False)

    def _run_lhs(self, n_samples: int):
        """Latin hypercube sampling — fastest, no convergence guarantee."""
        if self.verbose:
            print(f"[Optimizer] Latin hypercube | n_samples={n_samples}")

        n_dims = len(self.bounds)
        samples = np.zeros((n_samples, n_dims))
        for i, b in enumerate(self.bounds):
            perm = np.random.permutation(n_samples)
            samples[:, i] = b.low + (perm + np.random.uniform(size=n_samples)) / n_samples * (b.high - b.low)

        for row in samples:
            self._evaluate(row)


# ---------------------------------------------------------------------------
# Coil-specific optimizer (convenience wrapper)
# ---------------------------------------------------------------------------

class CoilOptimizer:
    """
    Convenience wrapper for solenoid/coil optimisation.

    Optimises simultaneously for:
    - Maximum mean field in target region
    - Minimum wire mass (material efficiency)
    - Minimum power consumption
    - Maximum field uniformity

    Example::

        from utils.optimizer import CoilOptimizer

        opt = CoilOptimizer(
            target_z_half_m=0.04,    # target region ±40 mm axially
            target_radius_m=0.02,    # target region radius
            wire_diameter_m=1e-3,
            current_a=2.0,
        )
        result = opt.run(n_iterations=300)
        best = result.pareto_front[0]
        print(best)
    """

    def __init__(
        self,
        target_z_half_m: float = 0.05,
        target_radius_m: float = 0.02,
        wire_diameter_m: float = 1e-3,
        current_a: float = 1.0,
        verbose: bool = True,
    ):
        from cad.magnetic_analyzer import MagneticAnalyzer, solenoid_path, MagneticAnalyzer

        self.analyzer = MagneticAnalyzer()
        self.target_z = target_z_half_m
        self.target_r = target_radius_m
        self.wire_d = wire_diameter_m
        self.current = current_a
        self.verbose = verbose

        # Pre-build evaluation grid
        self._grid = MagneticAnalyzer.volume_grid(
            r_max=target_radius_m,
            z_half=target_z_half_m,
            n=8,
        )

    def _make_coil(self, params: dict):
        from cad.magnetic_analyzer import solenoid_path
        return solenoid_path(
            radius_m=params["radius_mm"] * 1e-3,
            length_m=params["length_mm"] * 1e-3,
            turns=int(params["turns"]),
            current=self.current,
        )

    def run(
        self,
        n_iterations: int = 200,
        method: str = "differential_evolution",
        seed: Optional[int] = None,
    ) -> OptimizationResult:

        def neg_mean_field(params):
            coil = self._make_coil(params)
            result = self.analyzer.compute_field(coil, self._grid)
            return -result.mean_field()   # maximise → negate

        def wire_mass(params):
            coil = self._make_coil(params)
            wire_area = np.pi * (self.wire_d / 2) ** 2
            return coil.wire_length * wire_area * 8960.0  # copper kg

        def power(params):
            coil = self._make_coil(params)
            wire_area = np.pi * (self.wire_d / 2) ** 2
            resistance = 1.68e-8 * coil.wire_length / wire_area  # copper
            return self.current ** 2 * resistance

        def uniformity(params):
            coil = self._make_coil(params)
            result = self.analyzer.compute_field(coil, self._grid)
            return result.uniformity()

        opt = ParametricOptimizer(
            bounds=[
                ParameterBound("radius_mm",  5.0,  150.0),
                ParameterBound("length_mm",  10.0, 300.0),
                ParameterBound("turns",      5,    1000, integer=True),
            ],
            objectives=[
                Objective("mean_field_T",  neg_mean_field, minimize=True),   # internally minimised
                Objective("wire_mass_kg",  wire_mass,      minimize=True),
                Objective("power_W",       power,          minimize=True),
                Objective("uniformity",    uniformity,     minimize=True),
            ],
            verbose=self.verbose,
        )
        return opt.run(n_iterations=n_iterations, method=method, seed=seed)
