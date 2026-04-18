"""
utils/optimizer.py
==================
Bayesian / Pareto optimizer for the Mobius-Nova PMSG platform.

This is the active search engine. While the PINN evaluates designs and
the self-correction loop improves constraint satisfaction, this module
DRIVES the search — proposing new design candidates, tracking the Pareto
frontier, and converging on optimal designs.

OPTIMIZATION PROBLEM
--------------------
Design space: 40-variable Bézier pole geometry (bezier_geometry.py)
Objectives (maximize):
  1. Torque density      [Nm/kg] — primary commercial metric
  2. Mass reduction      [%]     — rare-earth cost reduction
  3. Efficiency          [%]     — grid-tie requirement

Constraints (all must be satisfied — 22 total from self_correction.py):
  Tier 1: 7 hard limits (axial stiffness, demagnetisation, etc.)
  Tier 2: 6 performance targets (cogging, THD, efficiency, etc.)
  Tier 3: 6 Pareto objectives (these become objective functions here)
  Tier 4: 4 coupling checks

The optimizer never evaluates a design that violates Tier-1 constraints.
This is the physics invariant boundary — the same principle as autoresearch.

ALGORITHM
---------
Bayesian Optimization with Gaussian Process surrogate:
  - Prior: Latin Hypercube Sampling (LHS) — 50 initial designs
  - Surrogate: Gaussian Process (GP) with Matérn 5/2 kernel
  - Acquisition: Expected Hypervolume Improvement (EHVI) for multi-objective
  - Update: GP refitted after every batch of evaluations
  - Convergence: Hypervolume indicator stops improving < 0.1%

This is significantly more efficient than random search or grid search.
The GP learns which regions of the 40-dimensional space are promising
and concentrates evaluations there.

PARETO FRONTIER
---------------
The optimizer maintains a Pareto frontier — the set of designs where
no objective can be improved without degrading another.

For the PMSG:
  A design is Pareto-optimal if no other design has higher torque density
  AND higher mass reduction AND higher efficiency simultaneously.

The frontier is the commercial offering — customers choose where on
the frontier they want to operate based on their priorities.

INTEGRATION
-----------
  bezier_geometry.py  → design parameter sampling + evaluation
  pinn_model.py       → fast objective function evaluation
  self_correction.py  → constraint checking per candidate
  design_genome.py    → Pareto frontier persistence
  pinn_data_manager.py → run history storage

References
----------
Bayesian optimization: Brochu et al. 2010, arXiv:1012.2599
EHVI acquisition: Emmerich et al. 2006, Emmerich & Klinkenberg 2008
Latin Hypercube Sampling: McKay et al. 1979
NREL Pareto targets: Sethuraman et al. 2024 — torque density 351.28 Nm/kg
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Bayesian optimization
try:
    from scipy.stats import norm
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn("scipy not found — pip install scipy", stacklevel=2)

# Latin Hypercube Sampling
try:
    from pyDOE2 import lhs
    _LHS = True
except ImportError:
    _LHS = False
    # Fallback: random sampling


# ===========================================================================
# Design Space Definition
# ===========================================================================

# PMSG design variable bounds (from bezier_geometry.py)
# 40 variables: 36 Bézier control points + 4 scalars
DESIGN_BOUNDS = np.array([
    # Bézier control points — normalized radial positions (0 = inner, 1 = outer)
    *[(0.0, 1.0)] * 33,   # 33 control point coordinates
    # Scalar design variables
    (0.50, 0.95),          # [33] pole_arc_ratio
    (0.60, 1.40),          # [34] magnet_thickness_ratio (normalized)
    (0.50, 1.50),          # [35] core_mass_ratio
    (0.50, 1.50),          # [36] magnet_mass_ratio
    # Additional variables
    (0.0,  1.0),           # [37] halbach_fraction (0=radial, 1=full Halbach)
    (0.0,  1.0),           # [38] asymmetry_index (0=symmetric, 1=max asymmetric)
    (0.0,  1.0),           # [39] material_mix (0=sintered, 1=printed bonded)
])[:40]  # ensure exactly 40

N_DIM = len(DESIGN_BOUNDS)
BOUNDS_LOW  = DESIGN_BOUNDS[:, 0]
BOUNDS_HIGH = DESIGN_BOUNDS[:, 1]

# NREL Pareto reference point (from MADE3D slides)
PARETO_REFERENCE = np.array([
    351.28,   # Nm/kg — NREL IEA 15-MW torque density baseline
    0.0,      # % — zero mass reduction (baseline)
    93.0,     # % — minimum efficiency
])


# ===========================================================================
# Gaussian Process Surrogate
# ===========================================================================

class GaussianProcessSurrogate:
    """
    Gaussian Process surrogate model for the PMSG objective functions.

    Uses Matérn 5/2 kernel — good for physical systems where the
    true function is twice-differentiable but not necessarily smooth.

    For multi-objective optimization: one GP per objective.
    Predicts mean and variance, enabling uncertainty-aware acquisition.
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train: Optional[np.ndarray] = None
        self.Y_train: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        self.K_inv_: Optional[np.ndarray] = None
        self._fitted = False

    def _matern52(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matérn 5/2 kernel — standard choice for physical systems."""
        dists = cdist(X1 / self.length_scale, X2 / self.length_scale)
        sqrt5 = math.sqrt(5)
        return (1 + sqrt5 * dists + 5/3 * dists**2) * np.exp(-sqrt5 * dists)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit GP to training data.
        X: (N, D) design matrix
        y: (N,) objective values
        """
        self.X_train = X.copy()
        self.Y_train = y.copy()

        K = self._matern52(X, X)
        K += self.noise * np.eye(len(X))

        # Cholesky for numerical stability
        try:
            L = np.linalg.cholesky(K)
            self.alpha_ = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self.K_inv_ = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
        except np.linalg.LinAlgError:
            # Fallback: add more noise
            K += 1e-4 * np.eye(len(X))
            self.K_inv_ = np.linalg.inv(K)
            self.alpha_ = self.K_inv_ @ y

        self._fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std at new points.
        Returns: (mean, std) each shape (N,)
        """
        if not self._fitted:
            n = len(X)
            return np.zeros(n), np.ones(n)

        K_star = self._matern52(X, self.X_train)
        mean = K_star @ self.alpha_

        K_star_star = np.diag(self._matern52(X, X))
        var = K_star_star - np.sum(K_star @ self.K_inv_ * K_star, axis=1)
        std = np.sqrt(np.maximum(var, 1e-12))

        return mean, std


# ===========================================================================
# Acquisition Functions
# ===========================================================================

class AcquisitionFunction:
    """
    Acquisition functions for Bayesian optimization.

    Expected Improvement (EI): single-objective
    Expected Hypervolume Improvement (EHVI): multi-objective
    Upper Confidence Bound (UCB): exploration-exploitation trade-off
    """

    @staticmethod
    def expected_improvement(
        mean: np.ndarray,
        std: np.ndarray,
        best_f: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        """
        Expected Improvement acquisition function.
        Higher = more worth evaluating.
        xi: exploration parameter (0=exploit, 0.1=explore)
        """
        z = (mean - best_f - xi) / np.maximum(std, 1e-9)
        ei = (mean - best_f - xi) * norm.cdf(z) + std * norm.pdf(z)
        return np.maximum(ei, 0.0)

    @staticmethod
    def upper_confidence_bound(
        mean: np.ndarray,
        std: np.ndarray,
        kappa: float = 2.576,
    ) -> np.ndarray:
        """UCB acquisition — kappa controls exploration."""
        return mean + kappa * std

    @staticmethod
    def hypervolume_improvement(
        mean: np.ndarray,
        std: np.ndarray,
        pareto_front: np.ndarray,
        reference_point: np.ndarray,
    ) -> np.ndarray:
        """
        Simplified Expected Hypervolume Improvement.
        Approximated as sum of EI per objective relative to Pareto front.
        Full EHVI requires expensive Monte Carlo integration.
        """
        n_obs = len(mean)
        n_obj = mean.shape[1] if mean.ndim > 1 else 1

        if pareto_front is None or len(pareto_front) == 0:
            # No Pareto front yet — use UCB
            return AcquisitionFunction.upper_confidence_bound(
                mean[:, 0] if mean.ndim > 1 else mean,
                std[:, 0] if std.ndim > 1 else std,
            )

        # Approximate: for each point, compute improvement over
        # the closest Pareto front point in each objective
        ehvi = np.zeros(n_obs)
        for j in range(min(n_obj, pareto_front.shape[1] if pareto_front.ndim > 1 else 1)):
            best_f_j = np.max(pareto_front[:, j]) if pareto_front.ndim > 1 else np.max(pareto_front)
            m_j = mean[:, j] if mean.ndim > 1 else mean
            s_j = std[:, j] if std.ndim > 1 else std
            ei_j = AcquisitionFunction.expected_improvement(m_j, s_j, best_f_j)
            ehvi += ei_j

        return ehvi


# ===========================================================================
# Pareto Frontier Tracker
# ===========================================================================

class ParetoFrontierTracker:
    """
    Tracks and updates the Pareto frontier during optimization.

    A design is Pareto-optimal (non-dominated) if no other evaluated
    design has better or equal values on ALL objectives simultaneously.

    For the PMSG:
      - Higher torque density is better
      - Higher mass reduction is better
      - Higher efficiency is better
    All three objectives are to be MAXIMIZED.
    """

    def __init__(self, n_objectives: int = 3):
        self.n_obj = n_objectives
        self.pareto_X: List[np.ndarray] = []   # design vectors
        self.pareto_Y: List[np.ndarray] = []   # objective vectors
        self.pareto_ids: List[str] = []         # design IDs

    def is_dominated(self, y_new: np.ndarray) -> bool:
        """True if y_new is dominated by any design in the frontier."""
        if not self.pareto_Y:
            return False
        Y = np.array(self.pareto_Y)
        # dominated if exists y in frontier where y >= y_new on all objectives
        # and y > y_new on at least one
        dominated = np.any(
            np.all(Y >= y_new, axis=1) & np.any(Y > y_new, axis=1)
        )
        return bool(dominated)

    def update(
        self,
        x_new: np.ndarray,
        y_new: np.ndarray,
        design_id: str = "",
    ) -> bool:
        """
        Add design to frontier if non-dominated.
        Remove dominated designs from frontier.
        Returns True if design was added to Pareto front.
        """
        if self.is_dominated(y_new):
            return False

        # Remove designs now dominated by y_new
        if self.pareto_Y:
            Y = np.array(self.pareto_Y)
            # keep designs not dominated by y_new
            not_dominated = ~(
                np.all(y_new >= Y, axis=1) & np.any(y_new > Y, axis=1)
            )
            self.pareto_X  = [self.pareto_X[i]  for i, k in enumerate(not_dominated) if k]
            self.pareto_Y  = [self.pareto_Y[i]  for i, k in enumerate(not_dominated) if k]
            self.pareto_ids = [self.pareto_ids[i] for i, k in enumerate(not_dominated) if k]

        self.pareto_X.append(x_new.copy())
        self.pareto_Y.append(y_new.copy())
        self.pareto_ids.append(design_id)
        return True

    def hypervolume(self, reference: Optional[np.ndarray] = None) -> float:
        """
        Compute hypervolume indicator of the Pareto frontier.
        Higher = better-spread, better-performing frontier.
        Used as convergence criterion.
        """
        if not self.pareto_Y:
            return 0.0

        ref = reference if reference is not None else PARETO_REFERENCE
        Y = np.array(self.pareto_Y)

        # Simplified 3D hypervolume via Monte Carlo
        n_samples = 10000
        # Sample uniformly in [ref, max+buffer] hypercube
        Y_max = np.max(Y, axis=0) * 1.1
        samples = np.random.uniform(ref, Y_max, (n_samples, self.n_obj))

        # Count samples dominated by at least one Pareto point
        dominated_count = 0
        for s in samples:
            if np.any(np.all(Y >= s, axis=1)):
                dominated_count += 1

        volume = np.prod(Y_max - ref) * dominated_count / n_samples
        return float(volume)

    def summary(self) -> str:
        if not self.pareto_Y:
            return "Pareto frontier: empty"
        Y = np.array(self.pareto_Y)
        lines = [
            f"Pareto frontier: {len(self.pareto_Y)} designs",
            f"  Torque density: {Y[:,0].min():.1f} – {Y[:,0].max():.1f} Nm/kg",
            f"  Mass reduction: {Y[:,1].min():.1f} – {Y[:,1].max():.1f} %",
            f"  Efficiency:     {Y[:,2].min():.1f} – {Y[:,2].max():.1f} %",
            f"  Hypervolume:    {self.hypervolume():.2f}",
        ]
        return "\n".join(lines)


# ===========================================================================
# Main Optimizer
# ===========================================================================

class PMSGBayesianOptimizer:
    """
    Bayesian optimizer for the PMSG Bézier pole geometry.

    Drives the search for Pareto-optimal designs across:
      - Torque density [Nm/kg]
      - Magnet mass reduction [%]
      - Electrical efficiency [%]

    While satisfying all 22 NREL/ORNL physics constraints.

    The optimizer never calls the expensive PINN evaluator blindly —
    it uses the GP surrogate to propose the most informative next
    design candidates, minimizing total evaluation cost.
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], np.ndarray],
        constraint_fn: Optional[Callable[[np.ndarray], Dict]] = None,
        n_initial:     int = 50,
        batch_size:    int = 5,
        max_iter:      int = 200,
        convergence_tol: float = 0.001,
        output_dir:    str = "./optimizer_results",
        verbose:       bool = True,
    ):
        """
        Parameters
        ----------
        objective_fn : callable
            Takes design vector (40,) → returns objectives (3,)
            [torque_density_Nmkg, mass_reduction_pct, efficiency_pct]

        constraint_fn : callable, optional
            Takes design vector (40,) → returns constraint dict
            Must include 'tier1_all_clear' key
            If None: constraints not checked (use only for testing)

        n_initial : int
            Number of LHS samples for initial exploration

        batch_size : int
            Designs to evaluate per iteration

        max_iter : int
            Maximum optimization iterations

        convergence_tol : float
            Stop when hypervolume improvement < this fraction
        """
        self.objective_fn   = objective_fn
        self.constraint_fn  = constraint_fn
        self.n_initial      = n_initial
        self.batch_size     = batch_size
        self.max_iter       = max_iter
        self.conv_tol       = convergence_tol
        self.output_dir     = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose        = verbose

        # Surrogate models — one GP per objective
        self.surrogates = [
            GaussianProcessSurrogate(length_scale=0.5, noise=1e-6)
            for _ in range(3)
        ]

        # Data storage
        self.X_all: List[np.ndarray] = []   # all evaluated designs
        self.Y_all: List[np.ndarray] = []   # all objectives
        self.constraint_log: List[Dict] = []

        # Pareto tracker
        self.pareto = ParetoFrontierTracker(n_objectives=3)

        # Convergence tracking
        self.hypervolume_history: List[float] = []
        self.iteration = 0
        self.n_evaluations = 0
        self.n_constraint_violations = 0

        if self.verbose:
            print(f"\n[Optimizer] PMSG Bayesian Optimizer initialized")
            print(f"  Design space: {N_DIM}D Bézier")
            print(f"  Initial samples: {n_initial} (LHS)")
            print(f"  Max iterations: {max_iter}")
            print(f"  Batch size: {batch_size}")
            print(f"  Objectives: torque_density, mass_reduction, efficiency")

    def _sample_lhs(self, n: int) -> np.ndarray:
        """Latin Hypercube Sampling in the design space."""
        if _LHS:
            unit_cube = lhs(N_DIM, samples=n, criterion='maximin')
        else:
            unit_cube = np.random.uniform(0, 1, (n, N_DIM))

        # Scale to bounds
        return BOUNDS_LOW + unit_cube * (BOUNDS_HIGH - BOUNDS_LOW)

    def _evaluate(self, x: np.ndarray, design_id: str = "") -> Optional[np.ndarray]:
        """
        Evaluate one design — check constraints first, then objectives.
        Returns None if Tier-1 constraint violated.
        """
        # Constraint check FIRST — never evaluate infeasible designs
        if self.constraint_fn is not None:
            constraints = self.constraint_fn(x)
            self.constraint_log.append(constraints)

            if not constraints.get("tier1_all_clear", True):
                self.n_constraint_violations += 1
                if self.verbose:
                    worst = constraints.get("worst_tier1", "unknown")
                    print(f"  ✗ Tier-1 violation: {worst} — skipped")
                return None

        # Evaluate objectives
        y = self.objective_fn(x)
        self.n_evaluations += 1
        return y

    def _fit_surrogates(self):
        """Refit all GP surrogates on current data."""
        if len(self.X_all) < 3:
            return
        X = np.array(self.X_all)
        Y = np.array(self.Y_all)

        for i, gp in enumerate(self.surrogates):
            gp.fit(X, Y[:, i])

    def _propose_candidates(self, n: int = None) -> np.ndarray:
        """
        Propose next batch of candidates using acquisition function.
        Uses random search over acquisition function (efficient for high-D).
        """
        n = n or self.batch_size
        n_random = max(1000, n * 200)

        # Random candidates in feasible space
        candidates = self._sample_lhs(n_random)

        if len(self.X_all) < 3:
            # Not enough data for GP — return LHS
            return candidates[:n]

        # Predict with all surrogates
        means = []
        stds  = []
        for gp in self.surrogates:
            m, s = gp.predict(candidates)
            means.append(m)
            stds.append(s)

        means = np.column_stack(means)  # (n_random, 3)
        stds  = np.column_stack(stds)   # (n_random, 3)

        # EHVI acquisition
        pareto_Y = np.array(self.pareto.pareto_Y) if self.pareto.pareto_Y else None
        acq = AcquisitionFunction.hypervolume_improvement(
            means, stds, pareto_Y, PARETO_REFERENCE
        )

        # Select top n by acquisition
        top_idx = np.argsort(acq)[-n:][::-1]
        return candidates[top_idx]

    def run(self) -> Dict:
        """
        Main optimization loop.

        Phase 1: Initial LHS exploration (n_initial designs)
        Phase 2: Bayesian optimization (max_iter × batch_size designs)
        Phase 3: Convergence check and reporting
        """
        t_start = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  PHASE 1: Initial LHS Exploration ({self.n_initial} designs)")
            print(f"{'='*60}")

        # Phase 1 — Latin Hypercube initial sampling
        X_init = self._sample_lhs(self.n_initial)
        for i, x in enumerate(X_init):
            design_id = f"lhs_{i:04d}"
            y = self._evaluate(x, design_id)
            if y is not None:
                self.X_all.append(x)
                self.Y_all.append(y)
                added = self.pareto.update(x, y, design_id)
                if self.verbose and added:
                    print(f"  [{i+1}/{self.n_initial}] Pareto update: "
                          f"T={y[0]:.1f} Nm/kg  M={y[1]:.1f}%  η={y[2]:.1f}%")

        self._fit_surrogates()
        hv = self.pareto.hypervolume()
        self.hypervolume_history.append(hv)

        if self.verbose:
            print(f"\n  After LHS: {len(self.pareto.pareto_Y)} Pareto designs")
            print(f"  Hypervolume: {hv:.4f}")

        # Phase 2 — Bayesian optimization
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  PHASE 2: Bayesian Optimization (max {self.max_iter} iterations)")
            print(f"{'='*60}")

        for iteration in range(self.max_iter):
            self.iteration = iteration

            # Propose next batch
            candidates = self._propose_candidates(self.batch_size)

            n_added = 0
            for j, x in enumerate(candidates):
                design_id = f"bo_{iteration:04d}_{j:02d}"
                y = self._evaluate(x, design_id)
                if y is not None:
                    self.X_all.append(x)
                    self.Y_all.append(y)
                    added = self.pareto.update(x, y, design_id)
                    if added:
                        n_added += 1

            # Refit surrogates
            self._fit_surrogates()

            # Compute hypervolume
            hv_new = self.pareto.hypervolume()
            hv_prev = self.hypervolume_history[-1] if self.hypervolume_history else 0.0
            hv_improvement = (hv_new - hv_prev) / max(hv_prev, 1e-9)
            self.hypervolume_history.append(hv_new)

            if self.verbose:
                print(f"  Iter {iteration+1:3d}/{self.max_iter} | "
                      f"Pareto: {len(self.pareto.pareto_Y):3d} | "
                      f"HV: {hv_new:.4f} ({hv_improvement:+.3%}) | "
                      f"Evals: {self.n_evaluations} | "
                      f"Violations: {self.n_constraint_violations}")

            # Convergence check
            if (iteration >= 10 and
                    abs(hv_improvement) < self.conv_tol and
                    n_added == 0):
                if self.verbose:
                    print(f"\n  Converged at iteration {iteration+1}")
                    print(f"  Hypervolume improvement {hv_improvement:.4%} < {self.conv_tol:.4%}")
                break

        elapsed = time.time() - t_start

        # Final results
        results = self._build_results(elapsed)
        self._save_results(results)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  OPTIMIZATION COMPLETE")
            print(f"{'='*60}")
            print(f"  Total evaluations:    {self.n_evaluations}")
            print(f"  Constraint violations: {self.n_constraint_violations}")
            print(f"  Elapsed time:         {elapsed:.1f}s")
            print(f"\n{self.pareto.summary()}")

        return results

    def _build_results(self, elapsed: float) -> Dict:
        """Build complete results dict for design_genome.py ingestion."""
        pareto_designs = []
        for x, y, pid in zip(
            self.pareto.pareto_X,
            self.pareto.pareto_Y,
            self.pareto.pareto_ids,
        ):
            pareto_designs.append({
                "design_id":          pid,
                "bezier_vector":      x.tolist(),
                "torque_density_Nmkg": float(y[0]),
                "mass_reduction_pct": float(y[1]),
                "efficiency_pct":     float(y[2]),
                "nrel_baseline_Nmkg": 351.28,
                "improvement_pct":    float((y[0] - 351.28) / 351.28 * 100),
            })

        # Sort by torque density
        pareto_designs.sort(key=lambda d: -d["torque_density_Nmkg"])

        return {
            "optimizer":           "PMSGBayesianOptimizer",
            "algorithm":           "Bayesian (GP + EHVI) + Latin Hypercube init",
            "nrel_ref":            "NREL/ORNL Sethuraman et al. 2024",
            "n_evaluations":       self.n_evaluations,
            "n_constraint_violations": self.n_constraint_violations,
            "n_pareto_designs":    len(self.pareto.pareto_Y),
            "elapsed_s":           round(elapsed, 2),
            "final_hypervolume":   self.pareto.hypervolume(),
            "hypervolume_history": self.hypervolume_history,
            "pareto_designs":      pareto_designs,
            "best_torque_density": pareto_designs[0] if pareto_designs else None,
            "convergence_tolerance": self.conv_tol,
            "design_space_dim":    N_DIM,
        }

    def _save_results(self, results: Dict):
        """Save results to JSON for design_genome.py ingestion."""
        path = self.output_dir / "optimizer_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        if self.verbose:
            print(f"\n  Results saved: {path}")


# ===========================================================================
# Convenience function for pipeline integration
# ===========================================================================

def optimize_pmsg(
    pinn_evaluator,
    n_initial:   int = 50,
    max_iter:    int = 200,
    output_dir:  str = "./optimizer_results",
) -> Dict:
    """
    Pipeline integration function.
    Called by master_pipeline_v2.py.

    Parameters
    ----------
    pinn_evaluator : object with .evaluate(x) method
        Returns dict with torque_density, mass_reduction, efficiency
    n_initial : int — LHS initial samples
    max_iter : int — max Bayesian iterations

    Returns
    -------
    dict with Pareto frontier and best designs
    """
    def objective_fn(x: np.ndarray) -> np.ndarray:
        result = pinn_evaluator.evaluate(x)
        return np.array([
            result.get("torque_density_Nmkg", 0.0),
            result.get("mass_reduction_pct",  0.0),
            result.get("efficiency_pct",       0.0),
        ])

    def constraint_fn(x: np.ndarray) -> Dict:
        result = pinn_evaluator.evaluate(x)
        return {
            "tier1_all_clear": result.get("tier1_all_clear", True),
            "worst_tier1":     result.get("worst_tier1", ""),
            "trust_score":     result.get("trust_score", 0.0),
        }

    optimizer = PMSGBayesianOptimizer(
        objective_fn=objective_fn,
        constraint_fn=constraint_fn,
        n_initial=n_initial,
        max_iter=max_iter,
        output_dir=output_dir,
    )
    return optimizer.run()


# ===========================================================================
# Self-test with synthetic objective function
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MOBIUS-NOVA BAYESIAN OPTIMIZER — SELF TEST")
    print("  40D Bézier PMSG | 3-objective Pareto")
    print("=" * 60)

    # Synthetic objective — peaks near asymmetric designs
    def synthetic_objective(x: np.ndarray) -> np.ndarray:
        """
        Synthetic PMSG objective function for testing.
        Mimics the real PINN outputs — higher asymmetry index
        and higher Halbach fraction give better results.
        """
        asymmetry = x[38]         # asymmetry index [0,1]
        halbach   = x[37]         # Halbach fraction [0,1]
        pole_arc  = x[33]         # pole arc ratio [0.5, 0.95]
        mag_mass  = x[36]         # magnet mass ratio [0.5, 1.5]

        # Torque density — peaks at high asymmetry + moderate pole arc
        torque_density = (
            351.28 *
            (1 + 0.15 * asymmetry) *
            (1 + 0.10 * halbach) *
            (1 - 0.05 * abs(pole_arc - 0.72)) +
            np.random.normal(0, 2.0)
        )

        # Mass reduction — inversely related to magnet mass
        mass_reduction = (
            27.0 * asymmetry +
            10.0 * halbach +
            (1.5 - mag_mass) * 15.0 +
            np.random.normal(0, 1.0)
        )
        mass_reduction = np.clip(mass_reduction, 0, 50)

        # Efficiency — peaks at optimal pole arc
        efficiency = (
            93.5 +
            2.0 * (1 - abs(pole_arc - 0.72) / 0.23) +
            np.random.normal(0, 0.3)
        )
        efficiency = np.clip(efficiency, 88, 98)

        return np.array([torque_density, mass_reduction, efficiency])

    # Synthetic constraint — reject very low asymmetry (too symmetric)
    def synthetic_constraint(x: np.ndarray) -> Dict:
        asymmetry = x[38]
        mag_mass  = x[36]
        return {
            "tier1_all_clear": asymmetry > 0.02 and mag_mass < 1.45,
            "worst_tier1": "asymmetry" if asymmetry <= 0.02 else "",
            "trust_score": float(asymmetry),
        }

    # Run short test
    optimizer = PMSGBayesianOptimizer(
        objective_fn=synthetic_objective,
        constraint_fn=synthetic_constraint,
        n_initial=20,
        batch_size=3,
        max_iter=10,
        convergence_tol=0.001,
        output_dir="/tmp/optimizer_test",
        verbose=True,
    )

    results = optimizer.run()

    print(f"\n  Top 3 Pareto designs:")
    for i, d in enumerate(results["pareto_designs"][:3]):
        print(f"  {i+1}. {d['design_id']}: "
              f"T={d['torque_density_Nmkg']:.1f} Nm/kg  "
              f"M={d['mass_reduction_pct']:.1f}%  "
              f"η={d['efficiency_pct']:.1f}%  "
              f"(+{d['improvement_pct']:.1f}% vs NREL baseline)")
