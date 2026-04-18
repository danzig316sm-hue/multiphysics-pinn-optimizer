"""
nvtx_integration.py
====================
NVTX annotation layer for the Mobius-Nova multiphysics PINN pipeline.

What this does:
  Wraps every major pipeline operation with NVTX range markers so that
  Nsight Systems, Nsight Compute, and Nsight Deep Learning Designer can
  show labeled performance timelines for every training run.

Usage — add to the top of any module:
  from utils.nvtx_integration import nvtx_range, PMSGNVTXRanges, pipeline_annotated

The markers show up as colored, labeled blocks in the Nsight profiler timeline.
No GPU required for the markers themselves — they work on CPU-only runs too
and simply become no-ops when NVIDIA profiling tools are not attached.

Color coding (consistent across all pipeline modules):
  GREEN    — Physics constraint evaluation (all 22 constraints)
  BLUE     — PINN forward pass / neural network operations
  ORANGE   — Self-correction decisions
  RED      — Tier-1 hard limit violations
  PURPLE   — Design genome read/write operations
  CYAN     — CFD/thermal/EM solver calls
  YELLOW   — Halbach field geometry computation
  WHITE    — QBlade aerodynamic evaluation
  MAGENTA  — Trust score engine updates

References:
  NVTX Python API: pip install nvtx
  Nsight Systems:  developer.nvidia.com/nsight-systems
  Nsight Compute:  developer.nvidia.com/nsight-compute
  Nsight DL Designer: developer.nvidia.com/nsight-dl-designer
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Callable, Optional

# NVTX import with graceful fallback
# When Nsight profiler is not attached, all annotations are zero-overhead no-ops
try:
    import nvtx
    _NVTX_AVAILABLE = True
except ImportError:
    _NVTX_AVAILABLE = False
    # Create stub so code runs unchanged without nvtx installed
    class _NVTXStub:
        def push_range(self, *a, **kw): pass
        def pop_range(self, *a, **kw): pass
        def annotate(self, *a, **kw):
            def decorator(fn): return fn
            return decorator
        class Profile:
            def __enter__(self): return self
            def __exit__(self, *a): pass
    nvtx = _NVTXStub()


# ===========================================================================
# Color constants — consistent across all pipeline modules
# ===========================================================================

class NVTXColor:
    """ARGB color constants for NVTX range markers."""
    GREEN   = 0xFF00FF00   # Physics constraint evaluation
    BLUE    = 0xFF0000FF   # PINN forward pass
    ORANGE  = 0xFFFF8C00   # Self-correction decisions
    RED     = 0xFFFF0000   # Tier-1 hard limit violations
    PURPLE  = 0xFF800080   # Design genome operations
    CYAN    = 0xFF00FFFF   # CFD/thermal/EM solver calls
    YELLOW  = 0xFFFFFF00   # Halbach field geometry
    WHITE   = 0xFFFFFFFF   # QBlade aerodynamic evaluation
    MAGENTA = 0xFFFF00FF   # Trust score engine
    GRAY    = 0xFF808080   # General / uncategorized


# ===========================================================================
# Core annotation utilities
# ===========================================================================

@contextmanager
def nvtx_range(name: str, color: int = NVTXColor.GRAY, domain: str = "MobiusNova"):
    """
    Context manager for NVTX range annotation.

    Usage:
        with nvtx_range("tier1_constraints", NVTXColor.GREEN):
            # evaluate all 7 Tier-1 constraints
            ...

    In Nsight Systems this shows as a labeled green block in the CPU timeline.
    """
    if _NVTX_AVAILABLE:
        nvtx.push_range(name, color=color, domain=domain)
    try:
        yield
    finally:
        if _NVTX_AVAILABLE:
            nvtx.pop_range()


def nvtx_annotate(name: Optional[str] = None, color: int = NVTXColor.GRAY):
    """
    Decorator for NVTX annotation of functions.

    Usage:
        @nvtx_annotate("pinn_forward_pass", NVTXColor.BLUE)
        def forward(self, x):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        label = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with nvtx_range(label, color):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


# ===========================================================================
# Pipeline-specific NVTX ranges
# ===========================================================================

class PMSGNVTXRanges:
    """
    Pre-defined NVTX ranges for every major operation in the PMSG pipeline.

    Use these as context managers to annotate the training loop, physics
    evaluation, and self-correction decisions consistently.

    All ranges are visible in Nsight Systems as labeled timeline blocks.
    """

    # ── PINN Training ─────────────────────────────────────────────────────────

    @staticmethod
    @contextmanager
    def training_epoch(epoch: int):
        with nvtx_range(f"epoch_{epoch:04d}", NVTXColor.BLUE):
            yield

    @staticmethod
    @contextmanager
    def forward_pass():
        with nvtx_range("pinn_forward_pass", NVTXColor.BLUE):
            yield

    @staticmethod
    @contextmanager
    def backward_pass():
        with nvtx_range("pinn_backward_pass", NVTXColor.BLUE):
            yield

    @staticmethod
    @contextmanager
    def loss_computation():
        with nvtx_range("total_loss_computation", NVTXColor.BLUE):
            yield

    # ── Physics Constraints ───────────────────────────────────────────────────

    @staticmethod
    @contextmanager
    def physics_loss_all():
        """Wraps the full 22-constraint physics loss computation."""
        with nvtx_range("physics_loss_22_constraints", NVTXColor.GREEN):
            yield

    @staticmethod
    @contextmanager
    def tier1_constraints():
        """7 hard limits — demagnetisation, axial_stiffness, etc."""
        with nvtx_range("tier1_hard_limits", NVTXColor.GREEN):
            yield

    @staticmethod
    @contextmanager
    def tier2_constraints():
        """6 performance targets — cogging, THD, efficiency, etc."""
        with nvtx_range("tier2_performance_targets", NVTXColor.GREEN):
            yield

    @staticmethod
    @contextmanager
    def tier3_constraints():
        """6 Pareto objectives — torque density, mass reduction, etc."""
        with nvtx_range("tier3_pareto_objectives", NVTXColor.GREEN):
            yield

    @staticmethod
    @contextmanager
    def tier4_constraints():
        """4 coupling checks — copper balance, radial stiffness, etc."""
        with nvtx_range("tier4_coupling_checks", NVTXColor.GREEN):
            yield

    @staticmethod
    @contextmanager
    def constraint(name: str):
        """Individual constraint evaluation — use inside tier context."""
        with nvtx_range(f"constraint_{name}", NVTXColor.GREEN):
            yield

    # ── Self-Correction ───────────────────────────────────────────────────────

    @staticmethod
    @contextmanager
    def self_correction_step(epoch: int):
        with nvtx_range(f"self_correction_epoch_{epoch:04d}", NVTXColor.ORANGE):
            yield

    @staticmethod
    @contextmanager
    def tier1_violation(constraint_name: str):
        """Red marker — visible immediately in Nsight as a problem epoch."""
        with nvtx_range(f"TIER1_VIOLATION_{constraint_name}", NVTXColor.RED):
            yield

    @staticmethod
    @contextmanager
    def physics_weight_boost(old_w: float, new_w: float):
        with nvtx_range(
            f"weight_boost_{old_w:.3f}_to_{new_w:.3f}", NVTXColor.ORANGE
        ):
            yield

    @staticmethod
    @contextmanager
    def checkpoint_restore():
        with nvtx_range("checkpoint_restore", NVTXColor.ORANGE):
            yield

    # ── CFD / Thermal / EM ────────────────────────────────────────────────────

    @staticmethod
    @contextmanager
    def cfd_evaluation():
        with nvtx_range("cfd_thermal_coupler_full", NVTXColor.CYAN):
            yield

    @staticmethod
    @contextmanager
    def centrifugal_pump():
        with nvtx_range("cfd_centrifugal_pump_head", NVTXColor.CYAN):
            yield

    @staticmethod
    @contextmanager
    def ram_pressure():
        with nvtx_range("cfd_ram_pressure_intake", NVTXColor.CYAN):
            yield

    @staticmethod
    @contextmanager
    def thermal_magnet():
        """Tachibana-Fukui Taylor-Couette convection computation."""
        with nvtx_range("thermal_taylor_couette_magnet", NVTXColor.CYAN):
            yield

    @staticmethod
    @contextmanager
    def thermal_winding():
        """Newton cooling winding temperature computation."""
        with nvtx_range("thermal_newton_cooling_winding", NVTXColor.CYAN):
            yield

    @staticmethod
    @contextmanager
    def iron_loss(material: str = "m15"):
        """Steinmetz core loss computation."""
        with nvtx_range(f"em_iron_loss_steinmetz_{material}", NVTXColor.CYAN):
            yield

    @staticmethod
    @contextmanager
    def copper_loss(fill_factor: float = 0.44):
        with nvtx_range(f"em_copper_loss_kfill{fill_factor:.2f}", NVTXColor.CYAN):
            yield

    # ── Halbach Field Geometry ────────────────────────────────────────────────

    @staticmethod
    @contextmanager
    def halbach_full_analysis():
        with nvtx_range("halbach_magpylib_full_analysis", NVTXColor.YELLOW):
            yield

    @staticmethod
    @contextmanager
    def halbach_build_array(n_segments: int):
        with nvtx_range(f"halbach_build_{n_segments}_segments", NVTXColor.YELLOW):
            yield

    @staticmethod
    @contextmanager
    def halbach_field_computation(n_points: int):
        with nvtx_range(f"halbach_getB_{n_points}_points", NVTXColor.YELLOW):
            yield

    @staticmethod
    @contextmanager
    def halbach_inter_pole_analysis():
        """Salcuni 2025 transition zone quantification."""
        with nvtx_range("halbach_inter_pole_transition_zones", NVTXColor.YELLOW):
            yield

    # ── Design Genome ─────────────────────────────────────────────────────────

    @staticmethod
    @contextmanager
    def genome_write(design_id: str = ""):
        with nvtx_range(f"genome_write_{design_id[:8]}", NVTXColor.PURPLE):
            yield

    @staticmethod
    @contextmanager
    def genome_pareto_update():
        with nvtx_range("genome_pareto_front_update", NVTXColor.PURPLE):
            yield

    @staticmethod
    @contextmanager
    def genome_similarity_search():
        with nvtx_range("genome_similarity_search", NVTXColor.PURPLE):
            yield

    # ── Trust Score Engine ────────────────────────────────────────────────────

    @staticmethod
    @contextmanager
    def trust_score_update(constraint: str = ""):
        with nvtx_range(f"trust_wilson_score_{constraint}", NVTXColor.MAGENTA):
            yield

    @staticmethod
    @contextmanager
    def trust_verdict():
        with nvtx_range("trust_design_verdict", NVTXColor.MAGENTA):
            yield

    # ── QBlade Aerodynamic ────────────────────────────────────────────────────

    @staticmethod
    @contextmanager
    def qblade_bem():
        with nvtx_range("qblade_BEM_evaluation", NVTXColor.WHITE):
            yield

    @staticmethod
    @contextmanager
    def qblade_dms():
        with nvtx_range("qblade_DMS_evaluation", NVTXColor.WHITE):
            yield

    @staticmethod
    @contextmanager
    def qblade_polar_parse():
        with nvtx_range("qblade_polar_parse_qpr", NVTXColor.WHITE):
            yield

    # ── CAD / Geometry ────────────────────────────────────────────────────────

    @staticmethod
    @contextmanager
    def cad_step_export():
        with nvtx_range("cad_cadquery_step_export", NVTXColor.GRAY):
            yield

    @staticmethod
    @contextmanager
    def biot_savart():
        with nvtx_range("cad_biot_savart_field", NVTXColor.GRAY):
            yield

    @staticmethod
    @contextmanager
    def freecad_fem():
        with nvtx_range("freecad_fem_structural", NVTXColor.GRAY):
            yield


# ===========================================================================
# Annotated training loop wrapper
# ===========================================================================

def pipeline_annotated(training_fn: Callable) -> Callable:
    """
    Decorator that adds full NVTX annotation to a training loop function.

    Usage:
        @pipeline_annotated
        def train_epoch(model, optimizer, loader, epoch):
            ...

    The decorated function will appear in Nsight Systems with:
      - Blue block for the entire epoch
      - Green block for physics constraint evaluation
      - Orange block for any self-correction actions
      - Red blocks for any Tier-1 violations
    """
    @functools.wraps(training_fn)
    def wrapper(*args, **kwargs):
        epoch = kwargs.get('epoch', args[3] if len(args) > 3 else 0)
        with PMSGNVTXRanges.training_epoch(epoch):
            return training_fn(*args, **kwargs)
    return wrapper


# ===========================================================================
# Performance timing utility (complements Nsight profiling)
# ===========================================================================

class PipelineTimer:
    """
    Lightweight CPU-side timer that complements Nsight GPU profiling.

    Records wall-clock time for each named pipeline stage and reports
    what fraction of total training time each stage consumes.

    Use alongside NVTX ranges — the timer gives you aggregate stats,
    Nsight gives you the detailed per-kernel breakdown.
    """

    def __init__(self):
        self._times: dict = {}
        self._counts: dict = {}
        self._start: dict = {}

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._times[name] = self._times.get(name, 0.0) + elapsed
            self._counts[name] = self._counts.get(name, 0) + 1

    def report(self) -> str:
        total = sum(self._times.values())
        if total == 0:
            return "No timing data recorded."

        lines = [
            "=" * 60,
            "  PIPELINE TIMING REPORT",
            f"  Total wall time: {total:.2f}s",
            "=" * 60,
        ]
        for name, t in sorted(self._times.items(), key=lambda x: -x[1]):
            pct = t / total * 100
            count = self._counts[name]
            avg_ms = t / count * 1000
            lines.append(
                f"  {name:<35} {pct:5.1f}%  "
                f"{t:6.2f}s  ({count}x, avg {avg_ms:.1f}ms)"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    print("NVTX Integration — Mobius-Nova PMSG Pipeline")
    print(f"NVTX available: {_NVTX_AVAILABLE}")
    print()

    timer = PipelineTimer()

    # Simulate a training epoch with full annotations
    with PMSGNVTXRanges.training_epoch(1):
        with timer.measure("forward_pass"):
            with PMSGNVTXRanges.forward_pass():
                time.sleep(0.01)

        with timer.measure("physics_constraints"):
            with PMSGNVTXRanges.physics_loss_all():
                with PMSGNVTXRanges.tier1_constraints():
                    with timer.measure("tier1"):
                        for c in ["demagnetisation", "axial_stiffness",
                                  "torque_adequacy", "bond_stress",
                                  "cfd_cooling", "thermal_magnet_proven",
                                  "thermal_winding_proven"]:
                            with PMSGNVTXRanges.constraint(c):
                                time.sleep(0.001)

                with PMSGNVTXRanges.tier2_constraints():
                    with timer.measure("tier2"):
                        time.sleep(0.005)

                with PMSGNVTXRanges.tier3_constraints():
                    with timer.measure("tier3"):
                        with PMSGNVTXRanges.halbach_inter_pole_analysis():
                            time.sleep(0.003)

                with PMSGNVTXRanges.tier4_constraints():
                    with timer.measure("tier4"):
                        time.sleep(0.002)

        with timer.measure("self_correction"):
            with PMSGNVTXRanges.self_correction_step(1):
                time.sleep(0.002)

        with timer.measure("genome_write"):
            with PMSGNVTXRanges.genome_write("abc12345"):
                time.sleep(0.005)

        with timer.measure("trust_score"):
            with PMSGNVTXRanges.trust_verdict():
                time.sleep(0.001)

    print(timer.report())
    print("\nAll NVTX ranges completed successfully.")
    print("Run under 'nsys profile python3 your_training_script.py'")
    print("to see labeled timeline in Nsight Systems.")
