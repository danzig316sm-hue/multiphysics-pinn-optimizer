"""
solvers/featool_solver.py
FEAToolSolver — unified solver that wraps your existing physics/ modules
and optionally uses FEATool (MATLAB/Octave) if installed.

Priority order for each domain:
  1. FEATool Python API  (pip install featool)
  2. Your existing physics/ modules  ← ALWAYS available
  3. Physics-consistent analytic stubs (last resort)

Drop-in replacement that master_multiphysics_pipeline.py expects.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

# Make sure project root is on path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from solvers.base_solver import GeometrySpec

# ---------------------------------------------------------------------------
# Try to import existing physics modules
# ---------------------------------------------------------------------------
try:
    from physics.multiphysics_orchestrator import MultiphysicsOrchestrator  # type: ignore
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

try:
    from physics.aerodynamics import AerodynamicsSolver  # type: ignore
    AERO_AVAILABLE = True
except ImportError:
    AERO_AVAILABLE = False

try:
    from physics.thermal import ThermalSolver  # type: ignore
    THERMAL_AVAILABLE = True
except ImportError:
    THERMAL_AVAILABLE = False

try:
    from physics.electromagnetics import ElectromagneticsSolver  # type: ignore
    EM_AVAILABLE = True
except ImportError:
    EM_AVAILABLE = False

try:
    from physics.structural import StructuralSolver  # type: ignore
    STRUCTURAL_AVAILABLE = True
except ImportError:
    STRUCTURAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional FEATool
# ---------------------------------------------------------------------------
try:
    import featool  # type: ignore
    FEATOOL_AVAILABLE = True
except ImportError:
    FEATOOL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional pyFEMM (EM fallback)
# ---------------------------------------------------------------------------
try:
    import femm  # type: ignore
    FEMM_AVAILABLE = True
except ImportError:
    FEMM_AVAILABLE = False


# ---------------------------------------------------------------------------
# FEAToolSolver
# ---------------------------------------------------------------------------

class FEAToolSolver:
    """
    Unified multi-physics solver adapter.

    Args:
        mesh_density: "coarse" | "medium" | "fine"
        verbose:      print solver progress
    """

    def __init__(self, mesh_density: str = "medium", verbose: bool = True):
        self.mesh_density = mesh_density
        self.verbose = verbose

        # Instantiate physics module singletons (they may be expensive to init)
        self._orchestrator = None
        self._aero = None
        self._thermal = None
        self._em = None
        self._structural = None

        if self.verbose:
            self._print_solver_status()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _print_solver_status(self):
        print("\n[FEAToolSolver] Solver availability:")
        print(f"  FEATool Python API : {'✓' if FEATOOL_AVAILABLE else '✗  (pip install featool)'}")
        print(f"  pyFEMM             : {'✓' if FEMM_AVAILABLE else '✗  (pip install pyfemm)'}")
        print(f"  MultiphysicsOrch.  : {'✓' if ORCHESTRATOR_AVAILABLE else '✗'}")
        print(f"  AerodynamicsSolver : {'✓' if AERO_AVAILABLE else '✗'}")
        print(f"  ThermalSolver      : {'✓' if THERMAL_AVAILABLE else '✗'}")
        print(f"  ElectromagneticsSolver: {'✓' if EM_AVAILABLE else '✗'}")
        print(f"  StructuralSolver   : {'✓' if STRUCTURAL_AVAILABLE else '✗'}")
        print(f"  Mesh density       : {self.mesh_density}\n")

    # ------------------------------------------------------------------
    # Lazy singleton getters
    # ------------------------------------------------------------------

    def _get_orchestrator(self):
        if self._orchestrator is None and ORCHESTRATOR_AVAILABLE:
            try:
                self._orchestrator = MultiphysicsOrchestrator(mesh_density=self.mesh_density)
            except Exception as e:
                if self.verbose:
                    print(f"[FEAToolSolver] Orchestrator init failed: {e}")
        return self._orchestrator

    def _get_aero(self):
        if self._aero is None and AERO_AVAILABLE:
            try:
                self._aero = AerodynamicsSolver()
            except Exception:
                pass
        return self._aero

    def _get_thermal(self):
        if self._thermal is None and THERMAL_AVAILABLE:
            try:
                self._thermal = ThermalSolver()
            except Exception:
                pass
        return self._thermal

    def _get_em(self):
        if self._em is None and EM_AVAILABLE:
            try:
                self._em = ElectromagneticsSolver()
            except Exception:
                pass
        return self._em

    def _get_structural(self):
        if self._structural is None and STRUCTURAL_AVAILABLE:
            try:
                self._structural = StructuralSolver()
            except Exception:
                pass
        return self._structural

    # ------------------------------------------------------------------
    # Geometry conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _geo_to_params(geometry: GeometrySpec) -> dict:
        """Convert GeometrySpec to flat dict for physics modules."""
        return {
            "bezier_curve1":       geometry.bezier_curve1,
            "bezier_curve2":       geometry.bezier_curve2,
            "bezier_curve3":       geometry.bezier_curve3,
            "ratio_parameter":     geometry.ratio_parameter,
            "magnet_grade":        geometry.magnet_grade,
            "remanence_T":         geometry.remanence_T,
            "coercivity_kAm":      geometry.coercivity_kAm,
            "rated_speed_rpm":     geometry.rated_speed_rpm,
            "rated_power_w":       geometry.rated_power_w,
            "pole_pairs":          geometry.pole_pairs,
            "stator_slots":        geometry.stator_slots,
            "dc_bus_voltage_v":    geometry.dc_bus_voltage_v,
            "phase_current_A":     geometry.phase_current_A,
            "ambient_temp_C":      geometry.ambient_temp_C,
            "coolant_flow_lpm":    geometry.coolant_flow_lpm,
            "safety_factor":       geometry.safety_factor,
        }

    # ------------------------------------------------------------------
    # CFD / Aerodynamics
    # ------------------------------------------------------------------

    def run_cfd(self, geometry: GeometrySpec, flow_conditions: Optional[dict] = None) -> dict:
        """
        Run aerodynamic / CFD simulation.
        Returns dict with keys: velocity_mps, pressure_Pa, drag_N, lift_N, solver, time_s
        """
        t0 = time.perf_counter()
        params = self._geo_to_params(geometry)
        if flow_conditions:
            params.update(flow_conditions)

        # --- 1. FEATool ---
        if FEATOOL_AVAILABLE:
            try:
                result = self._run_featool_cfd(params)
                result["solver"] = "FEATool-CFD"
                result["time_s"] = time.perf_counter() - t0
                return result
            except Exception as e:
                if self.verbose:
                    print(f"[CFD] FEATool failed ({e}), trying physics module...")

        # --- 2. Existing aerodynamics.py ---
        aero = self._get_aero()
        if aero is not None:
            try:
                result = aero.run(params)
                if isinstance(result, dict):
                    result["solver"] = "physics/aerodynamics.py"
                    result["time_s"] = time.perf_counter() - t0
                    return result
            except Exception as e:
                if self.verbose:
                    print(f"[CFD] Aerodynamics module failed ({e}), using analytic stub...")

        # --- 3. Analytic stub ---
        return self._analytic_cfd(geometry, t0)

    def _run_featool_cfd(self, params: dict) -> dict:
        """FEATool CFD — called only when featool is installed."""
        raise NotImplementedError("FEATool CFD adapter — implement with your FEATool license")

    def _analytic_cfd(self, geo: GeometrySpec, t0: float) -> dict:
        """Physics-consistent analytic estimate (no external deps)."""
        import math
        rho = 1.225          # air density kg/m³
        tip_speed = geo.rated_speed_rpm * math.pi / 30 * 1.5   # approx tip speed m/s
        q = 0.5 * rho * tip_speed ** 2
        return {
            "velocity_mps":   tip_speed,
            "pressure_Pa":    q,
            "drag_N":         q * 0.02,
            "lift_N":         q * 0.85,
            "Cp":             0.38,          # power coefficient estimate
            "solver":         "analytic-stub",
            "time_s":         time.perf_counter() - t0,
            "warning":        "Analytic stub — install FEATool or ensure physics/aerodynamics.py is importable",
        }

    # ------------------------------------------------------------------
    # Thermal
    # ------------------------------------------------------------------

    def run_thermal(self, geometry: GeometrySpec, heat_sources: Optional[dict] = None) -> dict:
        """
        Run thermal FEA.
        Returns dict with keys: T_winding_C, T_magnet_C, T_core_C, passed, solver, time_s
        """
        t0 = time.perf_counter()
        params = self._geo_to_params(geometry)
        if heat_sources:
            params.update(heat_sources)

        # --- 1. FEATool ---
        if FEATOOL_AVAILABLE:
            try:
                result = self._run_featool_thermal(params)
                result["solver"] = "FEATool-Thermal"
                result["time_s"] = time.perf_counter() - t0
                result["passed"] = self._check_thermal_targets(result, geometry)
                return result
            except Exception as e:
                if self.verbose:
                    print(f"[Thermal] FEATool failed ({e}), trying physics module...")

        # --- 2. Existing thermal.py ---
        thermal = self._get_thermal()
        if thermal is not None:
            try:
                result = thermal.run(params)
                if isinstance(result, dict):
                    result.setdefault("T_winding_C", result.get("winding_temp", result.get("T_max", 120.0)))
                    result.setdefault("T_magnet_C",  result.get("magnet_temp", 50.0))
                    result.setdefault("T_core_C",    result.get("core_temp", 80.0))
                    result["solver"] = "physics/thermal.py"
                    result["time_s"] = time.perf_counter() - t0
                    result["passed"] = self._check_thermal_targets(result, geometry)
                    return result
            except Exception as e:
                if self.verbose:
                    print(f"[Thermal] thermal.py failed ({e}), using analytic stub...")

        return self._analytic_thermal(geometry, t0)

    def _run_featool_thermal(self, params: dict) -> dict:
        raise NotImplementedError("FEATool Thermal adapter — implement with your FEATool license")

    @staticmethod
    def _check_thermal_targets(result: dict, geo: GeometrySpec) -> bool:
        tw = result.get("T_winding_C", 999)
        tm = result.get("T_magnet_C", 999)
        return tw < geo.max_winding_temp_C and tm < geo.max_magnet_temp_C

    def _analytic_thermal(self, geo: GeometrySpec, t0: float) -> dict:
        """Lumped-parameter thermal estimate."""
        copper_loss_w = geo.phase_current_A ** 2 * 0.15 * 3   # 3-phase, R≈0.15Ω
        iron_loss_w = geo.rated_power_w * 0.02
        total_loss_w = copper_loss_w + iron_loss_w
        Rth_winding = 0.8   # K/W typical
        Rth_magnet  = 1.2
        T_w = geo.ambient_temp_C + copper_loss_w * Rth_winding
        T_m = geo.ambient_temp_C + iron_loss_w   * Rth_magnet
        return {
            "T_winding_C": round(T_w, 1),
            "T_magnet_C":  round(T_m, 1),
            "T_core_C":    round(geo.ambient_temp_C + total_loss_w * 0.3, 1),
            "total_loss_W": round(total_loss_w, 1),
            "copper_loss_W": round(copper_loss_w, 1),
            "iron_loss_W":   round(iron_loss_w, 1),
            "passed":  T_w < geo.max_winding_temp_C and T_m < geo.max_magnet_temp_C,
            "solver":  "analytic-stub",
            "time_s":  time.perf_counter() - t0,
            "warning": "Analytic stub — install FEATool or ensure physics/thermal.py is importable",
        }

    # ------------------------------------------------------------------
    # Electromagnetic
    # ------------------------------------------------------------------

    def run_electromagnetic(self, geometry: GeometrySpec, materials: Optional[dict] = None) -> dict:
        """
        Run magnetostatic / EM FEA.
        Returns: efficiency_pct, cogging_Nm, Br_min_T, torque_Nm, passed, solver, time_s
        """
        t0 = time.perf_counter()
        params = self._geo_to_params(geometry)
        if materials:
            params.update(materials)

        # --- 1. FEATool ---
        if FEATOOL_AVAILABLE:
            try:
                result = self._run_featool_em(params)
                result["solver"] = "FEATool-EM"
                result["time_s"] = time.perf_counter() - t0
                result["passed"] = self._check_em_targets(result)
                return result
            except Exception as e:
                if self.verbose:
                    print(f"[EM] FEATool failed ({e}), trying physics module...")

        # --- 2. pyFEMM ---
        if FEMM_AVAILABLE:
            try:
                result = self._run_femm(params)
                result["solver"] = "pyFEMM"
                result["time_s"] = time.perf_counter() - t0
                result["passed"] = self._check_em_targets(result)
                return result
            except Exception as e:
                if self.verbose:
                    print(f"[EM] pyFEMM failed ({e}), trying physics module...")

        # --- 3. Existing electromagnetics.py ---
        em = self._get_em()
        if em is not None:
            try:
                result = em.run(params)
                if isinstance(result, dict):
                    result.setdefault("efficiency_pct", result.get("eta", result.get("efficiency", 92.0)) * 100 if result.get("eta", 1.0) <= 1.0 else result.get("eta", 92.0))
                    result.setdefault("cogging_Nm",   result.get("cogging_torque", 20.0))
                    result.setdefault("Br_min_T",     result.get("remanence_min",  0.65))
                    result["solver"] = "physics/electromagnetics.py"
                    result["time_s"] = time.perf_counter() - t0
                    result["passed"] = self._check_em_targets(result)
                    return result
            except Exception as e:
                if self.verbose:
                    print(f"[EM] electromagnetics.py failed ({e}), using analytic stub...")

        return self._analytic_em(geometry, t0)

    def _run_featool_em(self, params: dict) -> dict:
        raise NotImplementedError("FEATool EM adapter — implement with your FEATool license")

    def _run_femm(self, params: dict) -> dict:
        raise NotImplementedError("pyFEMM adapter — implement FEMM model setup")

    @staticmethod
    def _check_em_targets(result: dict) -> bool:
        eta  = result.get("efficiency_pct", 0)
        cog  = result.get("cogging_Nm", 999)
        br   = result.get("Br_min_T", 0)
        return eta >= 95.0 and cog < 25.0 and br > 0.3

    def _analytic_em(self, geo: GeometrySpec, t0: float) -> dict:
        """Magnetic circuit + efficiency estimate."""
        import math
        # Torque from power & speed
        omega = geo.rated_speed_rpm * math.pi / 30
        rated_torque = geo.rated_power_w / max(omega, 1e-6)
        # Cogging estimate: roughly 2% of rated torque for surface-mount PM
        cogging = rated_torque * 0.02
        # Efficiency: copper + iron losses
        copper_loss = geo.phase_current_A ** 2 * 0.15 * 3
        iron_loss   = geo.rated_power_w * 0.02
        eta = geo.rated_power_w / (geo.rated_power_w + copper_loss + iron_loss) * 100
        return {
            "efficiency_pct": round(eta, 2),
            "cogging_Nm":     round(cogging, 2),
            "Br_min_T":       geo.remanence_T * 0.85,   # accounting for geometry
            "torque_Nm":      round(rated_torque, 2),
            "flux_density_T": geo.remanence_T * 0.7,
            "passed":         eta >= 95.0 and cogging < 25.0 and geo.remanence_T * 0.85 > 0.3,
            "solver":         "analytic-stub",
            "time_s":         time.perf_counter() - t0,
            "warning":        "Analytic stub — install FEATool/pyFEMM or ensure physics/electromagnetics.py is importable",
        }

    # ------------------------------------------------------------------
    # Structural
    # ------------------------------------------------------------------

    def run_structural(self, geometry: GeometrySpec) -> dict:
        """Run structural / FEA stress analysis."""
        t0 = time.perf_counter()
        params = self._geo_to_params(geometry)

        struct = self._get_structural()
        if struct is not None:
            try:
                result = struct.run(params)
                if isinstance(result, dict):
                    result["solver"] = "physics/structural.py"
                    result["time_s"] = time.perf_counter() - t0
                    return result
            except Exception as e:
                if self.verbose:
                    print(f"[Structural] structural.py failed ({e}), using stub...")

        # Analytic stub
        import math
        omega = geometry.rated_speed_rpm * math.pi / 30
        centrifugal_stress = 6200.0 * (0.05 ** 2) * (omega ** 2)   # ρ·r²·ω²
        return {
            "max_stress_Pa":    round(centrifugal_stress, 0),
            "safety_factor":    geometry.safety_factor,
            "mass_kg":          round(geometry.rated_power_w / 1000 * 2.5, 1),
            "passed":           centrifugal_stress < 50e6,
            "solver":           "analytic-stub",
            "time_s":           time.perf_counter() - t0,
        }

    # ------------------------------------------------------------------
    # Run all domains
    # ------------------------------------------------------------------

    def run_all(self, geometry: GeometrySpec) -> dict:
        """
        Run all physics domains. Uses MultiphysicsOrchestrator if available,
        otherwise calls each domain individually and merges results.
        """
        t0 = time.perf_counter()

        # --- Try orchestrator first (most efficient) ---
        if ORCHESTRATOR_AVAILABLE:
            orch = self._get_orchestrator()
            if orch is not None:
                try:
                    params = self._geo_to_params(geometry)
                    result = orch.run_all(params)
                    if isinstance(result, dict):
                        result["solver"] = "MultiphysicsOrchestrator"
                        result["total_time_s"] = time.perf_counter() - t0
                        result["design_hash"] = geometry.design_hash()
                        if self.verbose:
                            self._print_results(result)
                        return result
                except Exception as e:
                    if self.verbose:
                        print(f"[run_all] Orchestrator failed ({e}), running domains individually...")

        # --- Run individually ---
        cfd        = self.run_cfd(geometry)
        thermal    = self.run_thermal(geometry)
        em         = self.run_electromagnetic(geometry)
        structural = self.run_structural(geometry)

        results = {
            "aerodynamic":      cfd,
            "thermal":          thermal,
            "electromagnetic":  em,
            "structural":       structural,
            "all_passed":       all([
                thermal.get("passed", False),
                em.get("passed", False),
                structural.get("passed", False),
            ]),
            "total_time_s":    time.perf_counter() - t0,
            "design_hash":     geometry.design_hash(),
            "solver":          "FEAToolSolver-sequential",
        }

        if self.verbose:
            self._print_results(results)

        return results

    def _print_results(self, results: dict):
        em  = results.get("electromagnetic", {})
        th  = results.get("thermal", {})
        print(f"\n[FEAToolSolver] Results summary:")
        print(f"  η         = {em.get('efficiency_pct', '—'):.1f}%   (target ≥ 95%)")
        print(f"  Cogging   = {em.get('cogging_Nm', '—'):.1f} N·m  (target < 25)")
        print(f"  Br_min    = {em.get('Br_min_T', '—'):.3f} T  (target > 0.3)")
        print(f"  T_winding = {th.get('T_winding_C', '—'):.1f}°C  (target < 180)")
        print(f"  T_magnet  = {th.get('T_magnet_C', '—'):.1f}°C  (target < 60)")
        print(f"  All pass  = {results.get('all_passed', '—')}")
        print(f"  Time      = {results.get('total_time_s', 0):.2f}s")
