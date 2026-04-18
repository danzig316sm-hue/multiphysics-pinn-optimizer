"""
solvers/base_solver.py
GeometrySpec — structured input for the multiphysics pipeline.

Bridges to:
  - utils/bezier_geometry.py  (Bézier curve evaluation)
  - CAD/parametric_designer.py (parametric geometry)
  - CAD/freecad_bridge.py      (STL/STP export)

Usage:
    from solvers.base_solver import GeometrySpec
    geo = GeometrySpec(
        bezier_curve1=[0.5]*6,
        bezier_curve2=[0.3,0.4,0.5,0.5,0.4,0.3],
        ...
    )
    print(geo.design_hash())
    tensor = geo.to_tensor()
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Optional bridge imports — graceful degradation if not on path
# ---------------------------------------------------------------------------
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.bezier_geometry import BezierCurve  # type: ignore
    BEZIER_AVAILABLE = True
except ImportError:
    BEZIER_AVAILABLE = False

try:
    from CAD.parametric_designer import ParametricDesigner  # type: ignore
    PARAMETRIC_AVAILABLE = True
except ImportError:
    PARAMETRIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# GeometrySpec
# ---------------------------------------------------------------------------

@dataclass
class GeometrySpec:
    """
    Structured geometry description for a single design candidate.

    Fields map to the 36-dim input vector used by TurboQuantPINN:
        - 18 Bézier control points (3 curves × 6 points)
        - 1  ratio_parameter
        - up to 17 material / operating condition scalars

    All numeric lists are stored as plain Python floats for JSON-safety.
    """

    # --- Bézier curves (6 control points each) ---
    bezier_curve1: List[float] = field(default_factory=lambda: [0.5] * 6)
    bezier_curve2: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.5, 0.4, 0.3])
    bezier_curve3: List[float] = field(default_factory=lambda: [0.2, 0.25, 0.3, 0.3, 0.25, 0.2])

    # --- Scalar design parameters ---
    ratio_parameter: float = 80.0          # pole arc ratio (%)

    # --- Material specification ---
    magnet_grade: str = "NdFeB_bonded_75vol"
    remanence_T: float = 0.72              # Br at 20°C (T)
    coercivity_kAm: float = 550.0          # Hcb (kA/m)
    magnet_density_kgm3: float = 6200.0    # kg/m³

    # --- Electrical / operating ---
    rated_speed_rpm: float = 150.0
    rated_power_w: float = 15000.0
    pole_pairs: int = 30                   # 60-pole → 30 pairs
    stator_slots: int = 60
    dc_bus_voltage_v: float = 480.0
    phase_current_A: float = 20.0

    # --- Thermal boundary conditions ---
    ambient_temp_C: float = 25.0
    coolant_flow_lpm: float = 0.0          # 0 = passive cooling
    max_winding_temp_C: float = 180.0
    max_magnet_temp_C: float = 60.0

    # --- Structural ---
    safety_factor: float = 2.0
    material_utilisation_target: float = 0.85   # drive towards 85% material use

    # --- Geometry metadata ---
    source_file: Optional[str] = None      # original STL/STP path
    created_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    def to_vector(self) -> List[float]:
        """Flatten to 36-dim float vector (PINN input format)."""
        vec = (
            list(self.bezier_curve1[:6])
            + list(self.bezier_curve2[:6])
            + list(self.bezier_curve3[:6])
            + [
                self.ratio_parameter / 100.0,          # normalise 0–1
                self.remanence_T,
                self.coercivity_kAm / 1000.0,          # → MA/m
                self.magnet_density_kgm3 / 8000.0,     # normalise
                self.rated_speed_rpm / 1500.0,
                self.rated_power_w / 100000.0,
                float(self.pole_pairs) / 50.0,
                float(self.stator_slots) / 100.0,
                self.dc_bus_voltage_v / 1000.0,
                self.phase_current_A / 100.0,
                self.ambient_temp_C / 100.0,
                self.coolant_flow_lpm / 10.0,
                self.safety_factor / 5.0,
                self.material_utilisation_target,
                # magnet_grade encoded as binary flags (2 bits)
                float("NdFeB" in self.magnet_grade),
                float("bonded" in self.magnet_grade),
            ]
        )
        # Pad / truncate to exactly 36
        vec = vec[:36] + [0.0] * max(0, 36 - len(vec))
        return vec

    def to_tensor(self, device: str = "cpu"):
        """Return (1, 36) float32 tensor."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed — cannot create tensor.")
        import torch
        return torch.tensor(self.to_vector(), dtype=torch.float32, device=device).unsqueeze(0)

    def design_hash(self) -> str:
        """Short deterministic hash of the geometry (for caching / logging)."""
        payload = json.dumps(self.to_vector(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometrySpec":
        # drop unknown keys for forward-compat
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str) -> "GeometrySpec":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    # ------------------------------------------------------------------
    # Bézier evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_bezier(self, curve_index: int = 1, n_points: int = 50) -> np.ndarray:
        """
        Evaluate one of the three Bézier curves at n_points.
        Uses utils/bezier_geometry.BezierCurve if available,
        otherwise falls back to De Casteljau.
        """
        curves = [self.bezier_curve1, self.bezier_curve2, self.bezier_curve3]
        ctrl = np.array(curves[curve_index - 1])

        if BEZIER_AVAILABLE:
            try:
                bc = BezierCurve(ctrl)
                t = np.linspace(0, 1, n_points)
                return bc.evaluate(t)
            except Exception:
                pass

        # De Casteljau fallback
        t = np.linspace(0, 1, n_points)
        n = len(ctrl) - 1
        result = np.zeros(n_points)
        for i, ti in enumerate(t):
            pts = ctrl.copy().astype(float)
            for r in range(1, n + 1):
                pts[:n - r + 1] = (1 - ti) * pts[:n - r + 1] + ti * pts[1:n - r + 2]
            result[i] = pts[0]
        return result

    # ------------------------------------------------------------------
    # FreeCAD export
    # ------------------------------------------------------------------

    def export_freecad(self, output_path: str, fmt: str = "stl") -> str:
        """
        Export geometry to STL/STP via CAD/freecad_bridge.py if available.
        Falls back to saving a JSON spec file.
        """
        output_path = str(output_path)
        if PARAMETRIC_AVAILABLE:
            try:
                designer = ParametricDesigner()
                return designer.export(self.to_dict(), output_path, fmt=fmt)
            except Exception as e:
                print(f"[GeometrySpec] ParametricDesigner export failed: {e} — saving JSON fallback")

        fallback = output_path.replace(f".{fmt}", "_spec.json")
        self.save_json(fallback)
        print(f"[GeometrySpec] Spec saved to {fallback}")
        return fallback

    def __repr__(self) -> str:
        return (
            f"GeometrySpec(hash={self.design_hash()}, "
            f"power={self.rated_power_w/1000:.0f}kW, "
            f"rpm={self.rated_speed_rpm}, "
            f"magnet={self.magnet_grade})"
        )
