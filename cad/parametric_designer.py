"""
Parametric CAD Designer — CadQuery-based geometry generation.

Creates parametric coil and structural geometries that feed into the
magnetic analyzer, FreeCAD FEM bridge, and PINN optimizer.

Dependencies:
    pip install cadquery
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import cadquery as cq
    _CQ_AVAILABLE = True
except ImportError:
    _CQ_AVAILABLE = False
    print("[ParametricDesigner] CadQuery not found. Install with: pip install cadquery")


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CoilParams:
    """Parameters for a wound coil geometry."""
    radius: float           # coil winding radius (mm)
    height: float           # solenoid axial length (mm)
    turns: int              # number of wire turns
    wire_diameter: float    # wire cross-section diameter (mm)
    core_radius: float = 0.0        # inner core radius; 0 = air core (mm)
    layer_count: int = 1            # number of winding layers
    pitch_ratio: float = 1.05       # turn pitch / wire_diameter (1.0 = touching)

    @property
    def pitch(self) -> float:
        return self.wire_diameter * self.pitch_ratio

    @property
    def wire_length_estimate(self) -> float:
        """Approximate total wire length in mm."""
        return self.turns * 2 * np.pi * self.radius * self.layer_count


@dataclass
class StructuralParams:
    """Parameters for a structural former / housing."""
    outer_radius: float     # outer radius (mm)
    inner_radius: float     # inner radius (0 = solid) (mm)
    height: float           # axial height (mm)
    wall_thickness: float   # wall thickness for shells (mm)
    n_ribs: int = 0         # number of longitudinal stiffening ribs
    rib_height: float = 2.0 # rib protrusion height (mm)
    rib_width: float = 1.5  # rib width (mm)
    fillet_radius: float = 0.5  # edge fillet radius (mm)


@dataclass
class HelmholtzParams:
    """Parameters for a Helmholtz (or Maxwell) coil pair."""
    coil_radius: float      # radius of each coil (mm)
    separation: float       # centre-to-centre axial separation (mm)
    turns: int              # turns per coil
    wire_diameter: float    # wire diameter (mm)
    current: float = 1.0    # operating current (A)


# ---------------------------------------------------------------------------
# Coil designer
# ---------------------------------------------------------------------------

class CoilDesigner:
    """
    Generates CadQuery solid models for wound coils.

    All dimensions in millimetres.
    """

    def _require_cq(self):
        if not _CQ_AVAILABLE:
            raise ImportError("CadQuery is required. Install with: pip install cadquery")

    # ------------------------------------------------------------------
    # Geometry builders
    # ------------------------------------------------------------------

    def solenoid_former(self, params: CoilParams) -> "cq.Workplane":
        """
        Build a cylindrical winding former (the bobbin) for a solenoid.

        Returns the solid body — the actual wire is represented analytically
        in MagneticAnalyzer rather than as individual CadQuery helices,
        which keeps mesh complexity manageable.
        """
        self._require_cq()

        body = (
            cq.Workplane("XY")
            .cylinder(params.height, params.radius + params.wire_diameter * params.layer_count)
        )

        # Subtract core if air-core (inner bore)
        if params.core_radius > 0:
            bore = cq.Workplane("XY").cylinder(params.height * 1.01, params.core_radius)
            body = body.cut(bore)

        return body

    def helmholtz_assembly(self, params: HelmholtzParams) -> "cq.Workplane":
        """
        Build a two-coil Helmholtz assembly — two thin torus rings separated
        axially, mounted on a shared axis.
        """
        self._require_cq()

        ring_height = params.wire_diameter * 2  # simplified flat ring

        def make_ring(z_offset: float) -> "cq.Workplane":
            return (
                cq.Workplane("XY")
                .workplane(offset=z_offset)
                .circle(params.coil_radius + params.wire_diameter)
                .circle(params.coil_radius - params.wire_diameter)
                .extrude(ring_height)
            )

        top = make_ring(params.separation / 2)
        bottom = make_ring(-params.separation / 2 - ring_height)
        return top.union(bottom)

    def custom_coil(
        self,
        wire_path: np.ndarray,
        wire_diameter: float,
    ) -> "cq.Workplane":
        """
        Sweep a circular cross-section along an arbitrary 3-D wire path.

        Args:
            wire_path: (N, 3) array of path points in mm.
            wire_diameter: Diameter of the circular wire cross-section (mm).

        Returns:
            CadQuery Workplane with the swept solid.
        """
        self._require_cq()

        pts = [cq.Vector(*p) for p in wire_path]
        spline = cq.Edge.makeSpline(pts)
        wire_edge = cq.Wire.assembleEdges([spline])

        profile = (
            cq.Workplane("XY")
            .workplane(offset=wire_path[0][2])
            .circle(wire_diameter / 2)
        )

        return profile.sweep(wire_edge, isFrenet=True)

    # ------------------------------------------------------------------
    # Structural formers
    # ------------------------------------------------------------------

    def hollow_cylinder(self, params: StructuralParams) -> "cq.Workplane":
        """Thin-walled cylindrical former, optionally with longitudinal ribs."""
        self._require_cq()

        outer = cq.Workplane("XY").cylinder(params.height, params.outer_radius)
        inner = cq.Workplane("XY").cylinder(
            params.height * 1.01, params.inner_radius
        )
        body = outer.cut(inner)

        if params.n_ribs > 0:
            body = self._add_ribs(body, params)

        if params.fillet_radius > 0:
            try:
                body = body.edges("|Z").fillet(params.fillet_radius)
            except Exception:
                pass  # fillet may fail on complex geometry; skip silently

        return body

    def _add_ribs(self, body: "cq.Workplane", params: StructuralParams) -> "cq.Workplane":
        """Add longitudinal stiffening ribs to a cylindrical body."""
        angle_step = 360.0 / params.n_ribs
        rib_r = params.outer_radius + params.rib_height

        for i in range(params.n_ribs):
            angle_rad = np.radians(i * angle_step)
            cx = np.cos(angle_rad) * params.outer_radius
            cy = np.sin(angle_rad) * params.outer_radius

            rib = (
                cq.Workplane("XY")
                .transformed(offset=cq.Vector(cx, cy, 0),
                              rotate=cq.Vector(0, 0, np.degrees(angle_rad)))
                .box(params.rib_width, params.rib_height, params.height,
                     centered=(True, False, True))
            )
            body = body.union(rib)

        return body

    # ------------------------------------------------------------------
    # Volume / mass utilities
    # ------------------------------------------------------------------

    def volume_mm3(self, shape: "cq.Workplane") -> float:
        """Return volume of the solid in mm³."""
        self._require_cq()
        return shape.val().Volume()

    def mass_kg(self, shape: "cq.Workplane", density_kg_per_m3: float) -> float:
        """Return mass in kg given material density."""
        vol_m3 = self.volume_mm3(shape) * 1e-9  # mm³ → m³
        return vol_m3 * density_kg_per_m3

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_step(self, shape: "cq.Workplane", path: str) -> str:
        """Export to STEP format. Returns the absolute path."""
        self._require_cq()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cq.exporters.export(shape, path)
        return os.path.abspath(path)

    def export_stl(
        self,
        shape: "cq.Workplane",
        path: str,
        tolerance: float = 0.01,
        angular_tolerance: float = 0.1,
    ) -> str:
        """Export to STL format. Returns the absolute path."""
        self._require_cq()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cq.exporters.export(
            shape, path,
            exportType="STL",
            tolerance=tolerance,
            angularTolerance=angular_tolerance,
        )
        return os.path.abspath(path)

    def export_dxf(self, shape: "cq.Workplane", path: str, section_z: float = 0.0) -> str:
        """Export a cross-section at z=section_z as DXF."""
        self._require_cq()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        section = shape.section(section_z)
        cq.exporters.export(section, path)
        return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Parameter sweep helper
# ---------------------------------------------------------------------------

def generate_parameter_grid(
    param_ranges: dict[str, tuple[float, float]],
    n_samples: int = 20,
    method: str = "latin_hypercube",
) -> list[dict[str, float]]:
    """
    Generate a grid of parameter combinations for batch evaluation.

    Args:
        param_ranges: Dict of {param_name: (min_val, max_val)}.
        n_samples: Number of samples to generate.
        method: "latin_hypercube" (default) or "uniform_grid".

    Returns:
        List of parameter dicts.
    """
    keys = list(param_ranges.keys())
    bounds = np.array([param_ranges[k] for k in keys])

    if method == "latin_hypercube":
        # Simple LHS: divide each dimension into n_samples bins, shuffle
        samples = np.zeros((n_samples, len(keys)))
        for i in range(len(keys)):
            lo, hi = bounds[i]
            perm = np.random.permutation(n_samples)
            samples[:, i] = lo + (perm + np.random.uniform(size=n_samples)) / n_samples * (hi - lo)
    else:
        # Uniform random
        samples = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(n_samples, len(keys))
        )

    return [dict(zip(keys, row)) for row in samples]
