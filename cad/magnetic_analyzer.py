"""
Magnetic Field Analyzer — Biot-Savart field computation and coil efficiency metrics.

Computes B-field distributions for arbitrary coil geometries, flux through
target regions, and efficiency metrics used by the optimizer.

All spatial units: metres (SI).  Input coil paths from ParametricDesigner
are in mm and must be converted before use (use CoilPath.from_mm()).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


MU_0 = 4.0 * np.pi * 1e-7   # permeability of free space (H/m)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CoilPath:
    """
    Discretised wire path for Biot-Savart integration.

    Attributes:
        points:  (N, 3) array of path vertices in metres.
        current: Current in amperes.
    """
    points: np.ndarray   # (N, 3) in metres
    current: float       # amperes

    @classmethod
    def from_mm(cls, points_mm: np.ndarray, current: float) -> "CoilPath":
        """Construct from millimetre coordinates."""
        return cls(points=np.asarray(points_mm, dtype=float) * 1e-3, current=current)

    @property
    def n_segments(self) -> int:
        return len(self.points) - 1

    @property
    def wire_length(self) -> float:
        """Total wire length in metres."""
        diffs = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))


@dataclass
class FieldResult:
    """Computed magnetic field over a grid of evaluation points."""
    points: np.ndarray    # (M, 3) evaluation points in metres
    B: np.ndarray         # (M, 3) field vectors in Tesla
    coil_path: CoilPath

    @property
    def B_magnitude(self) -> np.ndarray:
        """(M,) field magnitude in Tesla."""
        return np.linalg.norm(self.B, axis=1)

    @property
    def B_axial(self) -> np.ndarray:
        """(M,) z-component of B (axial field) in Tesla."""
        return self.B[:, 2]

    def uniformity(self) -> float:
        """
        Field uniformity over the evaluation region.
        Returns std/mean of |B| — lower is more uniform.
        """
        mag = self.B_magnitude
        mean = mag.mean()
        if mean == 0:
            return float("inf")
        return float(mag.std() / mean)

    def peak_field(self) -> float:
        """Peak |B| in Tesla."""
        return float(self.B_magnitude.max())

    def mean_field(self) -> float:
        """Mean |B| in Tesla."""
        return float(self.B_magnitude.mean())


# ---------------------------------------------------------------------------
# Coil path factories
# ---------------------------------------------------------------------------

def solenoid_path(
    radius_m: float,
    length_m: float,
    turns: int,
    current: float,
    n_points_per_turn: int = 36,
) -> CoilPath:
    """
    Generate a helical solenoid wire path.

    Args:
        radius_m:           Winding radius in metres.
        length_m:           Axial length in metres.
        turns:              Number of turns.
        current:            Operating current in amperes.
        n_points_per_turn:  Path resolution (points per turn).

    Returns:
        CoilPath ready for Biot-Savart integration.
    """
    n_points = turns * n_points_per_turn + 1
    theta = np.linspace(0, 2 * np.pi * turns, n_points)
    z = np.linspace(-length_m / 2, length_m / 2, n_points)
    x = radius_m * np.cos(theta)
    y = radius_m * np.sin(theta)
    return CoilPath(points=np.column_stack([x, y, z]), current=current)


def helmholtz_path(
    radius_m: float,
    separation_m: float,
    turns: int,
    current: float,
    n_points_per_turn: int = 36,
) -> CoilPath:
    """
    Generate a Helmholtz coil pair wire path (two circular loops, same current).

    Optimal Helmholtz separation = radius (maximises field uniformity at centre).

    Returns a single concatenated CoilPath representing both coils.
    """
    n = turns * n_points_per_turn + 1
    theta = np.linspace(0, 2 * np.pi * turns, n)

    def loop(z_offset: float) -> np.ndarray:
        x = radius_m * np.cos(theta)
        y = radius_m * np.sin(theta)
        z = np.full_like(theta, z_offset)
        return np.column_stack([x, y, z])

    top = loop(separation_m / 2)
    bottom = loop(-separation_m / 2)
    path = np.vstack([top, bottom])
    return CoilPath(points=path, current=current)


def maxwell_coil_path(
    radius_m: float,
    turns: int,
    current: float,
    n_points_per_turn: int = 36,
) -> CoilPath:
    """
    Three-coil Maxwell configuration for highly uniform gradient fields.
    Coils at z = ±R*√(3/7) and z = 0 with turn ratios 49:64:49.
    """
    z1 = radius_m * np.sqrt(3.0 / 7.0)
    n = turns * n_points_per_turn + 1
    theta = np.linspace(0, 2 * np.pi * turns, n)

    def loop(z_offset: float) -> np.ndarray:
        return np.column_stack([
            radius_m * np.cos(theta),
            radius_m * np.sin(theta),
            np.full_like(theta, z_offset),
        ])

    path = np.vstack([loop(z1), loop(0.0), loop(-z1)])
    return CoilPath(points=path, current=current)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

class MagneticAnalyzer:
    """
    Computes magnetic field distributions and coil efficiency metrics.

    Example::

        from cad.magnetic_analyzer import MagneticAnalyzer, solenoid_path

        coil = solenoid_path(radius_m=0.05, length_m=0.1, turns=100, current=2.0)
        analyzer = MagneticAnalyzer()

        # Evaluate on an axial line
        z_line = np.column_stack([
            np.zeros(50), np.zeros(50), np.linspace(-0.06, 0.06, 50)
        ])
        result = analyzer.compute_field(coil, z_line)
        print(f"Peak field: {result.peak_field()*1e3:.2f} mT")
        print(f"Uniformity: {result.uniformity()*100:.1f}%")
    """

    def compute_field(
        self,
        coil: CoilPath,
        eval_points: np.ndarray,
        batch_size: int = 512,
    ) -> FieldResult:
        """
        Biot-Savart integration over the coil wire path.

        Args:
            coil:        CoilPath with wire vertices and current.
            eval_points: (M, 3) field evaluation points in metres.
            batch_size:  Number of eval points processed per batch
                         (reduces peak memory for large grids).

        Returns:
            FieldResult with B vectors at each eval point.
        """
        eval_points = np.asarray(eval_points, dtype=float)
        M = len(eval_points)
        B_total = np.zeros((M, 3))

        # Wire segments: midpoint and dl vector
        pts = coil.points
        dl = np.diff(pts, axis=0)           # (N-1, 3) segment vectors
        mid = (pts[:-1] + pts[1:]) / 2.0   # (N-1, 3) segment midpoints
        prefactor = (MU_0 / (4.0 * np.pi)) * coil.current

        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            r_obs = eval_points[start:end]   # (batch, 3)

            # r = obs_point - midpoint: broadcast (batch, N-1, 3)
            r_vec = r_obs[:, np.newaxis, :] - mid[np.newaxis, :, :]   # (batch, N-1, 3)
            r_norm = np.linalg.norm(r_vec, axis=-1, keepdims=True)    # (batch, N-1, 1)

            # Avoid singularity (should not occur for well-separated wire/obs)
            r_norm = np.where(r_norm < 1e-12, 1e-12, r_norm)

            # cross(dl, r_vec): dl is (N-1,3), r_vec is (batch,N-1,3)
            dl_broad = dl[np.newaxis, :, :]  # (1, N-1, 3)
            cross = np.cross(dl_broad, r_vec)  # (batch, N-1, 3)

            dB = prefactor * cross / (r_norm ** 3)  # (batch, N-1, 3)
            B_total[start:end] = dB.sum(axis=1)

        return FieldResult(points=eval_points, B=B_total, coil_path=coil)

    def flux_through_surface(
        self,
        coil: CoilPath,
        surface_points: np.ndarray,
        normal: np.ndarray,
        area_element: float,
    ) -> float:
        """
        Compute magnetic flux Φ = ∫ B·dA through a flat surface.

        Args:
            coil:          Source CoilPath.
            surface_points: (M, 3) points covering the surface in metres.
            normal:        (3,) unit normal vector to the surface.
            area_element:  Area represented by each point (m²).

        Returns:
            Flux in Weber (Wb).
        """
        result = self.compute_field(coil, surface_points)
        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal)
        flux = np.dot(result.B, normal).sum() * area_element
        return float(flux)

    def axial_field_profile(
        self,
        coil: CoilPath,
        z_start: float,
        z_end: float,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute axial (z-axis) field profile.

        Returns:
            z_coords (m), Bz_values (T)
        """
        z = np.linspace(z_start, z_end, n_points)
        pts = np.column_stack([np.zeros(n_points), np.zeros(n_points), z])
        result = self.compute_field(coil, pts)
        return z, result.B_axial

    def efficiency_metrics(
        self,
        coil: CoilPath,
        target_region_points: np.ndarray,
        wire_resistivity: float = 1.68e-8,  # copper Ω·m
        wire_diameter_m: float = 1e-3,
    ) -> dict:
        """
        Compute coil efficiency metrics for the optimizer.

        Args:
            coil:                 Source CoilPath.
            target_region_points: (M, 3) points in the target field region.
            wire_resistivity:     Conductor resistivity (Ω·m). Default: copper.
            wire_diameter_m:      Wire cross-section diameter in metres.

        Returns:
            Dict with keys:
              mean_field_T, peak_field_T, uniformity,
              wire_length_m, resistance_ohm, power_W,
              field_per_watt, field_per_kg_wire
        """
        result = self.compute_field(coil, target_region_points)
        wire_area = np.pi * (wire_diameter_m / 2) ** 2
        resistance = wire_resistivity * coil.wire_length / wire_area
        power = coil.current ** 2 * resistance

        wire_volume = wire_area * coil.wire_length
        copper_density = 8960.0  # kg/m³
        wire_mass = wire_volume * copper_density

        mean_B = result.mean_field()
        return {
            "mean_field_T": mean_B,
            "peak_field_T": result.peak_field(),
            "uniformity": result.uniformity(),
            "wire_length_m": coil.wire_length,
            "resistance_ohm": resistance,
            "power_W": power,
            "field_per_watt": mean_B / power if power > 0 else 0.0,
            "field_per_kg_wire": mean_B / wire_mass if wire_mass > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    @staticmethod
    def axial_line(z_start: float, z_end: float, n: int = 100) -> np.ndarray:
        """Return evaluation points along the z-axis."""
        z = np.linspace(z_start, z_end, n)
        return np.column_stack([np.zeros(n), np.zeros(n), z])

    @staticmethod
    def midplane_grid(
        r_max: float,
        n_radial: int = 20,
        n_angular: int = 36,
    ) -> np.ndarray:
        """Return a polar grid in the z=0 plane."""
        r = np.linspace(0, r_max, n_radial)
        theta = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)
        rr, tt = np.meshgrid(r, theta)
        x = (rr * np.cos(tt)).ravel()
        y = (rr * np.sin(tt)).ravel()
        z = np.zeros_like(x)
        return np.column_stack([x, y, z])

    @staticmethod
    def volume_grid(
        r_max: float,
        z_half: float,
        n: int = 10,
    ) -> np.ndarray:
        """Return a 3-D cylindrical volume grid."""
        x = np.linspace(-r_max, r_max, n)
        y = np.linspace(-r_max, r_max, n)
        z = np.linspace(-z_half, z_half, n)
        xx, yy, zz = np.meshgrid(x, y, z)
        mask = xx**2 + yy**2 <= r_max**2
        return np.column_stack([xx[mask], yy[mask], zz[mask]])
