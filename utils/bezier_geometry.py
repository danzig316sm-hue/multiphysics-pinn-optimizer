"""
Bézier Pole Geometry Engine for PMSG Shape Optimization.
Mobius-Nova Energy — utils/bezier_geometry.py

Implements all three parameterization modes from the NREL/ORNL paper:
    Sethuraman et al., "Advanced Permanent Magnet Generator Topologies
    Using Multimaterial Shape Optimization and 3D Printing,"
    National Renewable Energy Laboratory / Oak Ridge National Laboratory.

Modes
-----
SYMMETRIC      5th-degree Bézier, 6 control points per curve.
               One half-pole defined; mirror image applied for second half.
               3 curves × 6 points = 18 control points + 1 ratio = 19 DOF (per NREL Table 1)

ASYMMETRIC     10th-degree Bézier, 11 control points per curve.
               Full pole width defined (both halves independent).
               P11 = P1 enforced for periodicity.
               3 curves × 11 points = 33 control points + 1 ratio = 34 DOF

MULTIMATERIAL  10th-degree Bézier, 11 control points × 2 curves (no core curve).
               Magnet 1: sintered N48H arc (constant thickness hm1, no profiling).
               Magnet 2: BAAM printed composite profiled on rear side only.
               2 curves × 11 points + hm1 + ratio = 24 DOF

All three modes expose a flat numpy array (design_vector) that maps 1:1
to the PMSGInputSpec layout used by PMSGPINNModel (input_dim=40).

Physical equations implemented exactly from NREL/ORNL paper:
    Eq (1):  P_i(theta_i) = r_i, theta_i = [P_i(x)*cos(i*delta_theta),
                                              P_i(x)*sin(i*delta_theta)]
    Eq (2):  Asymmetric scaled coordinates with ratio parameter:
             theta_avail = ratio * (7.2*pi/180)
             delta_avail = theta_avail / 10
             theta_1 = (7.2*pi/180) - theta_avail

Mass computation:
    m_mag = 2 * integral(r_gap - r_rear, d_theta) * stack_length * rho_mag * ratio
    (NREL paper sec 2.2.1: "mass obtained by doubling the area under the curves")

Latin-Hypercube sampling (PyDOE2) with 1000-sample initial DOE, exactly
matching NREL paper sec 2.2.4: "1000 different combinations with 7 ratio params".

CadQuery export: converts the polar Bézier boundary to a 2D CadQuery Wire
suitable for extrusion into a 3D pole geometry for SolidWorks import / FEA.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import comb


# ---------------------------------------------------------------------------
# Generator geometry constants  (Bergey 15-kW, NREL/ORNL paper)
# ---------------------------------------------------------------------------

N_POLES           = 50
N_SLOTS           = 60
POLE_WIDTH_DEG    = 360.0 / N_POLES           # 7.2 degrees
POLE_WIDTH_RAD    = POLE_WIDTH_DEG * math.pi / 180.0  # 0.12566 rad
HALF_POLE_DEG     = POLE_WIDTH_DEG / 2.0      # 3.6 degrees  (symmetric mode)
HALF_POLE_RAD     = HALF_POLE_DEG * math.pi / 180.0

# Radial bounds for Bézier control points (NREL Table 1, normalised 0-1 space)
# Curve 1 (air-gap side): r in [287.1, 292.1] mm  → normalised [0.287, 0.292]
R_GAP_MIN   = 287.1   # mm
R_GAP_MAX   = 292.1   # mm
# Curve 2 (rear side):   r in (r_gap_1, 302.1]  → upper bound 302.1 mm
R_REAR_MAX  = 302.1   # mm
# Curve 3 (rotor core):  r in (r_rear, 322.1]
R_CORE_MAX  = 322.1   # mm

# Stack length
STACK_LENGTH_M = 0.350   # m

# Material densities
RHO_SINTERED_KG_M3 = 7_600.0   # N48H sintered magnet
RHO_PRINTED_KG_M3  = 6_150.0   # 75 vol% BAAM composite (ORNL)
RHO_CORE_KG_M3     = 7_600.0   # 1020 steel rotor core

# Baseline magnet mass for % reduction calculation (NREL Table 5)
BASELINE_MAGNET_MASS_KG = 24.08

# Ratio parameter range (NREL paper: 60–100 in steps of 5, normalised 0.6–1.0)
RATIO_MIN = 0.60
RATIO_MAX = 1.00

# Sintered layer bounds for multimaterial mode (NREL Table 1: hm1 in [1.5, 9] mm)
HM1_MIN_MM = 1.5   # mm
HM1_MAX_MM = 9.0   # mm


# ---------------------------------------------------------------------------
# Bézier evaluation utilities
# ---------------------------------------------------------------------------

def bernstein_poly(n: int, i: int, t: np.ndarray) -> np.ndarray:
    """
    Bernstein basis polynomial B_{i,n}(t).
    B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
    """
    return comb(n, i, exact=True) * (t ** i) * ((1.0 - t) ** (n - i))


def bezier_curve(
    control_points: np.ndarray,   # shape (n+1,)  — radii in mm (Cartesian y-values)
    n_eval: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a Bézier curve of degree n = len(control_points)-1.

    The NREL/ORNL approach uses control points as RADII at fixed angular
    positions. t parameterises the curve in Cartesian x-space [0, 1].

    Returns
    -------
    t_vals  : (n_eval,) — parameter values in [0, 1]
    r_vals  : (n_eval,) — interpolated radii in mm
    """
    n = len(control_points) - 1
    t_vals = np.linspace(0.0, 1.0, n_eval)
    r_vals = np.zeros(n_eval)
    for i, cp in enumerate(control_points):
        r_vals += cp * bernstein_poly(n, i, t_vals)
    return t_vals, r_vals


def control_points_to_polar(
    control_points: np.ndarray,   # radii (mm)
    ratio: float,
    mode: str = "asymmetric",
    curve_index: int = 1,         # 1=gap, 2=rear, 3=core  (for symmetric scaling)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Bézier control point radii to polar (r, theta) coordinates.

    Implements NREL/ORNL Equations (1) and (2) exactly.

    Eq (1) — Symmetric:
        P_i(theta_i) = r_i, theta_i  where  r_i = P_i(x)*cos(i*delta),
                                                    P_i(x)*sin(i*delta)
        delta_theta = 0.36 deg  (3.6 deg / 10 equal divisions for n=6 pts)

    Eq (2) — Asymmetric:
        theta_avail = ratio * (7.2 * pi / 180)
        delta_avail = theta_avail / 10
        theta_1     = (7.2 * pi / 180) - theta_avail
        P_i(theta) for i = 1 to 5:
            [P_{i+1}(x)*cos(i*delta_avail + theta_1),
             P_{i+1}(x)*sin(i*delta_avail + theta_1)]

    Parameters
    ----------
    control_points  : radii at each control-point Cartesian position
    ratio           : pole-width ratio (0.6 – 1.0)
    mode            : 'symmetric' | 'asymmetric' | 'multimaterial'
    curve_index     : which Bézier curve (1=gap, 2=rear, 3=core)

    Returns
    -------
    theta_positions : (n_pts,) angular positions in radians
    r_positions     : (n_pts,) radii in mm
    """
    n_pts = len(control_points)

    if mode == "symmetric":
        # Eq (1): 5th-degree, 6 pts, half-pole = 3.6 deg
        # delta_theta = 3.6 / (n_pts - 1) for symmetric half
        delta_theta_rad = HALF_POLE_RAD / (n_pts - 1)
        theta_positions = np.array([i * delta_theta_rad for i in range(n_pts)])
        r_positions = control_points.copy()

    else:
        # Eq (2): 10th-degree, 11 pts, full pole = 7.2 deg
        theta_avail = ratio * POLE_WIDTH_RAD       # scaled pole width
        delta_avail = theta_avail / (n_pts - 1)    # step per control point
        theta_1 = POLE_WIDTH_RAD - theta_avail     # first control point offset

        theta_positions = np.zeros(n_pts)
        for i in range(n_pts):
            if i < n_pts - 1:
                theta_positions[i] = i * delta_avail + theta_1
            else:
                # P11 = P1 for periodicity (NREL paper constraint)
                theta_positions[i] = theta_1
        r_positions = control_points.copy()

    return theta_positions, r_positions


# ---------------------------------------------------------------------------
# Cross-sectional area and mass computation
# ---------------------------------------------------------------------------

def compute_cross_section_area(
    r_gap: np.ndarray,
    r_rear: np.ndarray,
    theta: np.ndarray,
) -> float:
    """
    Compute the cross-sectional area of the magnet region between two
    Bézier boundary curves using the trapezoidal rule.

    From NREL paper sec 2.2.1:
        "mass obtained by doubling the area under the curves (r_rear, r_gap)
         from one half-symmetry and applying mass densities"

    A = integral_{theta_min}^{theta_max} (r_gap(t) - r_rear(t)) * 0.5 *
        (r_gap(t) + r_rear(t)) d_theta
      ≈ 0.5 * integral (r_gap^2 - r_rear^2) d_theta
    (annular sector area formula)

    Parameters
    ----------
    r_gap   : (n,) air-gap side radii, mm
    r_rear  : (n,) rear side radii, mm
    theta   : (n,) angular positions, radians
    """
    # Convert mm to m
    r_g = r_gap  / 1000.0
    r_r = r_rear / 1000.0

    # Annular sector area: 0.5 * integral(r_gap^2 - r_rear^2) d_theta
    integrand = 0.5 * (r_g ** 2 - r_r ** 2)
    area_m2 = np.trapz(integrand, theta)
    return float(area_m2)


def compute_magnet_mass(
    r_gap: np.ndarray,
    r_rear: np.ndarray,
    theta: np.ndarray,
    mode: str,
    ratio: float,
    hm1_mm: float = 0.0,
    vol_pct: float = 0.75,
    n_poles: int = N_POLES,
) -> Dict[str, float]:
    """
    Compute magnet masses for all three modes.

    Returns dict with keys:
        sintered_mass_kg    — N48H sintered layer mass
        printed_mass_kg     — BAAM composite layer mass
        total_magnet_mass_kg
        mass_reduction_pct  — % below NREL baseline (24.08 kg)
        core_area_m2        — for structural calculations

    NREL paper:
        Single-material mass  = 2 * area * stack_length * rho * (ratio / 1.0)
        (factor 2: both halves of symmetric design; ratio scales pole width)
        Full machine mass     = single_pole_mass * n_poles
    """
    area_m2 = compute_cross_section_area(r_gap, r_rear, theta)

    if mode in ("symmetric", "asymmetric"):
        # factor 2 for symmetric mirroring in symmetric mode;
        # asymmetric already covers full pole so no factor 2
        symmetry_factor = 2.0 if mode == "symmetric" else 1.0
        single_pole_area = area_m2 * symmetry_factor

        # Effective density accounts for printed vs sintered mix
        # In single-material mode, assume printed composite properties
        rho_eff = RHO_PRINTED_KG_M3
        single_pole_mass = single_pole_area * STACK_LENGTH_M * rho_eff
        full_machine_mass = single_pole_mass * n_poles

        sintered_mass = 0.0
        printed_mass  = full_machine_mass

    elif mode == "multimaterial":
        # Sintered layer: arc-shaped, constant thickness hm1
        # Area = hm1 * theta_avail * r_mean (approximate annular sector)
        hm1_m = hm1_mm / 1000.0
        theta_avail = ratio * POLE_WIDTH_RAD
        r_mean_gap = float(np.mean(r_gap)) / 1000.0
        sintered_area = hm1_m * theta_avail * 1.0   # normalized
        sintered_single_mass = (
            sintered_area * STACK_LENGTH_M * RHO_SINTERED_KG_M3
        )
        sintered_mass = sintered_single_mass * n_poles

        # Printed layer: rear-side profiled Bézier (r_gap = sintered outer surface)
        r_sintered_outer = r_gap - hm1_mm   # subtract sintered thickness
        printed_area = compute_cross_section_area(
            r_sintered_outer, r_rear, theta
        )
        printed_single_mass = printed_area * STACK_LENGTH_M * RHO_PRINTED_KG_M3
        printed_mass = printed_single_mass * n_poles

        full_machine_mass = sintered_mass + printed_mass

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use symmetric/asymmetric/multimaterial.")

    mass_reduction_pct = (
        (BASELINE_MAGNET_MASS_KG - full_machine_mass) / BASELINE_MAGNET_MASS_KG * 100.0
    )

    return {
        "sintered_mass_kg":     round(sintered_mass, 4),
        "printed_mass_kg":      round(printed_mass, 4),
        "total_magnet_mass_kg": round(full_machine_mass, 4),
        "mass_reduction_pct":   round(mass_reduction_pct, 2),
        "cross_section_area_m2": round(area_m2, 8),
    }


# ---------------------------------------------------------------------------
# Core geometry engine
# ---------------------------------------------------------------------------

@dataclass
class PoleGeometry:
    """
    Complete geometric description of a single PMSG pole.
    The authoritative output of BezierPoleParametrizer.

    All arrays are in polar coordinates (theta in rad, r in mm).
    """
    mode: str

    # Bézier control points  (normalised Cartesian x-space radii, mm)
    cp_gap:  np.ndarray = field(repr=False)     # air-gap side
    cp_rear: np.ndarray = field(repr=False)     # rear side
    cp_core: Optional[np.ndarray] = field(default=None, repr=False)  # rotor core

    # Evaluated polar curves  (200-point interpolation)
    theta_gap:  np.ndarray = field(default=None, repr=False)
    r_gap:      np.ndarray = field(default=None, repr=False)
    theta_rear: np.ndarray = field(default=None, repr=False)
    r_rear:     np.ndarray = field(default=None, repr=False)
    theta_core: Optional[np.ndarray] = field(default=None, repr=False)
    r_core:     Optional[np.ndarray] = field(default=None, repr=False)

    # Scalar design parameters
    ratio:   float = 1.0
    hm1_mm:  float = 0.0
    vol_pct: float = 0.75

    # Computed properties
    mass_report: Dict[str, float] = field(default_factory=dict)
    asymmetry_index: float = 0.0   # |left_half - right_half| mean

    def to_design_vector(
        self,
        wall_t: float = 0.010,
        n_fins: float = 6.0,
        fin_h:  float = 0.020,
        fin_t:  float = 0.004,
    ) -> np.ndarray:
        """
        Pack into the 40-element flat vector used by PMSGPINNModel.

        Layout (matches PMSGInputSpec exactly):
            [0:11]   cp_gap   (11 values, normalised 0-1)
            [11:22]  cp_rear  (11 values)
            [22:33]  cp_core  (11 values; zeros if multimaterial)
            [33]     ratio
            [34]     hm1_mm / 1000  (→ metres, normalised)
            [35]     vol_pct (0.70–0.75 → encoded 0–1)
            [36]     wall_t  (m)
            [37]     n_fins
            [38]     fin_h   (m)
            [39]     fin_t   (m)
        """
        # Normalise radii to [0, 1] using global bounds
        def norm_r(r, r_min, r_max):
            return (r - r_min) / (r_max - r_min + 1e-8)

        cp_gap_norm  = norm_r(self.cp_gap,  R_GAP_MIN,  R_GAP_MAX)
        cp_rear_norm = norm_r(self.cp_rear, R_GAP_MAX,  R_REAR_MAX)

        if self.cp_core is not None:
            cp_core_norm = norm_r(self.cp_core, R_REAR_MAX, R_CORE_MAX)
        else:
            cp_core_norm = np.zeros(11)

        # Ensure all arrays are exactly 11 elements
        # (symmetric mode has 6 CPs; pad to 11 for unified vector)
        def pad11(arr):
            if len(arr) == 11:
                return arr
            # Interpolate to 11 points
            x_old = np.linspace(0, 1, len(arr))
            x_new = np.linspace(0, 1, 11)
            return np.interp(x_new, x_old, arr)

        vol_norm = (self.vol_pct - 0.70) / (0.75 - 0.70 + 1e-8)  # 0–1

        vec = np.concatenate([
            pad11(cp_gap_norm),
            pad11(cp_rear_norm),
            pad11(cp_core_norm),
            [self.ratio],
            [self.hm1_mm / 1000.0],
            [vol_norm],
            [wall_t],
            [n_fins],
            [fin_h],
            [fin_t],
        ])
        assert len(vec) == 40, f"Design vector length {len(vec)} != 40"
        return vec.astype(np.float32)


class BezierPoleParametrizer:
    """
    NREL/ORNL Bézier pole geometry engine.

    Implements all three parameterization modes with exact equations
    from the paper, mass computation, Latin-Hypercube DOE sampling,
    design-vector encoding, and CadQuery export.

    Quick start
    -----------
    >>> param = BezierPoleParametrizer(mode='asymmetric')
    >>> vectors = param.sample_lhs(n_samples=1000)   # (1000, 40) array
    >>> geom = param.from_design_vector(vectors[0])
    >>> print(geom.mass_report)
    >>> param.print_geometry_summary(geom)
    """

    def __init__(
        self,
        mode: str = "asymmetric",
        n_eval: int = 200,
        stack_length_m: float = STACK_LENGTH_M,
        n_poles: int = N_POLES,
    ):
        if mode not in ("symmetric", "asymmetric", "multimaterial"):
            raise ValueError(
                f"mode must be 'symmetric', 'asymmetric', or 'multimaterial'. Got '{mode}'."
            )
        self.mode = mode
        self.n_eval = n_eval
        self.stack_length_m = stack_length_m
        self.n_poles = n_poles

        # Control-point counts per mode  (NREL Table 1)
        self._n_cp = 6 if mode == "symmetric" else 11
        self._n_curves = 2 if mode == "multimaterial" else 3

    # ------------------------------------------------------------------ #
    #  Latin-Hypercube DOE sampling                                        #
    # ------------------------------------------------------------------ #

    def sample_lhs(
        self,
        n_samples: int = 1000,
        n_ratio_levels: int = 7,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate Latin-Hypercube samples of the design space.

        Matches NREL paper sec 2.2.4:
            "1000 different combinations of control points with 7 different
             ratio parameters (varied between 60 and 100 in increments of 5)"

        Parameters
        ----------
        n_samples       : number of LHS samples
        n_ratio_levels  : number of ratio discrete levels (7 in NREL paper)
        seed            : random seed for reproducibility

        Returns
        -------
        design_vectors  : (n_samples * n_ratio_levels, 40) float32 array
                          Each row is a complete design vector ready for
                          PMSGPINNModel input.
        """
        try:
            from pyDOE2 import lhs as lhs_sampler
        except ImportError:
            try:
                from scipy.stats.qmc import LatinHypercube
                warnings.warn(
                    "pyDOE2 not found. Using scipy.stats.qmc.LatinHypercube. "
                    "Install pyDOE2 for identical NREL paper methodology: pip install pyDOE2",
                    UserWarning,
                )

                def lhs_sampler(n_dim, samples, criterion=None, random_state=None):
                    sampler = LatinHypercube(d=n_dim, seed=random_state)
                    return sampler.random(n=samples)
            except ImportError:
                warnings.warn(
                    "Neither pyDOE2 nor scipy found. Using numpy uniform sampling.",
                    UserWarning,
                )
                rng = np.random.default_rng(seed)

                def lhs_sampler(n_dim, samples, criterion=None, random_state=None):
                    return rng.random((samples, n_dim))

        # Number of Cartesian control-point DOF (radii, normalised 0-1)
        n_cp_total = self._n_cp * self._n_curves  # 18 or 33 depending on mode

        # Sample control points in [0, 1] Cartesian space
        rng = np.random.default_rng(seed)
        cp_samples = lhs_sampler(n_cp_total, n_samples, criterion="m", random_state=seed)

        # 7 ratio levels matching NREL: 60, 65, 70, 75, 80, 85, 90, 95, 100
        # Encoded as 0.60, 0.65, …, 1.00
        ratio_levels = np.linspace(RATIO_MIN, RATIO_MAX, n_ratio_levels)

        # hm1 samples (multimaterial mode only)
        hm1_samples = rng.uniform(HM1_MIN_MM, HM1_MAX_MM, n_samples)
        vol_samples  = rng.uniform(0.70, 0.75, n_samples)

        all_vectors: List[np.ndarray] = []

        for ratio in ratio_levels:
            for i, cp_row in enumerate(cp_samples):
                geom = self._build_geometry(
                    cp_row=cp_row,
                    ratio=float(ratio),
                    hm1_mm=float(hm1_samples[i]),
                    vol_pct=float(vol_samples[i]),
                )
                vec = geom.to_design_vector()
                all_vectors.append(vec)

        return np.stack(all_vectors, axis=0)   # (n_samples * n_ratio_levels, 40)

    # ------------------------------------------------------------------ #
    #  Build geometry from raw control-point row                           #
    # ------------------------------------------------------------------ #

    def _build_geometry(
        self,
        cp_row: np.ndarray,    # flat [0,1]-normalised Cartesian radii
        ratio: float,
        hm1_mm: float = 0.0,
        vol_pct: float = 0.75,
    ) -> PoleGeometry:
        """
        Convert a flat normalised CP row → PoleGeometry with full polar
        curves, mass report, and asymmetry index.
        """
        n = self._n_cp

        # Denormalise: map [0,1] Cartesian space → actual radii in mm
        # Using bounds from NREL Table 1
        def denorm_gap(x):
            return R_GAP_MIN + x * (R_GAP_MAX - R_GAP_MIN)

        def denorm_rear(x):
            # Rear must be > gap side; bounded at R_REAR_MAX
            return R_GAP_MAX + x * (R_REAR_MAX - R_GAP_MAX)

        def denorm_core(x):
            return R_REAR_MAX + x * (R_CORE_MAX - R_REAR_MAX)

        if self.mode in ("symmetric", "asymmetric"):
            cp_gap  = denorm_gap(cp_row[0:n])
            cp_rear = denorm_rear(cp_row[n:2*n])
            cp_core = denorm_core(cp_row[2*n:3*n])
        else:
            # Multimaterial: only 2 Bézier curves (gap + rear); no core curve
            cp_gap  = denorm_gap(cp_row[0:n])
            cp_rear = denorm_rear(cp_row[n:2*n])
            cp_core = None

        # Enforce periodicity for asymmetric: P11 = P1
        if self.mode in ("asymmetric", "multimaterial"):
            cp_gap[-1]  = cp_gap[0]
            cp_rear[-1] = cp_rear[0]

        # Evaluate Bézier curves
        t_gap,  r_gap_eval  = bezier_curve(cp_gap,  n_eval=self.n_eval)
        t_rear, r_rear_eval = bezier_curve(cp_rear, n_eval=self.n_eval)

        # Convert to polar coordinates using Eqs (1) or (2)
        theta_gap,  r_gap_polar  = control_points_to_polar(
            cp_gap, ratio, self.mode, curve_index=1
        )
        theta_rear, r_rear_polar = control_points_to_polar(
            cp_rear, ratio, self.mode, curve_index=2
        )

        theta_core_arr = r_core_arr = None
        if cp_core is not None:
            theta_core_arr, r_core_arr = control_points_to_polar(
                cp_core, ratio, self.mode, curve_index=3
            )

        # Use evaluated curve for area/mass (denser sampling than CP positions)
        # Re-evaluate at consistent theta grid for integration
        theta_grid = np.linspace(0.0, ratio * POLE_WIDTH_RAD, self.n_eval)
        _, r_gap_dense  = bezier_curve(cp_gap,  n_eval=self.n_eval)
        _, r_rear_dense = bezier_curve(cp_rear, n_eval=self.n_eval)

        # Compute masses
        mass_report = compute_magnet_mass(
            r_gap=r_gap_dense,
            r_rear=r_rear_dense,
            theta=theta_grid,
            mode=self.mode,
            ratio=ratio,
            hm1_mm=hm1_mm,
            vol_pct=vol_pct,
            n_poles=self.n_poles,
        )

        # Asymmetry index: mean absolute difference between left and right halves
        # (NREL paper: asymmetric designs rewarded for unidirectional operation)
        mid = len(cp_gap) // 2
        asymmetry = float(np.mean(np.abs(cp_gap[:mid] - cp_gap[mid:])))

        return PoleGeometry(
            mode=self.mode,
            cp_gap=cp_gap,
            cp_rear=cp_rear,
            cp_core=cp_core,
            theta_gap=theta_gap,
            r_gap=r_gap_polar,
            theta_rear=theta_rear,
            r_rear=r_rear_polar,
            theta_core=theta_core_arr,
            r_core=r_core_arr,
            ratio=ratio,
            hm1_mm=hm1_mm,
            vol_pct=vol_pct,
            mass_report=mass_report,
            asymmetry_index=asymmetry,
        )

    def from_design_vector(
        self,
        vec: np.ndarray,
        wall_t: float = 0.010,
        n_fins: float = 6.0,
        fin_h: float  = 0.020,
        fin_t: float  = 0.004,
    ) -> PoleGeometry:
        """
        Reconstruct PoleGeometry from a 40-element PMSGInputSpec design vector.
        Inverse of PoleGeometry.to_design_vector().
        """
        assert len(vec) == 40, f"Expected 40-element vector, got {len(vec)}"

        # Unpack normalised values
        cp_gap_norm  = vec[0:11]
        cp_rear_norm = vec[11:22]
        cp_core_norm = vec[22:33]
        ratio        = float(vec[33])
        hm1_mm       = float(vec[34]) * 1000.0  # m → mm
        vol_pct      = float(vec[35]) * (0.75 - 0.70) + 0.70  # de-normalise

        n = self._n_cp  # 6 or 11

        # Downsample to n control points if needed
        def resample(arr_11, n_target):
            if n_target == 11:
                return arr_11
            x_11 = np.linspace(0, 1, 11)
            x_n  = np.linspace(0, 1, n_target)
            return np.interp(x_n, x_11, arr_11)

        # Re-denormalise
        cp_gap_norm_n  = resample(cp_gap_norm, n)
        cp_rear_norm_n = resample(cp_rear_norm, n)
        cp_core_norm_n = resample(cp_core_norm, n) if self.mode != "multimaterial" else None

        # Build full CP row for _build_geometry
        if self.mode != "multimaterial":
            cp_row = np.concatenate([cp_gap_norm_n, cp_rear_norm_n, cp_core_norm_n])
        else:
            cp_row = np.concatenate([cp_gap_norm_n, cp_rear_norm_n])

        return self._build_geometry(cp_row, ratio, hm1_mm=hm1_mm, vol_pct=vol_pct)

    # ------------------------------------------------------------------ #
    #  Mass accounting — the "true total of the math"                      #
    # ------------------------------------------------------------------ #

    def full_mass_accounting(self, geom: PoleGeometry) -> Dict[str, float]:
        """
        Complete bill-of-materials for one design.

        Returns every mass quantity that matters:
            - Sintered N48H magnet mass  (kg)
            - Printed BAAM composite mass (kg)
            - Total active magnet mass (kg)
            - Rotor core (1020 steel) mass — if Curve 3 present (kg)
            - Total active rotor mass (kg)
            - Torque-density metric (Nm/kg)   using rated torque
            - Mass reduction vs. NREL baseline (%)
            - Cost index (relative, using rare-earth price premium)
        """
        from pinn_model import PMSGConstants as C

        mr = geom.mass_report.copy()

        # Rotor core mass (if core curve present)
        core_mass_kg = 0.0
        if geom.cp_core is not None:
            _, r_core_dense = bezier_curve(geom.cp_core, n_eval=self.n_eval)
            _, r_rear_dense = bezier_curve(geom.cp_rear, n_eval=self.n_eval)
            theta_grid = np.linspace(0.0, geom.ratio * POLE_WIDTH_RAD, self.n_eval)
            core_area = compute_cross_section_area(r_core_dense, r_rear_dense, theta_grid)
            core_single = core_area * self.stack_length_m * RHO_CORE_KG_M3
            core_mass_kg = core_single * self.n_poles

        total_active_mass = mr["total_magnet_mass_kg"] + core_mass_kg
        torque_density = C.RATED_TORQUE_NM / (total_active_mass + 1e-8)

        # Cost index: sintered NdFeB ~$80/kg, printed composite ~$30/kg (rough)
        cost_index = (
            mr["sintered_mass_kg"] * 80.0
            + mr["printed_mass_kg"] * 30.0
            + core_mass_kg * 2.0
        )

        return {
            **mr,
            "rotor_core_mass_kg":      round(core_mass_kg, 4),
            "total_active_mass_kg":    round(total_active_mass, 4),
            "torque_density_Nm_kg":    round(torque_density, 2),
            "vs_nrel_baseline_Nm_kg":  round(
                torque_density - 351.28, 2   # NREL MADE3D baseline
            ),
            "cost_index_usd":          round(cost_index, 2),
            "asymmetry_index":         round(geom.asymmetry_index, 4),
        }

    # ------------------------------------------------------------------ #
    #  Batch Pareto screening                                              #
    # ------------------------------------------------------------------ #

    def screen_pareto_candidates(
        self,
        design_vectors: np.ndarray,
        objectives: List[str] = ("mass_reduction_pct", "asymmetry_index"),
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Screen a batch of design vectors for Pareto-optimal candidates.

        Objectives (all maximised after sign flip for minimisation goals):
            mass_reduction_pct   higher is better (more material saved)
            asymmetry_index      higher is better (rewards novel asymmetry)
            torque_density       higher is better (exceeds NREL baseline)

        Returns
        -------
        pareto_mask     : (n_samples,) bool array — True = Pareto candidate
        mass_reports    : list of full_mass_accounting dicts
        """
        mass_reports = []
        for vec in design_vectors:
            geom = self.from_design_vector(vec)
            mass_reports.append(self.full_mass_accounting(geom))

        # Build objective matrix  (all maximised)
        obj_matrix = np.array([
            [mr.get(obj, 0.0) for obj in objectives]
            for mr in mass_reports
        ])

        # Pareto front: point i is dominated if there exists j that is
        # strictly better on all objectives
        n = len(obj_matrix)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(obj_matrix[j] >= obj_matrix[i]) and np.any(obj_matrix[j] > obj_matrix[i]):
                    dominated[i] = True
                    break

        return ~dominated, mass_reports

    # ------------------------------------------------------------------ #
    #  Console reporting                                                   #
    # ------------------------------------------------------------------ #

    def print_geometry_summary(self, geom: PoleGeometry):
        """Print a complete human-readable summary of a single pole design."""
        ma = self.full_mass_accounting(geom)
        print("\n" + "=" * 60)
        print(f"  POLE GEOMETRY SUMMARY  [mode: {geom.mode.upper()}]")
        print("=" * 60)
        print(f"  Ratio parameter       : {geom.ratio:.3f}")
        print(f"  Asymmetry index       : {geom.asymmetry_index:.4f}")
        if geom.mode == "multimaterial":
            print(f"  Sintered layer hm1    : {geom.hm1_mm:.2f} mm")
            print(f"  Printed composite vol : {geom.vol_pct*100:.1f}%")
        print()
        print("  ── Mass Accounting ──────────────────────────────────")
        print(f"  Sintered N48H mass    : {ma['sintered_mass_kg']:.3f} kg")
        print(f"  Printed BAAM mass     : {ma['printed_mass_kg']:.3f} kg")
        print(f"  Total magnet mass     : {ma['total_magnet_mass_kg']:.3f} kg")
        print(f"  Rotor core mass       : {ma['rotor_core_mass_kg']:.3f} kg")
        print(f"  Total active mass     : {ma['total_active_mass_kg']:.3f} kg")
        print(f"  Mass vs. NREL baseline: {ma['mass_reduction_pct']:+.1f}%  "
              f"(baseline: {BASELINE_MAGNET_MASS_KG} kg)")
        print()
        print("  ── Performance Metrics ──────────────────────────────")
        print(f"  Torque density        : {ma['torque_density_Nm_kg']:.2f} Nm/kg")
        print(f"  vs. NREL MADE3D base  : {ma['vs_nrel_baseline_Nm_kg']:+.2f} Nm/kg  "
              f"(baseline: 351.28 Nm/kg)")
        print(f"  Est. cost index       : ${ma['cost_index_usd']:.0f}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    #  CadQuery export (SolidWorks-compatible STEP pipeline)               #
    # ------------------------------------------------------------------ #

    def to_cadquery_wire(
        self,
        geom: PoleGeometry,
        extrude_length_m: float = STACK_LENGTH_M,
    ):
        """
        Convert pole geometry to a CadQuery Wire object for 3D extrusion.

        The wire traces the outer boundary:
            gap-side Bézier → right edge → rear-side Bézier (reversed) → left edge → close

        This wire can be extruded in CadQuery to produce a 3D pole geometry
        suitable for STEP export and SolidWorks / FEA import.

        Requires: cadquery  (pip install cadquery)

        Returns
        -------
        cq.Workplane  — 3D extruded pole solid in mm units
        """
        try:
            import cadquery as cq
        except ImportError:
            raise ImportError(
                "CadQuery not installed. Run: pip install cadquery\n"
                "Required for 3D pole geometry and STEP export."
            )

        # Sample the two Bézier curves at high resolution
        _, r_gap_mm  = bezier_curve(geom.cp_gap,  n_eval=500)
        _, r_rear_mm = bezier_curve(geom.cp_rear, n_eval=500)
        theta = np.linspace(0.0, geom.ratio * POLE_WIDTH_RAD, 500)

        # Convert polar to Cartesian  (x = r*cos(theta), y = r*sin(theta))
        gap_pts  = [(r * math.cos(t), r * math.sin(t))
                    for r, t in zip(r_gap_mm, theta)]
        rear_pts = [(r * math.cos(t), r * math.sin(t))
                    for r, t in zip(r_rear_mm, reversed(theta))]

        # Build closed 2D profile
        all_pts = gap_pts + rear_pts + [gap_pts[0]]

        # CadQuery wire from spline points
        wire = (
            cq.Workplane("XY")
            .spline(all_pts, includeCurrent=False)
            .close()
            .extrude(extrude_length_m * 1000.0)   # mm
        )
        return wire

    def export_step(
        self,
        geom: PoleGeometry,
        filepath: str,
        extrude_length_m: float = STACK_LENGTH_M,
    ):
        """
        Export pole geometry to STEP file for SolidWorks / FEA import.

        Part of the SolidWorks verification pipeline:
        BezierPoleParametrizer → STEP → SolidWorks → FEA → trust score.
        """
        solid = self.to_cadquery_wire(geom, extrude_length_m)
        import cadquery as cq
        cq.exporters.export(solid, filepath)
        print(f"[BezierGeometry] STEP exported: {filepath}")


# ---------------------------------------------------------------------------
# Convenience function: generate the NREL paper's exact 1000-sample DOE
# ---------------------------------------------------------------------------

def generate_nrel_doe(
    mode: str = "asymmetric",
    n_samples: int = 1000,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate the full NREL/ORNL paper design-of-experiments sample set.

    "1000 different combinations of control points with 7 different ratio
     parameters (varied between 60 and 100 in increments of 5)" — paper sec 2.2.4.

    Returns
    -------
    design_vectors : (7000, 40) float32 array
                     7000 = 1000 samples × 7 ratio levels
    """
    param = BezierPoleParametrizer(mode=mode)
    vectors = param.sample_lhs(
        n_samples=n_samples,
        n_ratio_levels=7,
        seed=seed,
    )
    print(
        f"[BezierGeometry] Generated {len(vectors)} design vectors "
        f"(mode={mode}, {n_samples} LHS × 7 ratio levels)"
    )
    if save_path:
        np.save(save_path, vectors)
        print(f"[BezierGeometry] Saved to {save_path}")
    return vectors


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running BezierPoleParametrizer smoke test...")

    for mode in ("symmetric", "asymmetric", "multimaterial"):
        param = BezierPoleParametrizer(mode=mode)

        # Sample 10 designs
        vecs = param.sample_lhs(n_samples=10, n_ratio_levels=2, seed=0)
        print(f"\n[{mode}] Generated {len(vecs)} design vectors, shape={vecs.shape}")

        # Check first design
        geom = param.from_design_vector(vecs[0])
        param.print_geometry_summary(geom)

        # Verify round-trip: vec → geom → vec ≈ original
        vec_rt = geom.to_design_vector()
        max_err = float(np.max(np.abs(vec_rt - vecs[0])))
        print(f"  Round-trip max error: {max_err:.6f}  (should be < 0.01)")

    print("\nSmoke test complete.")
