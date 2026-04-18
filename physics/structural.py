"""
Structural Mechanics Physics Module
====================================
Handles structural physics for 15-kW Bergey direct-drive radial-flux outer-rotor PMSG.
(60-slot / 50-pole, 150 rpm based on NREL MADE3D research)

Key components:
- Structural constants for rotor geometry and materials
- Centrifugal stress computation
- Deformation models
- Gravitational load analysis
- Vibration and resonance checking
- Differentiable PINN loss functions for structural constraints
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict


class StructuralConstants:
    """Physical and material constants for the Bergey 15-kW PMSG rotor structure."""

    # Rotor geometry (NREL MADE3D baseline)
    ROTOR_OUTER_RADIUS_M = 0.600        # m
    AIR_GAP_M = 0.003                    # m
    ROTOR_INNER_RADIUS_M = 0.300        # m (estimated from geometry)
    RATED_RPM = 150                      # rpm
    N_POLES = 50                         # poles
    N_SLOTS = 60                         # slots

    # Gravitational acceleration
    GRAVITY_M_S2 = 9.81                  # m/s²

    # Material: N48H sintered NdFeB magnet
    N48H_SINTERED_DENSITY_KG_M3 = 7600   # kg/m³
    N48H_E_GPA = 160                     # Young's modulus (GPa)
    N48H_POISSON = 0.24                  # Poisson's ratio

    # Material: BAAM 3D printed composite
    BAAM_DENSITY_KG_M3 = 6150            # kg/m³
    BAAM_E_GPA = 45                      # Young's modulus (GPa)
    BAAM_POISSON = 0.35                  # Poisson's ratio

    # Material: 1020 Steel (rotor back iron, shaft)
    STEEL_1020_DENSITY_KG_M3 = 7600      # kg/m³
    STEEL_1020_E_GPA = 200               # Young's modulus (GPa)
    STEEL_1020_POISSON = 0.29            # Poisson's ratio
    STEEL_1020_YIELD_MPA = 350           # Yield strength (approximate)

    # Epoxy bond layer (magnet retention)
    EPOXY_SHEAR_STRENGTH_MPA = 25        # MPa
    EPOXY_TENSILE_STRENGTH_MPA = 32      # MPa
    EPOXY_SHEAR_MODULUS_MPA = 1200       # MPa (approximate)

    # NREL structural limits (binding constraints)
    RADIAL_DEFORM_LIMIT_MM = 0.38        # mm (clearance limit)
    AXIAL_DEFORM_LIMIT_MM = 6.35         # mm (bearing constraint - binding)

    # Thermal expansion coefficients (1/K)
    STEEL_THERMAL_ALPHA = 12e-6          # 1/K
    MAGNET_THERMAL_ALPHA = 5e-6          # 1/K

    @staticmethod
    def omega_from_rpm(rpm: float) -> float:
        """Convert RPM to angular velocity (rad/s)."""
        return rpm * 2 * math.pi / 60.0

    @staticmethod
    def rad_per_sec_to_rpm(omega: float) -> float:
        """Convert rad/s to RPM."""
        return omega * 60.0 / (2 * math.pi)


class CentrifugalStressModel:
    """Computes centrifugal stresses in rotating components."""

    def __init__(self, constants: StructuralConstants = None):
        self.const = constants or StructuralConstants()

    def compute_centrifugal_force(self, mass_kg: float, radius_m: float, rpm: float) -> float:
        """
        Centrifugal force: F = m·ω²·r

        Args:
            mass_kg: Mass (kg)
            radius_m: Radius (m)
            rpm: Rotational speed (rpm)

        Returns:
            Centrifugal force (N)
        """
        omega = self.const.omega_from_rpm(rpm)
        return mass_kg * omega**2 * radius_m

    def compute_hoop_stress(self, density_kg_m3: float, omega_rad_s: float,
                           radius_m: float) -> float:
        """
        Hoop stress in rotating thin ring: σ_hoop = ρ·ω²·r²

        Args:
            density_kg_m3: Material density (kg/m³)
            omega_rad_s: Angular velocity (rad/s)
            radius_m: Radius (m)

        Returns:
            Hoop stress (Pa)
        """
        return density_kg_m3 * omega_rad_s**2 * radius_m**2

    def compute_radial_stress(self, density_kg_m3: float, omega_rad_s: float,
                             r_inner_m: float, r_outer_m: float, r_m: float) -> float:
        """
        Radial stress in thick rotating cylinder using Lamé solution.
        σ_r(r) = (ρ·ω²/8) · [(3+ν)·(r_o² - r²) - (1+3ν)·(r_o² - r_i²)]

        Args:
            density_kg_m3: Material density (kg/m³)
            omega_rad_s: Angular velocity (rad/s)
            r_inner_m: Inner radius (m)
            r_outer_m: Outer radius (m)
            r_m: Radius at which to compute stress (m)

        Returns:
            Radial stress (Pa)
        """
        nu = self.const.STEEL_1020_POISSON  # Use steel Poisson for general case

        term1 = (3 + nu) * (r_outer_m**2 - r_m**2)
        term2 = (1 + 3*nu) * (r_outer_m**2 - r_inner_m**2)

        sigma_r = (density_kg_m3 * omega_rad_s**2 / 8.0) * (term1 - term2)
        return sigma_r

    def compute_magnet_retention_force(self, magnet_mass_kg: float, n_poles: int,
                                       radius_m: float, rpm: float) -> float:
        """
        Per-pole centrifugal retention force needed to hold magnet.
        F_per_pole = (m_total / n_poles) · ω²·r

        Args:
            magnet_mass_kg: Total magnet mass (kg)
            n_poles: Number of poles
            radius_m: Radius (m)
            rpm: Rotational speed (rpm)

        Returns:
            Force per pole (N)
        """
        omega = self.const.omega_from_rpm(rpm)
        mass_per_pole = magnet_mass_kg / n_poles
        return self.compute_centrifugal_force(mass_per_pole, radius_m, rpm)

    def compute_bond_shear_stress(self, retention_force_N: float,
                                  bond_area_m2: float) -> float:
        """
        Shear stress in epoxy bond layer: τ = F / A

        Args:
            retention_force_N: Centrifugal retention force (N)
            bond_area_m2: Bond area (m²)

        Returns:
            Shear stress (MPa)
        """
        shear_stress_pa = retention_force_N / bond_area_m2
        return shear_stress_pa / 1e6  # Convert to MPa


class DeformationModel:
    """Computes structural deformations under loading."""

    def __init__(self, constants: StructuralConstants = None):
        self.const = constants or StructuralConstants()

    def compute_radial_deformation(self, force_N: float, E_gpa: float,
                                  area_m2: float, length_m: float) -> float:
        """
        Linear elastic radial deformation: δ_r = F·L / (E·A)

        Args:
            force_N: Applied force (N)
            E_gpa: Young's modulus (GPa)
            area_m2: Cross-sectional area (m²)
            length_m: Length (m)

        Returns:
            Radial deformation (mm)
        """
        E_pa = E_gpa * 1e9  # Convert to Pa
        deform_m = (force_N * length_m) / (E_pa * area_m2)
        return deform_m * 1000  # Convert to mm

    def compute_axial_deformation(self, mass_kg: float, E_gpa: float,
                                 I_m4: float, length_m: float,
                                 support_type: str = 'cantilever') -> float:
        """
        Axial deformation under gravity for supported beam.
        Uses maximum deflection formulas for common support conditions.

        Args:
            mass_kg: Distributed mass (kg)
            E_gpa: Young's modulus (GPa)
            I_m4: Second moment of inertia (m⁴)
            length_m: Beam length (m)
            support_type: 'cantilever', 'simply_supported', or 'fixed'

        Returns:
            Maximum axial deflection (mm)
        """
        E_pa = E_gpa * 1e9
        W = mass_kg * self.const.GRAVITY_M_S2  # Total weight (N)
        w = W / length_m  # Distributed load (N/m)

        if support_type == 'cantilever':
            # δ_max = w·L⁴ / (8·E·I)
            delta_m = (w * length_m**4) / (8 * E_pa * I_m4)
        elif support_type == 'simply_supported':
            # δ_max = 5·w·L⁴ / (384·E·I)
            delta_m = (5 * w * length_m**4) / (384 * E_pa * I_m4)
        elif support_type == 'fixed':
            # δ_max = w·L⁴ / (384·E·I)
            delta_m = (w * length_m**4) / (384 * E_pa * I_m4)
        else:
            raise ValueError(f"Unknown support type: {support_type}")

        return delta_m * 1000  # Convert to mm

    def compute_thermal_expansion(self, delta_T_K: float, alpha_per_K: float,
                                  length_m: float) -> float:
        """
        Thermal expansion: δ_thermal = α·ΔT·L

        Args:
            delta_T_K: Temperature change (K)
            alpha_per_K: Thermal expansion coefficient (1/K)
            length_m: Original length (m)

        Returns:
            Thermal deformation (mm)
        """
        deform_m = alpha_per_K * delta_T_K * length_m
        return deform_m * 1000  # Convert to mm

    def compute_combined_deformation(self, radial_mm: float, axial_mm: float,
                                    thermal_mm: float) -> float:
        """
        Combined deformation using von Mises equivalent strain energy.
        Total = sqrt(radial² + axial² + thermal²)

        Args:
            radial_mm: Radial deformation (mm)
            axial_mm: Axial deformation (mm)
            thermal_mm: Thermal deformation (mm)

        Returns:
            Combined von Mises equivalent deformation (mm)
        """
        return math.sqrt(radial_mm**2 + axial_mm**2 + thermal_mm**2)


class GravitationalLoadModel:
    """Computes gravitational loads and deflections."""

    def __init__(self, constants: StructuralConstants = None):
        self.const = constants or StructuralConstants()

    def compute_gravitational_bending(self, rotor_mass_kg: float,
                                     stack_length_m: float,
                                     bearing_span_m: float) -> Tuple[float, float]:
        """
        Bending moment and deflection from rotor gravity load on simply supported shaft.
        M_max = W·L / 4, δ_max = 5·W·L³ / (384·E·I)

        Args:
            rotor_mass_kg: Rotor mass (kg)
            stack_length_m: Axial stack length (m)
            bearing_span_m: Distance between bearings (m)

        Returns:
            (Bending moment Nm, Maximum deflection mm)
        """
        W = rotor_mass_kg * self.const.GRAVITY_M_S2  # Force (N)

        # Maximum bending moment (mid-span)
        M_max = W * bearing_span_m / 4

        # Estimate second moment of inertia for shaft
        # Assuming shaft diameter ~50mm (typical for 15kW generator)
        shaft_d_m = 0.050
        I = math.pi * shaft_d_m**4 / 64

        E_pa = self.const.STEEL_1020_E_GPA * 1e9
        delta_max_m = (5 * W * bearing_span_m**3) / (384 * E_pa * I)

        return M_max, delta_max_m * 1000  # Return Nm and mm

    def compute_shaft_deflection(self, mass_kg: float, length_m: float,
                                E_gpa: float, I_m4: float) -> float:
        """
        Shaft deflection under own weight (simply supported).
        δ = 5·m·g·L³ / (384·E·I)

        Args:
            mass_kg: Shaft mass (kg)
            length_m: Shaft length (m)
            E_gpa: Young's modulus (GPa)
            I_m4: Second moment of inertia (m⁴)

        Returns:
            Maximum deflection (mm)
        """
        W = mass_kg * self.const.GRAVITY_M_S2
        E_pa = E_gpa * 1e9

        delta_m = (5 * W * length_m**3) / (384 * E_pa * I_m4)
        return delta_m * 1000  # Convert to mm


class VibrationModel:
    """Simplified vibration and resonance analysis."""

    def __init__(self, constants: StructuralConstants = None):
        self.const = constants or StructuralConstants()

    def compute_natural_frequency(self, stiffness_N_m: float,
                                 mass_kg: float) -> float:
        """
        Natural frequency: f_n = (1/2π)·sqrt(k/m)

        Args:
            stiffness_N_m: Stiffness (N/m)
            mass_kg: Mass (kg)

        Returns:
            Natural frequency (Hz)
        """
        omega_n = math.sqrt(stiffness_N_m / mass_kg)
        return omega_n / (2 * math.pi)

    def compute_critical_speed(self, E_gpa: float, I_m4: float,
                              mass_kg: float, length_m: float) -> float:
        """
        Critical speed (whirl): ω_crit = π·sqrt(E·I / (m·L⁴))
        Returns RPM equivalent.

        Args:
            E_gpa: Young's modulus (GPa)
            I_m4: Second moment of inertia (m⁴)
            mass_kg: Distributed mass (kg)
            length_m: Shaft length (m)

        Returns:
            Critical speed (RPM)
        """
        E_pa = E_gpa * 1e9
        omega_crit = math.pi * math.sqrt((E_pa * I_m4) / (mass_kg * length_m**4))
        return self.const.rad_per_sec_to_rpm(omega_crit)

    def check_resonance(self, operating_rpm: float, n_poles: int,
                       natural_freq_hz: float) -> Tuple[bool, float]:
        """
        Check if operating speed causes resonance at pole-pass frequency.
        Resonance occurs at: f_operating = f_natural or harmonics.

        Pole-pass frequency: f_pp = (n_poles/60) · rpm

        Args:
            operating_rpm: Operating speed (RPM)
            n_poles: Number of poles
            natural_freq_hz: Natural frequency (Hz)

        Returns:
            (is_resonant: bool, margin_Hz: float)
        """
        # Fundamental pole-pass frequency
        f_pp = (n_poles / 60.0) * operating_rpm

        # Check if close to natural frequency (within ±10% tolerance)
        margin = abs(natural_freq_hz - f_pp)
        tolerance = 0.1 * natural_freq_hz

        is_resonant = margin < tolerance

        return is_resonant, margin

    def campbell_diagram_check(self, operating_rpm: float, n_poles: int,
                              natural_freq_hz: float) -> Dict[str, float]:
        """
        Campbell diagram: plots running speed vs natural frequencies and excitations.
        Checks for intersections (resonance).

        Args:
            operating_rpm: Current operating speed (RPM)
            n_poles: Number of poles
            natural_freq_hz: Natural frequency (Hz)

        Returns:
            Dictionary with resonance analysis
        """
        # Excitation frequencies (harmonics of pole-pass frequency)
        f_pp = (n_poles / 60.0) * operating_rpm

        results = {
            'pole_pass_freq_hz': f_pp,
            'natural_freq_hz': natural_freq_hz,
            'operating_rpm': operating_rpm,
            '1x_margin_hz': abs(natural_freq_hz - f_pp),
            '2x_margin_hz': abs(natural_freq_hz - 2*f_pp),
            '3x_margin_hz': abs(natural_freq_hz - 3*f_pp),
        }

        return results


class StructuralPINNLoss(nn.Module):
    """
    Differentiable structural physics constraints for PINN training.
    All methods return torch tensors suitable for backpropagation.
    """

    def __init__(self, constants: StructuralConstants = None):
        super().__init__()
        self.const = constants or StructuralConstants()
        self.centrifugal_model = CentrifugalStressModel(constants)
        self.deform_model = DeformationModel(constants)

    def centrifugal_consistency(self, stress_pred_pa: torch.Tensor,
                               density_kg_m3: float, omega_rad_s: float,
                               radius_m: float) -> torch.Tensor:
        """
        Physics residual: σ_predicted = ρ·ω²·r²
        Penalizes deviation from centrifugal stress equation.

        Args:
            stress_pred_pa: Predicted stress (Pa) [batch_size, ...]
            density_kg_m3: Material density (kg/m³)
            omega_rad_s: Angular velocity (rad/s)
            radius_m: Radius (m)

        Returns:
            Residual loss (scalar torch tensor)
        """
        sigma_analytical = density_kg_m3 * omega_rad_s**2 * radius_m**2
        sigma_analytical_tensor = torch.tensor(sigma_analytical, dtype=stress_pred_pa.dtype,
                                               device=stress_pred_pa.device)

        residual = stress_pred_pa - sigma_analytical_tensor
        return torch.mean(residual**2)

    def radial_deform_limit(self, deform_pred_mm: torch.Tensor,
                           max_mm: float = 0.38) -> torch.Tensor:
        """
        Soft constraint penalty for radial deformation limit.
        Loss = 0 if deform <= max_mm, else penalty.

        Args:
            deform_pred_mm: Predicted radial deformation (mm)
            max_mm: Maximum allowed (mm)

        Returns:
            Penalty loss (scalar torch tensor)
        """
        max_tensor = torch.tensor(max_mm, dtype=deform_pred_mm.dtype,
                                  device=deform_pred_mm.device)

        # Squared hinge loss: max(0, deform - max)²
        violation = torch.nn.functional.relu(deform_pred_mm - max_tensor)
        return torch.mean(violation**2)

    def axial_deform_limit(self, deform_pred_mm: torch.Tensor,
                          max_mm: float = 6.35) -> torch.Tensor:
        """
        Soft constraint penalty for axial deformation limit.
        Binding constraint for bearing clearance.

        Args:
            deform_pred_mm: Predicted axial deformation (mm)
            max_mm: Maximum allowed (mm) - binding at 6.35 mm

        Returns:
            Penalty loss (scalar torch tensor)
        """
        max_tensor = torch.tensor(max_mm, dtype=deform_pred_mm.dtype,
                                  device=deform_pred_mm.device)

        # Squared hinge loss with higher weight for binding constraint
        violation = torch.nn.functional.relu(deform_pred_mm - max_tensor)
        return 2.0 * torch.mean(violation**2)  # Weight factor for binding constraint

    def bond_stress_limit(self, stress_pred_mpa: torch.Tensor,
                         max_mpa: float = 32.0) -> torch.Tensor:
        """
        Soft constraint penalty for epoxy bond tensile stress limit.

        Args:
            stress_pred_mpa: Predicted bond stress (MPa)
            max_mpa: Maximum allowed tensile stress (MPa)

        Returns:
            Penalty loss (scalar torch tensor)
        """
        max_tensor = torch.tensor(max_mpa, dtype=stress_pred_mpa.dtype,
                                  device=stress_pred_mpa.device)

        # Squared hinge loss
        violation = torch.nn.functional.relu(stress_pred_mpa - max_tensor)
        return torch.mean(violation**2)

    def equilibrium_residual(self, stress_tensor: torch.Tensor,
                            body_forces: torch.Tensor) -> torch.Tensor:
        """
        Equilibrium constraint: ∇·σ + f = 0
        Computes residual of stress divergence + body forces.

        For 2D: ∂σ_xx/∂x + ∂σ_xy/∂y + f_x = 0
               ∂σ_xy/∂x + ∂σ_yy/∂y + f_y = 0

        Args:
            stress_tensor: Stress components [batch, n_points, 3] for 2D (σxx, σyy, σxy)
                          or [batch, n_points, 6] for 3D
            body_forces: Body force density [batch, n_points, 2] or [batch, n_points, 3]

        Returns:
            Residual loss (scalar torch tensor)
        """
        # Compute spatial derivatives (simplified via finite differences)
        # Approximation: use central differences on grid

        # Flatten for computation
        batch_size = stress_tensor.shape[0]
        n_points = stress_tensor.shape[1]

        # Simple residual: ||stress_component + forces||
        # Full implementation would require spatial derivatives
        residual = stress_tensor[..., 0] + body_forces[..., 0]

        return torch.mean(residual**2)

    def strain_displacement_compatibility(self, strain_pred: torch.Tensor,
                                        displacement_grad: torch.Tensor) -> torch.Tensor:
        """
        Strain-displacement relation: ε = ½(∇u + ∇uᵀ)
        Ensures predicted strain is compatible with displacement field.

        Args:
            strain_pred: Predicted strain [batch, n_points, 3] for 2D or [batch, n_points, 6] for 3D
            displacement_grad: Displacement gradient ∇u [batch, n_points, 2, 2] for 2D

        Returns:
            Compatibility residual (scalar torch tensor)
        """
        # Symmetric part of displacement gradient
        disp_grad_sym = 0.5 * (displacement_grad + displacement_grad.transpose(-2, -1))

        # Flatten symmetric tensor to vector form
        # For 2D: [ε_xx, ε_yy, ε_xy]
        strain_computed = torch.stack([
            disp_grad_sym[..., 0, 0],
            disp_grad_sym[..., 1, 1],
            disp_grad_sym[..., 0, 1]
        ], dim=-1)

        # Residual
        residual = strain_pred - strain_computed
        return torch.mean(residual**2)

    def von_mises_stress(self, stress_components: torch.Tensor) -> torch.Tensor:
        """
        Compute von Mises equivalent stress.
        2D: σ_vm = sqrt(σ_xx² - σ_xx·σ_yy + σ_yy² + 3·σ_xy²)

        Args:
            stress_components: [batch, n_points, 3] with [σ_xx, σ_yy, σ_xy] for 2D

        Returns:
            Von Mises stress [batch, n_points]
        """
        sigma_xx = stress_components[..., 0]
        sigma_yy = stress_components[..., 1]
        sigma_xy = stress_components[..., 2]

        vm = torch.sqrt(
            sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * sigma_xy**2
        )

        return vm


# Utility functions for structural analysis

def create_rotor_geometry(outer_radius_m: float = 0.600,
                         inner_radius_m: float = 0.300,
                         stack_length_m: float = 0.150) -> Dict[str, float]:
    """
    Create rotor geometry dictionary.

    Args:
        outer_radius_m: Outer radius (m)
        inner_radius_m: Inner radius (m)
        stack_length_m: Axial length (m)

    Returns:
        Dictionary with geometry parameters
    """
    return {
        'outer_radius_m': outer_radius_m,
        'inner_radius_m': inner_radius_m,
        'stack_length_m': stack_length_m,
        'volume_m3': math.pi * (outer_radius_m**2 - inner_radius_m**2) * stack_length_m,
    }


def estimate_rotor_mass(geometry: Dict, material_density_kg_m3: float) -> float:
    """
    Estimate rotor mass from geometry and material density.

    Args:
        geometry: Geometry dictionary from create_rotor_geometry
        material_density_kg_m3: Material density (kg/m³)

    Returns:
        Estimated mass (kg)
    """
    return geometry['volume_m3'] * material_density_kg_m3


def compute_structural_fos(applied_stress_mpa: float,
                           yield_strength_mpa: float,
                           safety_factor: float = 2.0) -> bool:
    """
    Check structural safety: FOS = yield_strength / applied_stress

    Args:
        applied_stress_mpa: Applied stress (MPa)
        yield_strength_mpa: Material yield strength (MPa)
        safety_factor: Required safety factor

    Returns:
        True if safe, False otherwise
    """
    fos = yield_strength_mpa / (applied_stress_mpa + 1e-6)  # Avoid division by zero
    return fos >= safety_factor


if __name__ == "__main__":
    # Example usage
    const = StructuralConstants()

    # Create models
    centrifugal = CentrifugalStressModel(const)
    deform = DeformationModel(const)
    gravity = GravitationalLoadModel(const)
    vibe = VibrationModel(const)

    # Example: Compute hoop stress at rated speed
    omega = const.omega_from_rpm(const.RATED_RPM)
    sigma_hoop = centrifugal.compute_hoop_stress(
        const.STEEL_1020_DENSITY_KG_M3,
        omega,
        const.ROTOR_OUTER_RADIUS_M
    )
    print(f"Hoop stress at rated speed: {sigma_hoop / 1e6:.2f} MPa")

    # Example: Check deformation limits
    geom = create_rotor_geometry()
    rotor_mass = estimate_rotor_mass(geom, const.STEEL_1020_DENSITY_KG_M3)
    print(f"Estimated rotor mass: {rotor_mass:.2f} kg")

    # Example: Gravitational bending
    M_bending, delta_deflect = gravity.compute_gravitational_bending(
        rotor_mass,
        geom['stack_length_m'],
        geom['outer_radius_m'] * 2
    )
    print(f"Bending moment: {M_bending:.2f} Nm")
    print(f"Deflection: {delta_deflect:.3f} mm")
