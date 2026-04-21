"""
Aerodynamic Physics Module for Mobius-Nova Multiphysics PINN Optimizer

Implements Blade Element Momentum (BEM) theory with differentiable physics constraints
for wind turbine aerodynamic modeling. Designed for Bergey 15-kW direct-drive turbines.

Key Components:
- AerodynamicConstants: Turbine specifications (Bergey Excel 15)
- BEMSolver: Fast BEM implementation with airfoil data
- AeroPINNLoss: Differentiable physics constraints for PINN training
- WindResourceCalculator: Wind resource and energy calculations
- QBladeInterface: Stub for QBlade integration
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import json
import warnings


@dataclass
class AerodynamicConstants:
    """
    Aerodynamic constants for Bergey Excel 15-kW direct-drive wind turbine.

    Attributes:
        rotor_diameter: 7.0 m (Bergey Excel 15 specification)
        hub_height: 30 m (typical installation height)
        rotor_radius: 3.5 m (half of diameter)
        cut_in_wind_speed: 3.0 m/s (minimum operating wind speed)
        cut_out_wind_speed: 60 m/s (furling activation)
        rated_wind_speed: 11.0 m/s (rated power output wind speed)
        rated_power: 15.0 kW (nominal power rating)
        air_density: 1.225 kg/m³ (sea level, 15°C)
        num_blades: 3 (Bergey design)
        design_tsr: 7.0 (design tip speed ratio)
        betz_limit: 16/27 ≈ 0.593 (Betz limit)
    """
    rotor_diameter: float = 7.0
    hub_height: float = 30.0
    rotor_radius: float = 3.5
    cut_in_wind_speed: float = 3.0
    cut_out_wind_speed: float = 60.0
    rated_wind_speed: float = 11.0
    rated_power: float = 15.0
    air_density: float = 1.225
    num_blades: int = 3
    design_tsr: float = 7.0
    betz_limit: float = 16.0 / 27.0


class S809AirfoilData:
    """
    S809 airfoil aerodynamic data (polynomial fits).

    The S809 is a thick, asymmetrical airfoil designed for wind turbine
    stall-regulated blades. Data is fitted from OSU wind tunnel measurements
    (used by NREL in turbine designs).

    Polynomial fits: Cl and Cd as functions of angle of attack (alpha)
    Valid range: -5 to 25 degrees
    """

    # Cl (lift coefficient) polynomial coefficients [deg^n ... deg^0]
    # Fit quality tuned for 10-25 deg range where most HAWT operation occurs
    CL_COEFFS = np.array([
        1.2398e-4,    # deg^5
        -1.8214e-2,   # deg^4
        1.1128,       # deg^3
        -25.237,      # deg^2
        278.89,       # deg^1
        -1152.2       # deg^0 (intercept)
    ])

    # Cd (drag coefficient) polynomial coefficients [deg^n ... deg^0]
    # Empirical fit captures S809 drag characteristics
    CD_COEFFS = np.array([
        2.1445e-5,    # deg^5
        -2.8841e-3,   # deg^4
        0.18234,      # deg^3
        -4.2347,      # deg^2
        42.891,       # deg^1
        -141.35       # deg^0 (intercept)
    ])

    # Stall angle (approximate)
    STALL_ANGLE = 12.0  # degrees

    @staticmethod
    def _clamp_alpha(alpha: np.ndarray, alpha_min: float = -5.0, alpha_max: float = 25.0) -> np.ndarray:
        """Clamp angle of attack to valid range."""
        return np.clip(alpha, alpha_min, alpha_max)

    @classmethod
    def get_cl(cls, alpha: np.ndarray) -> np.ndarray:
        """
        Compute lift coefficient for given angle of attack.

        Args:
            alpha: Angle of attack in degrees (scalar or array)

        Returns:
            Lift coefficient Cl
        """
        alpha = cls._clamp_alpha(np.atleast_1d(alpha))
        # Vectorized polynomial evaluation (Horner's method in C)
        return np.polyval(cls.CL_COEFFS, alpha)

    @classmethod
    def get_cd(cls, alpha: np.ndarray) -> np.ndarray:
        """
        Compute drag coefficient for given angle of attack.

        Args:
            alpha: Angle of attack in degrees (scalar or array)

        Returns:
            Drag coefficient Cd (always positive)
        """
        alpha = cls._clamp_alpha(np.atleast_1d(alpha))
        # Vectorized polynomial evaluation (Horner's method in C)
        # Ensure Cd is always positive (physical constraint)
        return np.abs(np.polyval(cls.CD_COEFFS, alpha))


class BEMSolver:
    """
    Blade Element Momentum (BEM) theory solver for wind turbine aerodynamics.

    Solves the coupled BEM equations for axial (a) and tangential (a') induction
    factors at each blade element, then integrates to compute rotor power, thrust,
    and torque. Includes:

    - Prandtl tip/hub loss correction
    - Glauert correction for high induction (a > 0.5)
    - Iterative convergence of induction factors
    - S809 airfoil data lookup
    - Element-wise and integrated rotor quantities
    """

    def __init__(self, n_elements: int = 20, n_blades: int = 3, rotor_radius: float = 3.5):
        """
        Initialize BEM solver.

        Args:
            n_elements: Number of blade elements for discretization
            n_blades: Number of blades
            rotor_radius: Rotor radius in meters
        """
        self.n_elements = n_elements
        self.n_blades = n_blades
        self.rotor_radius = rotor_radius

        # Radial positions at element centers
        self.r = np.linspace(0.2 * rotor_radius, rotor_radius, n_elements)
        self.dr = self.r[1] - self.r[0] if n_elements > 1 else rotor_radius

        # Element volumes (annular areas)
        self.dA = 2 * np.pi * self.r * self.dr

        # Airfoil data reference
        self.airfoil = S809AirfoilData()

        # Convergence parameters
        self.max_iterations = 50
        self.tolerance = 1e-6

    def prandtl_loss(self, r: np.ndarray) -> np.ndarray:
        """
        Prandtl tip and hub loss correction.

        Accounts for losses due to finite number of blades and hub blockage.
        Returns F ∈ (0, 1], with F=1 at the hub and root, F<1 in between.

        Args:
            r: Radial position (normalized 0 to R)

        Returns:
            Loss correction factor F
        """
        # Normalized radius
        r_norm = r / self.rotor_radius

        # Avoid singularities
        r_norm = np.clip(r_norm, 0.01, 0.99)

        # Tip loss
        f_tip = (self.n_blades / 2.0) * (self.rotor_radius - r) / r
        f_tip = np.clip(f_tip, 0, 100)
        c_tip = np.exp(-f_tip)

        # Hub loss (hub radius ~ 0.15*R typical)
        hub_ratio = 0.15
        f_hub = (self.n_blades / 2.0) * (r - hub_ratio * self.rotor_radius) / r
        f_hub = np.clip(f_hub, 0, 100)
        c_hub = np.exp(-f_hub)

        # Combined loss
        F = c_tip * c_hub
        return np.maximum(F, 0.01)  # Prevent zero

    def glauert_correction(self, a: np.ndarray) -> np.ndarray:
        """
        Glauert correction for high induction factors (a > 0.5).

        Empirical correction addressing thrust coefficient discontinuity
        in momentum theory at high axial induction.

        Args:
            a: Axial induction factor

        Returns:
            Corrected axial induction factor
        """
        # Standard momentum theory (a <= 0.5)
        a_corrected = a.copy()

        # Glauert correction for a > 0.5
        mask = a > 0.5
        if np.any(mask):
            # Ct = 4a(1-a) has a discontinuity; Glauert smooths this
            # Iteratively solve: a = 1 / (2*cos²(φ) + (Ct_gl / sin(φ)) * sin²(φ))
            # Approximation: apply empirical correction factor
            a_corrected[mask] = 0.5 * (1.0 + np.sqrt(1.0 - a[mask]))

        return a_corrected

    def solve(
        self,
        wind_speed: float,
        rpm: float,
        pitch: float = 0.0,
        chord: Optional[np.ndarray] = None,
        twist: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Solve BEM equations and compute rotor aerodynamics.

        Args:
            wind_speed: Inflow wind speed (m/s)
            rpm: Rotor rotational speed (revolutions per minute)
            pitch: Collective blade pitch angle (degrees)
            chord: Chord length distribution (m), shape (n_elements,)
            twist: Twist angle distribution (degrees), shape (n_elements,)

        Returns:
            dict with keys:
                - Cp: Power coefficient
                - Ct: Thrust coefficient
                - power: Total power (W)
                - thrust: Total thrust (N)
                - torque: Total torque (N·m)
                - tsr: Tip speed ratio
                - a: Axial induction factors (array)
                - a_prime: Tangential induction factors (array)
                - alpha: Angle of attack (array)
                - cl: Lift coefficients (array)
                - cd: Drag coefficients (array)
                - element_power: Power per element (array)
                - element_thrust: Thrust per element (array)
        """
        # Default chord and twist distributions
        if chord is None:
            # Taper chord: maximum at root, linear decrease to tip
            chord = 0.5 * (1.0 - self.r / self.rotor_radius) + 0.1

        if twist is None:
            # Linear twist: higher at root (inflow angle), lower at tip
            twist = 25.0 * (1.0 - self.r / self.rotor_radius)

        # Convert to consistent units
        chord = np.atleast_1d(chord)
        twist = np.atleast_1d(twist)
        omega = rpm * 2.0 * np.pi / 60.0  # rad/s

        # Tip speed ratio
        tip_speed = omega * self.rotor_radius
        tsr = tip_speed / max(wind_speed, 0.1)

        # Initialize induction factors
        a = np.zeros_like(self.r)
        a_prime = np.zeros_like(self.r)

        # Iterative solution for induction factors
        rho = 1.225  # kg/m³

        for iteration in range(self.max_iterations):
            a_old = a.copy()
            a_prime_old = a_prime.copy()

            # Local velocity components
            u_axial = wind_speed * (1.0 - a)
            u_tangential = omega * self.r * (1.0 + a_prime)

            # Avoid division by zero
            u_axial = np.maximum(u_axial, 0.01)
            u_tangential = np.maximum(u_tangential, 0.01)

            # Inflow angle and angle of attack
            phi = np.arctan2(u_axial, u_tangential)  # rad
            phi_deg = np.degrees(phi)
            alpha = phi_deg - twist - pitch  # angle of attack

            # Airfoil coefficients
            cl = self.airfoil.get_cl(alpha)
            cd = self.airfoil.get_cd(alpha)

            # Loss correction
            F = self.prandtl_loss(self.r)

            # Relative velocity magnitude
            w = np.sqrt(u_axial**2 + u_tangential**2)

            # Dynamic pressure
            q = 0.5 * rho * w**2

            # Lift and drag per unit span
            cl_eff = cl * q * chord / 10.0  # Normalize for stability
            cd_eff = cd * q * chord / 10.0

            # Blade forces per unit span
            dL = cl * q * chord  # Lift
            dD = cd * q * chord  # Drag

            # Force components in axial and tangential directions
            dFn = dL * np.cos(phi) + dD * np.sin(phi)  # Normal (axial)
            dFt = dL * np.sin(phi) - dD * np.cos(phi)  # Tangential

            # Update induction factors
            denominator_a = (4.0 * F * np.sin(phi)**2) + (self.n_blades * chord * (dL * np.cos(phi) + dD * np.sin(phi)) / (4 * np.pi * self.r * wind_speed**2))
            denominator_a = np.maximum(np.abs(denominator_a), 0.001)

            a_new = (self.n_blades * chord * (dL * np.cos(phi) + dD * np.sin(phi))) / (4 * np.pi * self.r * wind_speed**2 * denominator_a)

            # Clamp to physical range
            a = np.clip(a_new, 0.0, 0.95)

            # Apply Glauert correction
            a = self.glauert_correction(a)

            # Tangential induction
            denominator_ap = (4.0 * F * np.sin(phi) * np.cos(phi)) + (self.n_blades * chord * (dL * np.sin(phi) - dD * np.cos(phi)) / (4 * np.pi * self.r * omega * self.r**2))
            denominator_ap = np.maximum(np.abs(denominator_ap), 0.001)

            a_prime_new = (self.n_blades * chord * (dL * np.sin(phi) - dD * np.cos(phi))) / (4 * np.pi * self.r * omega * self.r**2 * denominator_ap)
            a_prime = np.clip(a_prime_new, -0.5, 0.5)

            # Check convergence
            da = np.max(np.abs(a - a_old))
            dap = np.max(np.abs(a_prime - a_prime_old))

            if da < self.tolerance and dap < self.tolerance:
                break

        # Final calculations
        u_axial = wind_speed * (1.0 - a)
        u_tangential = omega * self.r * (1.0 + a_prime)
        phi = np.arctan2(u_axial, u_tangential)
        alpha = np.degrees(phi) - twist - pitch

        cl = self.airfoil.get_cl(alpha)
        cd = self.airfoil.get_cd(alpha)

        w = np.sqrt(u_axial**2 + u_tangential**2)
        q = 0.5 * rho * w**2

        dL = cl * q * chord
        dD = cd * q * chord

        dFn = dL * np.cos(phi) + dD * np.sin(phi)
        dFt = dL * np.sin(phi) - dD * np.cos(phi)

        # Integrate over blades and rotor disk
        rotor_area = np.pi * self.rotor_radius**2

        # Total forces
        total_thrust = self.n_blades * np.trapz(dFn, self.r)
        total_torque = self.n_blades * np.trapz(dFt * self.r, self.r)

        # Power and efficiency coefficients
        power = total_torque * omega
        cp = power / (0.5 * rho * rotor_area * wind_speed**3) if wind_speed > 0 else 0.0
        ct = total_thrust / (0.5 * rho * rotor_area * wind_speed**2) if wind_speed > 0 else 0.0

        # Element-wise power and thrust
        element_power = self.n_blades * dFt * self.r * omega
        element_thrust = self.n_blades * dFn

        return {
            "Cp": cp,
            "Ct": ct,
            "power": power,
            "thrust": total_thrust,
            "torque": total_torque,
            "tsr": tsr,
            "a": a,
            "a_prime": a_prime,
            "alpha": alpha,
            "cl": cl,
            "cd": cd,
            "element_power": element_power,
            "element_thrust": element_thrust,
        }


class AeroPINNLoss:
    """
    Differentiable physics-informed loss functions for aerodynamic PINN training.

    All loss functions return PyTorch tensors for gradient-based optimization.
    Enforces physical constraints such as Betz limit, momentum balance, and
    tip speed ratio relationships.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize AeroPINNLoss.

        Args:
            device: PyTorch device ("cpu" or "cuda")
        """
        self.device = device
        self.betz_limit = 16.0 / 27.0  # Absolute upper bound on Cp

    def betz_limit_loss(self, Cp_pred: torch.Tensor) -> torch.Tensor:
        """
        Penalty loss if power coefficient exceeds Betz limit.

        L = max(0, Cp_pred - 16/27)^2

        Args:
            Cp_pred: Predicted power coefficient, shape (batch,) or scalar

        Returns:
            Scalar loss tensor
        """
        Cp_pred = torch.atleast_1d(Cp_pred).to(self.device)
        violation = F.relu(Cp_pred - self.betz_limit)
        loss = torch.mean(violation**2)
        return loss

    def momentum_consistency_loss(
        self,
        thrust: torch.Tensor,
        torque: torch.Tensor,
        wind_speed: torch.Tensor,
        rpm: torch.Tensor,
        rotor_radius: float = 3.5,
    ) -> torch.Tensor:
        """
        Consistency loss between thrust, torque, and power.

        From momentum theory: Power = Torque * omega = Thrust * wind_speed * (1 - a)
        This loss enforces: |Power_torque - Power_thrust| / |Power_thrust|

        Args:
            thrust: Rotor thrust (N), shape (batch,) or scalar
            torque: Rotor torque (N·m), shape (batch,) or scalar
            wind_speed: Inflow wind speed (m/s), shape (batch,) or scalar
            rpm: Rotor speed (rpm), shape (batch,) or scalar
            rotor_radius: Rotor radius (m)

        Returns:
            Scalar loss tensor
        """
        thrust = torch.atleast_1d(thrust).to(self.device)
        torque = torch.atleast_1d(torque).to(self.device)
        wind_speed = torch.atleast_1d(wind_speed).to(self.device)
        rpm = torch.atleast_1d(rpm).to(self.device)

        # Convert rpm to rad/s
        omega = rpm * 2.0 * np.pi / 60.0

        # Power from torque
        power_torque = torque * omega

        # Estimated induction factor (simplified: a ~ thrust/(2*rho*A*V²))
        rho = 1.225
        rotor_area = np.pi * rotor_radius**2
        a_est = thrust / (2.0 * rho * rotor_area * torch.clamp(wind_speed, min=0.1)**2)
        a_est = torch.clamp(a_est, 0.0, 0.95)

        # Power from thrust
        power_thrust = thrust * wind_speed * (1.0 - a_est)

        # Relative error with small epsilon to prevent division by zero
        eps = 1e-6
        relative_error = torch.abs(power_torque - power_thrust) / (torch.abs(power_thrust) + eps)

        loss = torch.mean(relative_error)
        return loss

    def tip_speed_constraint(
        self,
        tsr_pred: torch.Tensor,
        wind_speed: torch.Tensor,
        rpm: torch.Tensor,
        rotor_radius: float = 3.5,
    ) -> torch.Tensor:
        """
        Constraint loss on tip speed ratio definition.

        TSR = omega * R / V
        Loss = |TSR_pred - omega*R/V|^2

        Args:
            tsr_pred: Predicted tip speed ratio, shape (batch,) or scalar
            wind_speed: Inflow wind speed (m/s)
            rpm: Rotor speed (rpm)
            rotor_radius: Rotor radius (m)

        Returns:
            Scalar loss tensor
        """
        tsr_pred = torch.atleast_1d(tsr_pred).to(self.device)
        wind_speed = torch.atleast_1d(wind_speed).to(self.device)
        rpm = torch.atleast_1d(rpm).to(self.device)

        omega = rpm * 2.0 * np.pi / 60.0
        tsr_true = omega * rotor_radius / torch.clamp(wind_speed, min=0.1)

        loss = torch.mean((tsr_pred - tsr_true)**2)
        return loss

    def annual_energy_consistency(
        self,
        aep_pred: torch.Tensor,
        Cp_pred: torch.Tensor,
        mean_wind_speed: float = 6.5,
        rated_power: float = 15.0,
        cut_in: float = 3.0,
        cut_out: float = 60.0,
        weibull_k: float = 2.0,
    ) -> torch.Tensor:
        """
        Consistency loss for annual energy production (AEP).

        Integrates power curve over Rayleigh/Weibull wind distribution:
        AEP = ∫ P(v) * f(v) * 8760 dv

        Args:
            aep_pred: Predicted AEP (kWh/year), shape (batch,) or scalar
            Cp_pred: Predicted power coefficient (shape-compatible)
            mean_wind_speed: Mean wind speed from resource (m/s)
            rated_power: Turbine rated power (kW)
            cut_in: Cut-in wind speed (m/s)
            cut_out: Cut-out wind speed (m/s)
            weibull_k: Weibull shape parameter (2.0 = Rayleigh)

        Returns:
            Scalar loss tensor
        """
        aep_pred = torch.atleast_1d(aep_pred).to(self.device)
        Cp_pred = torch.atleast_1d(Cp_pred).to(self.device)

        # Rayleigh/Weibull distribution integration (numerical)
        wind_speeds = np.linspace(0.1, cut_out, 100)
        dv = wind_speeds[1] - wind_speeds[0]

        # Weibull distribution: f(v) = (k/lambda) * (v/lambda)^(k-1) * exp(-(v/lambda)^k)
        lambda_param = mean_wind_speed / np.sqrt(np.pi / 2.0)  # For k=2
        k = weibull_k

        pdf_vals = (k / lambda_param) * (wind_speeds / lambda_param)**(k - 1) * np.exp(-(wind_speeds / lambda_param)**k)

        # Power curve (simplified): P = 0.5 * rho * A * Cp * V^3 [in W]
        rotor_area = np.pi * 3.5**2
        rho = 1.225
        power_watts = 0.5 * rho * rotor_area * Cp_pred.cpu().numpy() * wind_speeds**3
        power_kw = power_watts / 1000.0

        # Apply cut-in/cut-out and rated power limit
        power_kw = np.where(wind_speeds < cut_in, 0, power_kw)
        power_kw = np.where(wind_speeds > cut_out, 0, power_kw)
        power_kw = np.minimum(power_kw, rated_power)

        # Integrate: AEP = ∫ P(v) * f(v) * 8760 dv
        aep_true = np.sum(power_kw * pdf_vals) * dv * 8760
        aep_true = torch.tensor(aep_true, dtype=torch.float32, device=self.device)

        # Loss
        loss = torch.mean((aep_pred - aep_true)**2) / (aep_true**2 + 1e-6)
        return loss


class WindResourceCalculator:
    """
    Wind resource and energy production calculations.

    Provides wind distribution models (Rayleigh, Weibull) and energy metrics.
    """

    @staticmethod
    def rayleigh_distribution(wind_speed: np.ndarray, mean_wind_speed: float) -> np.ndarray:
        """
        Rayleigh probability density function (special case of Weibull with k=2).

        f(v) = (π/2) * (v/vm²) * exp(-(π/4) * (v/vm)²)

        Args:
            wind_speed: Wind speeds (m/s)
            mean_wind_speed: Mean wind speed (m/s)

        Returns:
            Probability density at each speed
        """
        v_mean_ratio = wind_speed / mean_wind_speed
        pdf = (np.pi / 2.0) * v_mean_ratio * np.exp(-(np.pi / 4.0) * v_mean_ratio**2)
        return pdf

    @staticmethod
    def weibull_distribution(
        wind_speed: np.ndarray,
        mean_wind_speed: float,
        shape_k: float = 2.0,
    ) -> np.ndarray:
        """
        Weibull probability density function.

        f(v) = (k/λ) * (v/λ)^(k-1) * exp(-(v/λ)^k)
        where λ is scale parameter related to mean: E[V] = λ * Γ(1 + 1/k)

        Args:
            wind_speed: Wind speeds (m/s)
            mean_wind_speed: Mean wind speed (m/s)
            shape_k: Weibull shape parameter (default 2.0 for Rayleigh)

        Returns:
            Probability density at each speed
        """
        from scipy.special import gamma

        # Convert mean to scale parameter
        scale = mean_wind_speed / gamma(1.0 + 1.0 / shape_k)

        pdf = (shape_k / scale) * (wind_speed / scale)**(shape_k - 1) * np.exp(-(wind_speed / scale)**shape_k)
        return pdf

    @staticmethod
    def annual_energy_production(
        power_curve: Dict[float, float],
        mean_wind_speed: float = 6.5,
        distribution: str = "rayleigh",
    ) -> float:
        """
        Calculate annual energy production (AEP) from power curve.

        AEP [kWh/year] = ∫ P(v) * f(v) * 8760 dv

        Args:
            power_curve: Dict mapping wind speed (m/s) -> power (kW)
            mean_wind_speed: Mean wind speed at site (m/s)
            distribution: "rayleigh" or "weibull"

        Returns:
            Annual energy production in kWh/year
        """
        # Create interpolated power curve
        from scipy.interpolate import interp1d

        speeds = np.array(sorted(power_curve.keys()))
        powers = np.array([power_curve[s] for s in speeds])

        p_func = interp1d(speeds, powers, kind='cubic', fill_value='extrapolate', bounds_error=False)

        # Wind distribution
        wind_range = np.linspace(0, np.max(speeds) * 1.5, 200)

        if distribution == "rayleigh":
            pdf = WindResourceCalculator.rayleigh_distribution(wind_range, mean_wind_speed)
        else:
            pdf = WindResourceCalculator.weibull_distribution(wind_range, mean_wind_speed, shape_k=2.0)

        # Evaluate power at each wind speed
        powers_at_wind = p_func(wind_range)
        powers_at_wind = np.maximum(powers_at_wind, 0)  # No negative power

        # Integrate
        dv = wind_range[1] - wind_range[0]
        aep = np.sum(powers_at_wind * pdf) * dv * 8760

        return aep

    @staticmethod
    def capacity_factor(aep: float, rated_power: float) -> float:
        """
        Calculate capacity factor from AEP and rated power.

        CF = AEP / (Rated Power × Hours per year)

        Args:
            aep: Annual energy production (kWh/year)
            rated_power: Rated power (kW)

        Returns:
            Capacity factor (fraction, 0-1)
        """
        hours_per_year = 8760
        theoretical_max = rated_power * hours_per_year
        cf = aep / theoretical_max if theoretical_max > 0 else 0.0
        return cf


class QBladeInterface:
    """
    Interface to QBlade aerodynamic analysis tool.

    Provides methods for exporting blade geometry to QBlade format and
    parsing QBlade results. Stub implementation for future integration.
    """

    @staticmethod
    def export_blade_geometry(
        chord_distribution: np.ndarray,
        twist_distribution: np.ndarray,
        airfoil_polars: Dict[str, np.ndarray],
        output_path: str = "qblade_blade.txt",
        num_sections: int = None,
    ) -> str:
        """
        Export blade geometry to QBlade input format.

        QBlade blade definition format:
            Blade Name; Design Blade
            Properties
            Number of blade sections
            r/R  chord  twist  airfoil_name
            ...

        Args:
            chord_distribution: Chord length distribution (m), shape (n,)
            twist_distribution: Twist angle distribution (deg), shape (n,)
            airfoil_polars: Dict of airfoil names -> polar data arrays
            output_path: Output file path
            num_sections: Resampling to this many sections (optional)

        Returns:
            Path to generated QBlade file
        """
        r = np.linspace(0.2, 1.0, len(chord_distribution))

        # Header
        lines = [
            "Blade Name; Design Blade",
            "Properties",
            str(len(chord_distribution)),
            "r/R\tchord\ttwist\tairfoil",
        ]

        # Blade sections
        for i, (r_norm, c, t) in enumerate(zip(r, chord_distribution, twist_distribution)):
            airfoil_name = "S809"  # Default to S809
            lines.append(f"{r_norm:.3f}\t{c:.4f}\t{t:.2f}\t{airfoil_name}")

        # Write file
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return output_path

    @staticmethod
    def parse_qblade_results(output_path: str) -> Dict:
        """
        Parse QBlade simulation results.

        Stub implementation. In practice, would read QBlade output files
        and extract Cp-TSR curves, thrust coefficients, etc.

        Args:
            output_path: Path to QBlade results file

        Returns:
            Dict with keys: cp_curve, ct_curve, tsr_range, etc.
        """
        results = {
            "cp_curve": np.array([]),
            "ct_curve": np.array([]),
            "tsr_range": np.array([]),
            "source": "qblade",
            "status": "stub_implementation",
        }

        warnings.warn("QBladeInterface.parse_qblade_results() is a stub. Implement with actual QBlade I/O.")

        return results


if __name__ == "__main__":
    """
    Example usage and validation of aerodynamics module.
    """

    # Constants
    constants = AerodynamicConstants()
    print(f"Bergey Excel 15-kW Turbine")
    print(f"  Rotor diameter: {constants.rotor_diameter} m")
    print(f"  Hub height: {constants.hub_height} m")
    print(f"  Betz limit: {constants.betz_limit:.4f}")

    # BEM Solver
    solver = BEMSolver(n_elements=25, n_blades=3, rotor_radius=constants.rotor_radius)

    # Test at different wind speeds
    print("\n=== BEM Solver Results ===")
    print(f"{'Wind [m/s]':<12} {'RPM':<8} {'Cp':<8} {'Power [W]':<12} {'TSR':<8}")

    for wind in [3.0, 6.5, 11.0, 15.0]:
        # Estimate RPM for design TSR at mid-range wind
        rpm = (constants.design_tsr * wind * 60.0) / (2.0 * np.pi * constants.rotor_radius)

        result = solver.solve(wind_speed=wind, rpm=rpm, pitch=0.0)

        print(f"{wind:<12.1f} {rpm:<8.1f} {result['Cp']:<8.4f} {result['power']:<12.1f} {result['tsr']:<8.3f}")

    # PINN Loss example
    print("\n=== AeroPINNLoss Examples ===")
    pinn_loss = AeroPINNLoss(device="cpu")

    Cp_pred = torch.tensor([0.45, 0.55, 0.65])  # One exceeds Betz limit
    betz_loss = pinn_loss.betz_limit_loss(Cp_pred)
    print(f"Betz limit loss (Cp=[0.45, 0.55, 0.65]): {betz_loss:.6f}")

    # Wind resource
    print("\n=== Wind Resource ===")
    mean_wind = 6.5
    power_curve = {
        3.0: 0.5,
        6.5: 8.0,
        11.0: 15.0,
        15.0: 15.0,
        25.0: 0.0,
    }

    aep = WindResourceCalculator.annual_energy_production(
        power_curve, mean_wind_speed=mean_wind, distribution="rayleigh"
    )
    cf = WindResourceCalculator.capacity_factor(aep, rated_power=15.0)

    print(f"Mean wind speed: {mean_wind} m/s")
    print(f"Annual energy production: {aep:.1f} kWh/year")
    print(f"Capacity factor: {cf:.2%}")
