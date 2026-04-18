"""
Advanced Thermal Physics Module for 15-kW Bergey Direct-Drive Radial-Flux Outer-Rotor PMSG.

Based on NREL MADE3D research. Implements lumped-parameter thermal networks, convection models,
radiation effects, passive intake cooling (critical innovation), and differentiable PINN losses.

Author: Multiphysics PINN Optimizer
Date: 2025
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class ThermalConstants:
    """Material properties and thermal boundaries for Bergey PMSG."""

    # Temperature boundaries (Celsius)
    AMBIENT_TEMP = 20.0
    MAGNET_MAX_TEMP = 60.0  # N48H demagnetization risk per NREL
    WINDING_MAX_TEMP = 180.0  # H-grade insulation

    # Thermal conductivity (W/m·K)
    CONDUCTIVITY = {
        'air': 0.026,
        'copper': 385.0,
        'steel_m15': 25.0,
        'magnet_n48h': 9.0,
        'baam_composite': 3.5,
        'epoxy': 0.2,
    }

    # Specific heat capacity (J/kg·K)
    SPECIFIC_HEAT = {
        'air': 1005.0,
        'copper': 385.0,
        'steel': 490.0,
        'magnet': 420.0,
        'epoxy': 1700.0,
    }

    # Density (kg/m³)
    DENSITY = {
        'air': 1.225,
        'copper': 8960.0,
        'steel': 7750.0,
        'magnet': 7450.0,
        'epoxy': 1200.0,
    }

    # Air properties
    AIR_KINEMATIC_VISCOSITY = 1.5e-5  # m²/s
    AIR_PRANDTL_NUMBER = 0.71

    # Radiation
    STEFAN_BOLTZMANN = 5.67e-8  # W/m²·K⁴


class LumpedParameterThermalNetwork:
    """
    Nodal thermal circuit for Bergey PMSG.

    Nodes:
    - winding: copper windings (primary loss source)
    - stator_core: laminated steel core
    - air_gap: convective medium between rotor and stator
    - magnet_sintered: bulk sintered NdFeB magnets
    - magnet_printed: printed magnets (if present)
    - rotor_core: solid steel rotor back-iron
    - housing: aluminum/steel enclosure
    - ambient: external environment
    """

    # Node indices for matrix operations
    NODES = {
        'winding': 0,
        'stator_core': 1,
        'air_gap': 2,
        'magnet_sintered': 3,
        'magnet_printed': 4,
        'rotor_core': 5,
        'housing': 6,
        'ambient': 7,
    }

    NUM_NODES = len(NODES)

    def __init__(self):
        """Initialize thermal network."""
        self.R_matrix = None
        self.C_vector = None
        self.node_names = list(self.NODES.keys())

    def build_resistance_matrix(self, geometry_dict: Dict) -> np.ndarray:
        """
        Build thermal resistance matrix from geometry.

        Args:
            geometry_dict: {
                'winding_volume': float,
                'winding_surface': float,
                'stator_core_volume': float,
                'air_gap_distance': float (m),
                'air_gap_area': float (m²),
                'rotor_radius': float (m),
                'stack_length': float (m),
                'magnet_volume': float,
                'rotor_core_volume': float,
                'housing_surface': float,
            }

        Returns:
            R_matrix: (NUM_NODES, NUM_NODES) thermal resistance matrix (K/W)
                R[i,j] = thermal resistance from node i to node j
        """
        R = np.zeros((self.NUM_NODES, self.NUM_NODES))

        # Extract geometry
        d_gap = geometry_dict.get('air_gap_distance', 0.005)  # 5mm default
        A_gap = geometry_dict.get('air_gap_area', 0.5)
        V_wind = geometry_dict.get('winding_volume', 0.01)
        A_wind = geometry_dict.get('winding_surface', 0.2)
        A_housing = geometry_dict.get('housing_surface', 1.5)

        # Winding to stator core (conduction + epoxy bond)
        # Series: copper conductance + epoxy layer
        R[0, 1] = 0.003 / (ThermalConstants.CONDUCTIVITY['copper'] * A_wind * 0.8)
        R[1, 0] = R[0, 1]

        # Stator core to air gap (convection + radiation)
        # Convection dominates in gap
        R[1, 2] = 0.002 / (ThermalConstants.CONDUCTIVITY['steel_m15'] * A_gap * 0.5)
        R[2, 1] = R[1, 2]

        # Air gap itself (conduction across gap)
        R_gap_cond = d_gap / (ThermalConstants.CONDUCTIVITY['air'] * A_gap)
        R[2, 2] = 0.0  # Air gap node internal resistance

        # Air gap to rotor (convection)
        R[2, 5] = R_gap_cond / 2  # Symmetric
        R[5, 2] = R[2, 5]

        # Rotor core to magnet (conduction)
        magnet_volume = geometry_dict.get('magnet_volume', 0.002)
        magnet_thickness = 0.002
        magnet_area = magnet_volume / magnet_thickness if magnet_thickness > 0 else A_gap * 0.5
        R[5, 3] = magnet_thickness / (ThermalConstants.CONDUCTIVITY['magnet_n48h'] * magnet_area)
        R[3, 5] = R[5, 3]

        # Magnet sintered to magnet printed (parallel if present)
        R[3, 4] = 0.001 / (ThermalConstants.CONDUCTIVITY['epoxy'] * A_gap * 0.3)
        R[4, 3] = R[3, 4]

        # Rotor to housing (conduction + convection)
        R[5, 6] = 0.01 / (ThermalConstants.CONDUCTIVITY['steel_m15'] * A_gap * 0.5)
        R[6, 5] = R[5, 6]

        # Winding to housing (convective path through epoxy/air)
        R[0, 6] = 0.015 / (ThermalConstants.CONDUCTIVITY['epoxy'] * A_wind * 0.6)
        R[6, 0] = R[0, 6]

        # Housing to ambient (convection + radiation via external h)
        R[6, 7] = 0.1 / A_housing  # Dominated by external convection coefficient
        R[7, 6] = R[6, 7]

        # Magnet nodes to ambient (thermal path via housing)
        R[3, 7] = R[3, 5] + R[5, 6] + R[6, 7]
        R[7, 3] = R[3, 7]

        R[4, 7] = R[4, 3] + R[3, 7]
        R[7, 4] = R[4, 7]

        self.R_matrix = R
        return R

    def build_capacitance_vector(self, mass_dict: Dict) -> np.ndarray:
        """
        Build thermal capacitance vector from material masses.

        Args:
            mass_dict: {
                'winding': float (kg),
                'stator_core': float,
                'air_gap': float,
                'magnet_sintered': float,
                'magnet_printed': float,
                'rotor_core': float,
                'housing': float,
            }

        Returns:
            C_vector: (NUM_NODES,) thermal capacitance (J/K)
        """
        C = np.zeros(self.NUM_NODES)

        # Map node names to materials for specific heat lookup
        material_map = {
            'winding': 'copper',
            'stator_core': 'steel',
            'air_gap': 'air',
            'magnet_sintered': 'magnet',
            'magnet_printed': 'epoxy',
            'rotor_core': 'steel',
            'housing': 'steel',
            'ambient': 'air',
        }

        for node_idx, node_name in enumerate(self.node_names):
            if node_name in mass_dict and node_name != 'ambient':
                mass = mass_dict[node_name]
                material = material_map.get(node_name, 'steel')
                c_p = ThermalConstants.SPECIFIC_HEAT.get(material, 500.0)
                C[node_idx] = mass * c_p

        # Ambient is infinite heat sink
        C[self.NODES['ambient']] = 1e6

        self.C_vector = C
        return C

    def compute_steady_state_temperatures(
        self,
        heat_sources: Dict[str, float],
        R_matrix: Optional[np.ndarray] = None,
        T_ambient: float = ThermalConstants.AMBIENT_TEMP,
    ) -> np.ndarray:
        """
        Solve steady-state nodal temperatures.

        Args:
            heat_sources: {
                'copper_loss': float (W),
                'iron_loss': float (W),
                'eddy_loss': float (W),
                'cooling_power': float (W, negative),
            }
            R_matrix: Thermal resistance matrix. If None, uses self.R_matrix.
            T_ambient: Ambient temperature (°C)

        Returns:
            T_vector: (NUM_NODES,) temperatures (°C)
        """
        if R_matrix is None:
            R_matrix = self.R_matrix
            if R_matrix is None:
                raise ValueError("Must call build_resistance_matrix() first")

        # Build heat source vector
        Q = np.zeros(self.NUM_NODES)
        Q[self.NODES['winding']] = heat_sources.get('copper_loss', 0.0)
        Q[self.NODES['stator_core']] = heat_sources.get('iron_loss', 0.0)
        Q[self.NODES['magnet_sintered']] = heat_sources.get('eddy_loss', 0.0)

        # Cooling (negative heat source)
        cooling = heat_sources.get('cooling_power', 0.0)
        Q[self.NODES['air_gap']] -= cooling * 0.5
        Q[self.NODES['rotor_core']] -= cooling * 0.5

        # Ambient is clamped to T_ambient (boundary condition)
        # Build condensed system excluding ambient node
        n = self.NUM_NODES - 1
        R_c = R_matrix[:n, :n]
        T_amb_vec = R_matrix[:n, -1] * T_ambient

        try:
            T_condensed = np.linalg.solve(R_c, Q[:n] + T_amb_vec)
            T = np.zeros(self.NUM_NODES)
            T[:n] = T_condensed
            T[-1] = T_ambient
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            R_pinv = np.linalg.pinv(R_c)
            T_condensed = R_pinv @ (Q[:n] + T_amb_vec)
            T = np.zeros(self.NUM_NODES)
            T[:n] = T_condensed
            T[-1] = T_ambient

        return T

    def compute_transient_response(
        self,
        heat_sources: Dict[str, float],
        R_matrix: Optional[np.ndarray] = None,
        C_vector: Optional[np.ndarray] = None,
        dt: float = 1.0,
        n_steps: int = 100,
        T_init: Optional[np.ndarray] = None,
        T_ambient: float = ThermalConstants.AMBIENT_TEMP,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute transient thermal response using forward Euler integration.

        Args:
            heat_sources: Dict of heat sources (constant during transient)
            R_matrix: Thermal resistance matrix
            C_vector: Thermal capacitance vector
            dt: Time step (s)
            n_steps: Number of steps
            T_init: Initial temperature vector. If None, computes steady-state.
            T_ambient: Ambient temperature

        Returns:
            (t_array, T_history): Time steps and (n_steps, NUM_NODES) temperature history
        """
        if R_matrix is None:
            R_matrix = self.R_matrix
        if C_vector is None:
            C_vector = self.C_vector

        if R_matrix is None or C_vector is None:
            raise ValueError("Must initialize resistance and capacitance matrices first")

        # Initial condition
        if T_init is None:
            T_init = self.compute_steady_state_temperatures(heat_sources, R_matrix, T_ambient)

        T_history = np.zeros((n_steps, self.NUM_NODES))
        T_current = T_init.copy()
        T_history[0] = T_current

        # Build heat source vector
        Q = np.zeros(self.NUM_NODES)
        Q[self.NODES['winding']] = heat_sources.get('copper_loss', 0.0)
        Q[self.NODES['stator_core']] = heat_sources.get('iron_loss', 0.0)
        Q[self.NODES['magnet_sintered']] = heat_sources.get('eddy_loss', 0.0)
        cooling = heat_sources.get('cooling_power', 0.0)
        Q[self.NODES['air_gap']] -= cooling * 0.5
        Q[self.NODES['rotor_core']] -= cooling * 0.5

        # Conductance matrix (inverse of resistance, with regularization)
        G = np.zeros_like(R_matrix)
        for i in range(self.NUM_NODES):
            for j in range(self.NUM_NODES):
                if i != j and abs(R_matrix[i, j]) > 1e-8:
                    G[i, j] = 1.0 / R_matrix[i, j]

        # Forward Euler integration
        for step in range(1, n_steps):
            dT_dt = np.zeros(self.NUM_NODES)

            for i in range(self.NUM_NODES - 1):  # Exclude ambient (fixed)
                # Heat in from sources and conduction
                Q_net = Q[i]
                for j in range(self.NUM_NODES):
                    if i != j:
                        Q_net += G[i, j] * (T_current[j] - T_current[i])

                # Temperature rate
                if C_vector[i] > 0:
                    dT_dt[i] = Q_net / C_vector[i]

            T_current = T_current + dT_dt * dt
            T_current[-1] = T_ambient  # Enforce ambient boundary
            T_history[step] = T_current

        t_array = np.arange(n_steps) * dt
        return t_array, T_history


class ConvectionModel:
    """Convection heat transfer correlations for Bergey PMSG."""

    @staticmethod
    def compute_airgap_convection(
        rpm: float,
        air_gap: float,
        rotor_radius: float,
        stack_length: float,
    ) -> float:
        """
        Compute air gap convection coefficient via Taylor-Couette flow.

        Args:
            rpm: Rotor speed (rev/min)
            air_gap: Gap distance (m)
            rotor_radius: Outer rotor radius (m)
            stack_length: Axial stack length (m)

        Returns:
            h_gap: Convection coefficient (W/m²·K)

        Reference: Turbulent Taylor-Couette flow, Nu = 0.128·Ta^0.367
        """
        omega = rpm * 2 * math.pi / 60.0  # rad/s

        # Taylor number
        nu = ThermalConstants.AIR_KINEMATIC_VISCOSITY
        Ta = omega * rotor_radius * air_gap / nu

        # Nusselt number for turbulent gap flow
        if Ta > 1700:  # Turbulent
            Nu = 0.128 * (Ta ** 0.367)
        else:  # Laminar (rare for PMSG)
            Nu = 2.0

        # Characteristic length = gap
        h_gap = (Nu * ThermalConstants.CONDUCTIVITY['air']) / air_gap

        return max(h_gap, 1.0)  # Minimum 1 W/m²·K

    @staticmethod
    def compute_external_convection(
        wind_speed: float,
        characteristic_length: float,
    ) -> float:
        """
        Compute external convection via Churchill-Bernstein for cylinder in crossflow.

        Args:
            wind_speed: Wind speed (m/s)
            characteristic_length: Diameter or characteristic length (m)

        Returns:
            h_ext: External convection coefficient (W/m²·K)
        """
        if wind_speed < 0.1:
            return 1.0  # Natural convection minimum

        # Properties at ~40°C (average)
        rho = 1.205  # kg/m³
        cp = 1006.0  # J/kg·K
        mu = 1.9e-5  # Pa·s (dynamic viscosity)
        k = 0.027  # W/m·K

        # Reynolds number
        Re = rho * wind_speed * characteristic_length / mu
        Pr = ThermalConstants.AIR_PRANDTL_NUMBER

        # Churchill-Bernstein correlation
        C = 0.3 if Re < 1e5 else 0.2
        m = 0.62 if Re < 1e5 else 0.6

        Nu = (
            0.3 +
            (C * (Re ** m) * (Pr ** (1/3))) /
            ((1 + (0.4 / Pr) ** (2/3)) ** 0.25)
        )

        h_ext = (Nu * k) / characteristic_length

        return max(h_ext, 1.0)

    @staticmethod
    def compute_fin_effectiveness(
        h: float,
        k_fin: float,
        fin_height: float,
        fin_thickness: float,
    ) -> float:
        """
        Compute fin efficiency (effectiveness).

        Args:
            h: Convection coefficient (W/m²·K)
            k_fin: Material thermal conductivity (W/m·K)
            fin_height: Fin height from base (m)
            fin_thickness: Fin thickness (m)

        Returns:
            eta_fin: Fin efficiency (0-1)
        """
        if fin_height < 1e-6:
            return 1.0

        # Fin parameter
        m = math.sqrt(2 * h / (k_fin * fin_thickness))
        L_c = fin_height + fin_thickness / 2
        mL = m * L_c

        # Hyperbolic tangent approximation
        eta_fin = math.tanh(mL) / mL
        return max(min(eta_fin, 1.0), 0.0)

    @staticmethod
    def compute_finned_surface_convection(
        h_base: float,
        n_fins: int,
        fin_height: float,
        fin_thickness: float,
        fin_spacing: float,
        fin_length: float,
        k_fin: float = ThermalConstants.CONDUCTIVITY['steel_m15'],
    ) -> float:
        """
        Compute effective convection coefficient for finned surface.

        Args:
            h_base: Base convection coefficient (W/m²·K)
            n_fins: Number of fins
            fin_height: Fin height (m)
            fin_thickness: Fin thickness (m)
            fin_spacing: Spacing between fins (m)
            fin_length: Axial fin length (m)
            k_fin: Fin material conductivity (W/m·K)

        Returns:
            h_eff: Effective convection coefficient (W/m²·K)
        """
        if n_fins == 0:
            return h_base

        # Surface areas
        A_base = fin_spacing * fin_length * n_fins
        A_fin = 2 * fin_height * fin_length * n_fins  # Two sides

        # Fin efficiency
        eta_fin = ConvectionModel.compute_fin_effectiveness(
            h_base, k_fin, fin_height, fin_thickness
        )

        # Effective surface area
        A_eff = A_base + eta_fin * A_fin

        # Total area (base + spacing)
        A_total = (fin_spacing + fin_thickness) * fin_length * n_fins

        # Effective h
        h_eff = h_base * A_eff / A_total

        return h_eff


class RadiationModel:
    """Thermal radiation heat transfer."""

    @staticmethod
    def compute_radiation_heat_transfer(
        T_surface: float,
        T_ambient: float,
        emissivity: float,
        area: float,
    ) -> float:
        """
        Compute radiative heat transfer via Stefan-Boltzmann law.

        Args:
            T_surface: Surface temperature (°C)
            T_ambient: Ambient temperature (°C)
            emissivity: Surface emissivity (0-1)
            area: Surface area (m²)

        Returns:
            Q_rad: Radiative heat transfer (W)
        """
        T_s_K = T_surface + 273.15
        T_a_K = T_ambient + 273.15

        sigma = ThermalConstants.STEFAN_BOLTZMANN
        Q_rad = emissivity * sigma * area * (T_s_K**4 - T_a_K**4)

        return max(Q_rad, 0.0)

    @staticmethod
    def compute_linearized_radiation_coefficient(
        T_surface: float,
        T_ambient: float,
        emissivity: float,
    ) -> float:
        """
        Compute linearized radiation coefficient for network model.
        h_rad = 4 * epsilon * sigma * T_mean³

        Args:
            T_surface: Surface temperature (°C)
            T_ambient: Ambient temperature (°C)
            emissivity: Surface emissivity

        Returns:
            h_rad: Equivalent convection coefficient (W/m²·K)
        """
        T_s_K = T_surface + 273.15
        T_a_K = T_ambient + 273.15
        T_mean = (T_s_K + T_a_K) / 2.0

        sigma = ThermalConstants.STEFAN_BOLTZMANN
        h_rad = 4.0 * emissivity * sigma * (T_mean ** 3)

        return max(h_rad, 0.1)


class PassiveIntakeCoolingModel:
    """
    Passive intake cooling model - critical innovation from trust_score_engine.

    Combines:
    - Ram pressure recovery (wind-induced)
    - Centrifugal pumping (rotor-induced)
    - Intake geometry effects
    - Cooling effectiveness
    """

    @staticmethod
    def compute_ram_pressure(
        wind_speed: float,
        air_density: float = ThermalConstants.DENSITY['air'],
    ) -> float:
        """
        Compute dynamic pressure from ambient wind.

        Args:
            wind_speed: Wind speed (m/s)
            air_density: Air density (kg/m³)

        Returns:
            q_ram: Ram pressure (Pa)
        """
        q_ram = 0.5 * air_density * (wind_speed ** 2)
        return max(q_ram, 0.0)

    @staticmethod
    def compute_intake_recovery(
        q_ram: float,
        Cp_intake: float = 0.7,
    ) -> float:
        """
        Compute pressure recovery from intake design.

        Args:
            q_ram: Ram pressure (Pa)
            Cp_intake: Pressure recovery coefficient (0-1)

        Returns:
            dP_recovered: Recovered pressure (Pa)
        """
        dP_recovered = Cp_intake * q_ram
        return max(dP_recovered, 0.0)

    @staticmethod
    def compute_centrifugal_pump_head(
        rpm: float,
        r_inner: float,
        r_outer: float,
    ) -> float:
        """
        Compute centrifugal pressure head from rotor rotation.

        Assumes radial flow path through rotor geometry.
        P = (1/2) * rho * omega² * (r_outer² - r_inner²)

        Args:
            rpm: Rotor speed (rev/min)
            r_inner: Inner radius (m)
            r_outer: Outer radius (m)

        Returns:
            dP_centrifugal: Centrifugal pressure head (Pa)
        """
        omega = rpm * 2 * math.pi / 60.0  # rad/s
        rho = ThermalConstants.DENSITY['air']

        dP_centrifugal = (
            0.5 * rho * (omega ** 2) * (r_outer**2 - r_inner**2)
        )

        return max(dP_centrifugal, 0.0)

    @staticmethod
    def compute_total_driving_pressure(
        dP_centrifugal: float,
        dP_ram: float = 0.0,
        dP_losses: float = 0.0,
    ) -> float:
        """
        Compute total driving pressure for intake flow.

        Args:
            dP_centrifugal: Centrifugal pump pressure (Pa)
            dP_ram: Ram pressure recovery (Pa)
            dP_losses: Flow losses (Pa)

        Returns:
            dP_total: Total driving pressure (Pa)
        """
        dP_total = dP_centrifugal + dP_ram - dP_losses
        return max(dP_total, 0.0)

    @staticmethod
    def compute_cooling_mass_flow(
        dP_total: float,
        flow_resistance: float,
    ) -> float:
        """
        Compute cooling air mass flow from driving pressure.

        Args:
            dP_total: Total driving pressure (Pa)
            flow_resistance: Flow path resistance (Pa·s/kg)

        Returns:
            m_dot: Mass flow rate (kg/s)
        """
        if flow_resistance <= 0:
            return 0.0

        m_dot = dP_total / flow_resistance
        return max(m_dot, 0.0)

    @staticmethod
    def compute_convective_cooling(
        m_dot: float,
        cp_air: float = ThermalConstants.SPECIFIC_HEAT['air'],
        T_inlet: float = ThermalConstants.AMBIENT_TEMP,
        T_outlet: float = None,
        delta_T: float = None,
    ) -> float:
        """
        Compute convective cooling power from intake air.

        Args:
            m_dot: Mass flow rate (kg/s)
            cp_air: Specific heat of air (J/kg·K)
            T_inlet: Inlet air temperature (°C)
            T_outlet: Outlet air temperature (°C), or use delta_T
            delta_T: Temperature rise of coolant (K), if outlet not given

        Returns:
            Q_cool: Convective cooling power (W)
        """
        if delta_T is not None:
            dT = delta_T
        elif T_outlet is not None:
            dT = T_outlet - T_inlet
        else:
            dT = 10.0  # Default 10 K rise through cooler

        Q_cool = m_dot * cp_air * dT
        return max(Q_cool, 0.0)


class ThermalPINNLoss:
    """
    Differentiable thermal physics constraints for PINN training.
    All methods return torch tensors for backpropagation.
    """

    @staticmethod
    def energy_conservation(
        copper_loss: torch.Tensor,
        iron_loss: torch.Tensor,
        cooling_power: torch.Tensor,
        radiation_power: torch.Tensor,
    ) -> torch.Tensor:
        """
        Energy conservation residual: Q_in - Q_out = 0

        Args:
            copper_loss: Copper joule loss (W), shape (batch,)
            iron_loss: Iron eddy/hysteresis loss (W)
            cooling_power: Convective cooling (W)
            radiation_power: Radiative cooling (W)

        Returns:
            residual: Q_in - Q_out, should be near zero
        """
        Q_in = copper_loss + iron_loss
        Q_out = cooling_power + radiation_power
        residual = Q_in - Q_out

        return residual

    @staticmethod
    def fourier_law_residual(
        heat_flux: torch.Tensor,
        conductivity: torch.Tensor,
        grad_T: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fourier's law residual: q + k∇T = 0

        Args:
            heat_flux: Heat flux vector (W/m²), shape (batch, 3) or (batch, 2)
            conductivity: Material thermal conductivity (W/m·K), shape (batch,)
            grad_T: Temperature gradient (K/m), shape (batch, 3) or (batch, 2)

        Returns:
            residual: ||q + k∇T||
        """
        predicted_flux = conductivity.unsqueeze(-1) * grad_T
        residual = torch.norm(heat_flux + predicted_flux, dim=-1)

        return residual

    @staticmethod
    def magnet_temp_limit(
        T_magnet: torch.Tensor,
        T_max: float = ThermalConstants.MAGNET_MAX_TEMP,
    ) -> torch.Tensor:
        """
        Soft penalty for magnet temperature exceeding limit.
        Uses smooth max function.

        Args:
            T_magnet: Predicted magnet temperature (°C), shape (batch,)
            T_max: Maximum allowed temperature (°C)

        Returns:
            penalty: Smooth penalty, 0 if T < T_max
        """
        excess = T_magnet - T_max
        penalty = F.softplus(excess, beta=10.0)

        return penalty

    @staticmethod
    def winding_temp_limit(
        T_winding: torch.Tensor,
        T_max: float = ThermalConstants.WINDING_MAX_TEMP,
    ) -> torch.Tensor:
        """
        Soft penalty for winding temperature exceeding limit.

        Args:
            T_winding: Predicted winding temperature (°C), shape (batch,)
            T_max: Maximum allowed temperature (°C)

        Returns:
            penalty: Smooth penalty
        """
        excess = T_winding - T_max
        penalty = F.softplus(excess, beta=5.0)  # More aggressive than magnet

        return penalty

    @staticmethod
    def thermal_resistance_consistency(
        R_predicted: torch.Tensor,
        geometry_dict: Dict,
        material_props: Dict,
    ) -> torch.Tensor:
        """
        Residual enforcing thermal resistance consistency with physics.

        Args:
            R_predicted: Predicted resistance matrix, shape (num_nodes, num_nodes)
            geometry_dict: Geometric parameters
            material_props: Material thermal properties

        Returns:
            residual: Deviation from expected resistances
        """
        # Build expected resistances from first principles
        # This is a simplified version; full version would compute all paths

        d_gap = geometry_dict.get('air_gap_distance', 0.005)
        A_gap = geometry_dict.get('air_gap_area', 0.5)
        k_air = ThermalConstants.CONDUCTIVITY['air']

        R_expected_gap = d_gap / (k_air * A_gap)

        # Extract predicted gap resistance
        R_pred_gap = R_predicted[2, 5] if R_predicted.shape[0] > 5 else 0.0

        residual = R_pred_gap - R_expected_gap

        return residual

    @staticmethod
    def newton_cooling_residual(
        Q_convection: torch.Tensor,
        h: torch.Tensor,
        area: torch.Tensor,
        T_surface: torch.Tensor,
        T_fluid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Newton's cooling law residual: Q - hA(Ts - Tf) = 0

        Args:
            Q_convection: Convective heat transfer (W), shape (batch,)
            h: Convection coefficient (W/m²·K), shape (batch,)
            area: Surface area (m²), shape (batch,)
            T_surface: Surface temperature (°C), shape (batch,)
            T_fluid: Fluid temperature (°C), shape (batch,)

        Returns:
            residual: Q - hA(Ts - Tf)
        """
        Q_expected = h * area * (T_surface - T_fluid)
        residual = Q_convection - Q_expected

        return residual

    @staticmethod
    def compute_total_thermal_loss(
        energy_residual: torch.Tensor,
        fourier_residual: torch.Tensor,
        magnet_penalty: torch.Tensor,
        winding_penalty: torch.Tensor,
        cooling_residual: torch.Tensor,
        weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Weighted sum of all thermal physics residuals.

        Args:
            energy_residual: Energy conservation residual
            fourier_residual: Fourier law residual
            magnet_penalty: Magnet temperature penalty
            winding_penalty: Winding temperature penalty
            cooling_residual: Cooling law residual
            weights: Loss weights (default: equal)

        Returns:
            total_loss: Scalar loss tensor
        """
        if weights is None:
            weights = {
                'energy': 1.0,
                'fourier': 1.0,
                'magnet': 10.0,
                'winding': 10.0,
                'cooling': 1.0,
            }

        loss = (
            weights['energy'] * torch.mean(energy_residual**2) +
            weights['fourier'] * torch.mean(fourier_residual**2) +
            weights['magnet'] * torch.mean(magnet_penalty) +
            weights['winding'] * torch.mean(winding_penalty) +
            weights['cooling'] * torch.mean(cooling_residual**2)
        )

        return loss


# Convenience function for module initialization
def create_thermal_network_bergey() -> LumpedParameterThermalNetwork:
    """
    Factory function creating a pre-configured thermal network for 15-kW Bergey PMSG.

    Returns:
        network: Initialized LumpedParameterThermalNetwork
    """
    network = LumpedParameterThermalNetwork()

    # Typical Bergey geometry (conservative estimates)
    geometry = {
        'winding_volume': 0.012,
        'winding_surface': 0.3,
        'stator_core_volume': 0.015,
        'air_gap_distance': 0.004,  # 4mm
        'air_gap_area': 0.7,
        'rotor_radius': 0.15,
        'stack_length': 0.15,
        'magnet_volume': 0.003,
        'rotor_core_volume': 0.020,
        'housing_surface': 2.0,
    }

    mass = {
        'winding': 0.8,
        'stator_core': 1.5,
        'air_gap': 0.001,
        'magnet_sintered': 0.3,
        'magnet_printed': 0.05,
        'rotor_core': 2.0,
        'housing': 0.5,
    }

    network.build_resistance_matrix(geometry)
    network.build_capacitance_vector(mass)

    return network
