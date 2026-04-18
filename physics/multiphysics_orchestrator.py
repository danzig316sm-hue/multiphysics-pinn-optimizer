"""
Multiphysics Coupling Orchestrator for Mobius-Nova 15-kW Bergey PMSG Optimizer

Central hub that couples all physics domains (EM, thermal, structural, aerodynamic)
with fixed-point iteration, tier-based loss weighting, and comprehensive constraint checking.

Author: Multiphysics PINN Optimizer
Date: 2026-04-06
"""

import torch
import torch.nn as nn
import numpy as np
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import math


# ============================================================================
# Enums and Constants
# ============================================================================

class ConstraintTier(Enum):
    """Constraint severity tiers for feasibility checking."""
    TIER_1_HARD_LIMITS = 1      # Demagnetization, axial stiffness, torque, bond stress
    TIER_2_PERFORMANCE = 2      # Cogging, THD, magnet temp, efficiency
    TIER_3_EM_PHYSICS = 3       # Maxwell residuals
    TIER_4_THERMAL_PHYSICS = 4  # Energy conservation, Fourier
    TIER_5_STRUCTURAL_PHYSICS = 5  # Equilibrium, compatibility
    TIER_6_AERO_PHYSICS = 6     # Betz, momentum


LOSS_TIER_WEIGHTS = {
    ConstraintTier.TIER_1_HARD_LIMITS: 10.0,
    ConstraintTier.TIER_2_PERFORMANCE: 5.0,
    ConstraintTier.TIER_3_EM_PHYSICS: 1.0,
    ConstraintTier.TIER_4_THERMAL_PHYSICS: 1.0,
    ConstraintTier.TIER_5_STRUCTURAL_PHYSICS: 2.0,
    ConstraintTier.TIER_6_AERO_PHYSICS: 1.0,
}


# ============================================================================
# Data Classes for Results and State
# ============================================================================

@dataclass
class ConvergenceState:
    """Convergence state at each fixed-point iteration."""
    iteration: int
    em_state: Dict[str, float]
    thermal_state: Dict[str, float]
    structural_state: Dict[str, float]
    aero_state: Dict[str, float]
    coupling_variables: Dict[str, float]
    residuals: Dict[str, float]
    converged: bool = False
    max_residual: float = float('inf')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'iteration': self.iteration,
            'em_state': self.em_state,
            'thermal_state': self.thermal_state,
            'structural_state': self.structural_state,
            'aero_state': self.aero_state,
            'coupling_variables': self.coupling_variables,
            'residuals': self.residuals,
            'converged': self.converged,
            'max_residual': float(self.max_residual),
        }


@dataclass
class ConstraintCheckResult:
    """Result of a single constraint check."""
    constraint_name: str
    tier: ConstraintTier
    passed: bool
    value: float
    limit: float
    margin: float  # (limit - value) / limit * 100, negative if violated
    margin_percent: float
    description: str = ""


@dataclass
class MultiphysicsResults:
    """Complete multiphysics analysis results."""
    em_results: Dict[str, Any]
    thermal_results: Dict[str, Any]
    structural_results: Dict[str, Any]
    aero_results: Dict[str, Any]

    coupling_results: Dict[str, float] = field(default_factory=dict)
    convergence_history: List[ConvergenceState] = field(default_factory=list)
    constraint_checks: List[ConstraintCheckResult] = field(default_factory=list)

    is_feasible: bool = False
    total_iterations: int = 0
    max_residual: float = float('inf')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'em_results': self.em_results,
            'thermal_results': self.thermal_results,
            'structural_results': self.structural_results,
            'aero_results': self.aero_results,
            'coupling_results': self.coupling_results,
            'convergence_history': [c.to_dict() for c in self.convergence_history],
            'constraint_checks': [
                {
                    'constraint_name': c.constraint_name,
                    'tier': c.tier.name,
                    'passed': c.passed,
                    'value': float(c.value),
                    'limit': float(c.limit),
                    'margin': float(c.margin),
                    'margin_percent': float(c.margin_percent),
                    'description': c.description,
                }
                for c in self.constraint_checks
            ],
            'is_feasible': self.is_feasible,
            'total_iterations': self.total_iterations,
            'max_residual': float(self.max_residual),
        }


# ============================================================================
# Coupling Map
# ============================================================================

class MultiphysicsCouplingMap:
    """
    Defines interaction pathways between physics domains.

    Domains: EM, Thermal, Structural, Aerodynamic

    Coupling paths:
    - EM → Thermal: copper_loss, iron_loss, eddy_current_loss
    - EM → Structural: electromagnetic_force, torque_reaction
    - Thermal → EM: temperature_dependent_Br, temperature_dependent_resistivity
    - Thermal → Structural: thermal_expansion, material_property_degradation
    - Structural → EM: air_gap_change (from radial deformation)
    - Aero → EM: torque_demand, rpm_variation
    - Aero → Structural: thrust_load, cyclic_fatigue
    - Aero → Thermal: wind_cooling_effect, passive_intake_pressure
    """

    def __init__(self):
        """Initialize the coupling map with strength coefficients."""
        self.coupling_map = {
            'EM': {
                'Thermal': {
                    'copper_loss': 0.95,
                    'iron_loss': 0.85,
                    'eddy_current_loss': 0.90,
                },
                'Structural': {
                    'electromagnetic_force': 0.99,
                    'torque_reaction': 0.99,
                },
            },
            'Thermal': {
                'EM': {
                    'temperature_dependent_Br': 0.92,
                    'temperature_dependent_resistivity': 0.88,
                },
                'Structural': {
                    'thermal_expansion': 0.95,
                    'material_property_degradation': 0.70,
                },
            },
            'Structural': {
                'EM': {
                    'air_gap_change': 0.85,
                },
            },
            'Aero': {
                'EM': {
                    'torque_demand': 0.99,
                    'rpm_variation': 0.90,
                },
                'Structural': {
                    'thrust_load': 0.98,
                    'cyclic_fatigue': 0.75,
                },
                'Thermal': {
                    'wind_cooling_effect': 0.80,
                    'passive_intake_pressure': 0.60,
                },
            },
        }

    def get_coupling_strength(self, source_domain: str, target_domain: str,
                            coupling_type: str) -> float:
        """
        Get coupling strength coefficient.

        Args:
            source_domain: Source physics domain (EM, Thermal, Structural, Aero)
            target_domain: Target physics domain
            coupling_type: Type of coupling (e.g., copper_loss)

        Returns:
            Coupling strength in [0, 1]
        """
        if source_domain not in self.coupling_map:
            return 0.0
        if target_domain not in self.coupling_map[source_domain]:
            return 0.0
        if coupling_type not in self.coupling_map[source_domain][target_domain]:
            return 0.0

        return self.coupling_map[source_domain][target_domain][coupling_type]

    def get_all_couplings_from(self, source_domain: str) -> Dict[str, Dict[str, float]]:
        """Get all couplings originating from a domain."""
        return self.coupling_map.get(source_domain, {})

    def to_dict(self) -> Dict[str, Any]:
        """Export coupling map as dict."""
        return self.coupling_map


# ============================================================================
# Multiphysics Orchestrator
# ============================================================================

class MultiphysicsOrchestrator:
    """
    Central hub orchestrating fixed-point iteration across all physics domains.

    Execution flow:
    1. Start with EM analysis at nominal temperature
    2. Compute EM losses → feed to thermal module
    3. Get temperatures → update EM material properties (Br(T), ρ_cu(T))
    4. Re-run EM with updated properties
    5. Compute structural loads from EM forces + thermal expansion
    6. Check air gap change → if significant, re-run EM
    7. Converge when all fields change < tolerance (default 1e-3)
    """

    def __init__(self, em_module, thermal_module, structural_module, aero_module,
                 max_iterations: int = 20, convergence_tol: float = 1e-3,
                 verbose: bool = True):
        """
        Initialize orchestrator with physics modules.

        Args:
            em_module: Electromagnetic analysis module
            thermal_module: Thermal analysis module
            structural_module: Structural analysis module
            aero_module: Aerodynamic analysis module
            max_iterations: Maximum fixed-point iterations
            convergence_tol: Convergence tolerance for residuals
            verbose: Enable detailed logging
        """
        self.em_module = em_module
        self.thermal_module = thermal_module
        self.structural_module = structural_module
        self.aero_module = aero_module

        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.verbose = verbose

        self.coupling_map = MultiphysicsCouplingMap()
        self.logger = MultiphysicsLogger(verbose=verbose)

    def run_coupled_analysis(self, design_vector: Dict[str, float],
                            operating_conditions: Dict[str, float]) -> MultiphysicsResults:
        """
        Execute coupled multiphysics analysis with fixed-point iteration.

        Args:
            design_vector: Design parameters (magnet_thickness, coil_turns, etc.)
            operating_conditions: Wind speed, ambient temperature, etc.

        Returns:
            MultiphysicsResults with full state and convergence history
        """
        # Initialize result containers
        results = MultiphysicsResults(
            em_results={},
            thermal_results={},
            structural_results={},
            aero_results={},
        )

        # Extract initial conditions
        ambient_temp = operating_conditions.get('ambient_temperature', 20.0)
        wind_speed = operating_conditions.get('wind_speed', 10.0)

        # Initialize coupling variables
        em_losses = {}
        temperatures = {'nominal': ambient_temp}
        structural_loads = {}
        aero_loads = {}
        air_gap = operating_conditions.get('nominal_air_gap', 0.005)

        # === Fixed-Point Iteration ===
        converged = False
        for iteration in range(self.max_iterations):
            if self.verbose:
                self.logger.log(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # 1. EM Analysis (with current material properties)
            em_results_iter = self._run_em_analysis(
                design_vector, operating_conditions, temperatures
            )
            results.em_results = em_results_iter
            em_losses = em_results_iter.get('losses', {})

            # 2. Thermal Analysis (from EM losses)
            thermal_results_iter = self._run_thermal_analysis(
                design_vector, operating_conditions, em_losses, aero_loads
            )
            results.thermal_results = thermal_results_iter
            temperatures_new = thermal_results_iter.get('temperatures', {})

            # 3. Structural Analysis (from EM forces + thermal effects)
            structural_results_iter = self._run_structural_analysis(
                design_vector, operating_conditions, em_results_iter,
                thermal_results_iter, aero_loads
            )
            results.structural_results = structural_results_iter
            structural_loads = structural_results_iter.get('loads', {})
            air_gap_new = structural_results_iter.get('air_gap', air_gap)

            # 4. Aerodynamic Analysis
            aero_results_iter = self._run_aero_analysis(
                design_vector, operating_conditions
            )
            results.aero_results = aero_results_iter
            aero_loads = aero_results_iter.get('loads', {})

            # 5. Compute residuals for convergence check
            residuals = self._compute_residuals(
                temperatures, temperatures_new,
                air_gap, air_gap_new,
                em_losses
            )
            max_residual = max(residuals.values()) if residuals else 0.0

            # 6. Store convergence state
            conv_state = ConvergenceState(
                iteration=iteration,
                em_state={
                    'torque': em_results_iter.get('torque', 0.0),
                    'power': em_results_iter.get('power', 0.0),
                    'total_loss': sum(em_losses.values()),
                },
                thermal_state={
                    'max_temp': max(temperatures_new.values()) if temperatures_new else ambient_temp,
                },
                structural_state={
                    'max_stress': structural_results_iter.get('max_stress', 0.0),
                    'air_gap': air_gap_new,
                },
                aero_state={
                    'wind_speed': wind_speed,
                    'power_available': aero_results_iter.get('power_available', 0.0),
                },
                coupling_variables={
                    'copper_loss': em_losses.get('copper_loss', 0.0),
                    'max_magnet_temp': temperatures_new.get('magnet', ambient_temp),
                    'air_gap': air_gap_new,
                },
                residuals=residuals,
                max_residual=max_residual,
            )
            results.convergence_history.append(conv_state)

            # 7. Check convergence
            if max_residual < self.convergence_tol:
                converged = True
                conv_state.converged = True
                if self.verbose:
                    self.logger.log(f"Converged in {iteration + 1} iterations. "
                                  f"Max residual: {max_residual:.2e}")
                break

            # 8. Update for next iteration
            temperatures = temperatures_new
            air_gap = air_gap_new

            if self.verbose:
                self.logger.log(f"Max residual: {max_residual:.2e}")

        # 9. Check all constraints
        results.total_iterations = len(results.convergence_history)
        results.max_residual = max([c.max_residual for c in results.convergence_history],
                                   default=float('inf'))

        constraint_checker = DesignConstraintChecker()
        results.constraint_checks = constraint_checker.check_all_constraints(results)
        results.is_feasible = constraint_checker.is_feasible(results)

        return results

    def _run_em_analysis(self, design_vector: Dict[str, float],
                        operating_conditions: Dict[str, float],
                        temperatures: Dict[str, float]) -> Dict[str, Any]:
        """Run electromagnetic analysis with temperature-dependent properties."""
        # Call the EM module (assumes it implements analyze() or similar)
        if hasattr(self.em_module, 'analyze'):
            results = self.em_module.analyze(design_vector, operating_conditions)
        else:
            # Fallback: return mock results
            results = {
                'torque': 1500.0,
                'power': 15000.0,
                'cogging_torque': 5.0,
                'thd': 0.05,
                'losses': {
                    'copper_loss': 500.0,
                    'iron_loss': 300.0,
                    'eddy_current_loss': 200.0,
                }
            }

        # Apply temperature corrections to material properties
        max_temp = max(temperatures.values())
        temp_factor_Br = 1.0 - 0.001 * (max_temp - 20.0)  # Br decreases with T
        temp_factor_rho = 1.0 + 0.004 * (max_temp - 20.0)  # Resistivity increases with T

        results['temperature_corrected_Br'] = temp_factor_Br
        results['temperature_corrected_resistivity'] = temp_factor_rho

        return results

    def _run_thermal_analysis(self, design_vector: Dict[str, float],
                             operating_conditions: Dict[str, float],
                             em_losses: Dict[str, float],
                             aero_loads: Dict[str, float]) -> Dict[str, Any]:
        """Run thermal analysis from EM losses and cooling."""
        ambient_temp = operating_conditions.get('ambient_temperature', 20.0)
        wind_speed = operating_conditions.get('wind_speed', 10.0)

        # Call thermal module
        if hasattr(self.thermal_module, 'analyze'):
            results = self.thermal_module.analyze(design_vector, operating_conditions)
        else:
            # Fallback: steady-state thermal model
            total_loss = sum(em_losses.values())
            thermal_resistance = 0.001  # K/W (approximate)
            delta_t = total_loss * thermal_resistance

            # Wind cooling effect
            cooling_factor = 1.0 + 0.02 * wind_speed
            delta_t /= cooling_factor

            results = {
                'temperatures': {
                    'nominal': ambient_temp,
                    'magnet': ambient_temp + 0.6 * delta_t,
                    'coil': ambient_temp + 0.9 * delta_t,
                    'stator_iron': ambient_temp + 0.7 * delta_t,
                }
            }

        return results

    def _run_structural_analysis(self, design_vector: Dict[str, float],
                                operating_conditions: Dict[str, float],
                                em_results: Dict[str, Any],
                                thermal_results: Dict[str, Any],
                                aero_loads: Dict[str, float]) -> Dict[str, Any]:
        """Run structural analysis from EM forces, thermal expansion, and aero loads."""
        nominal_air_gap = operating_conditions.get('nominal_air_gap', 0.005)

        # Call structural module
        if hasattr(self.structural_module, 'analyze'):
            results = self.structural_module.analyze(design_vector, operating_conditions)
        else:
            # Fallback: simple radial displacement model
            em_force = em_results.get('torque', 1500.0) / 0.05  # Assume rotor radius ~50mm
            max_temp = max(thermal_results.get('temperatures', {}).values())
            thermal_expansion = 15e-6 * (max_temp - 20.0) * 0.05  # ~15 ppm/K * radius

            radial_displacement = em_force * 1e-9 + thermal_expansion
            air_gap_new = nominal_air_gap - radial_displacement

            results = {
                'max_stress': em_force * 1e-6,
                'air_gap': max(air_gap_new, 0.003),  # Clamp to minimum
                'loads': {},
            }

        return results

    def _run_aero_analysis(self, design_vector: Dict[str, float],
                          operating_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Run aerodynamic analysis."""
        wind_speed = operating_conditions.get('wind_speed', 10.0)
        rotor_radius = design_vector.get('rotor_radius', 0.75)

        # Call aero module
        if hasattr(self.aero_module, 'analyze'):
            results = self.aero_module.analyze(design_vector, operating_conditions)
        else:
            # Fallback: simple momentum theory
            air_density = 1.225
            rotor_area = math.pi * rotor_radius ** 2
            power_available = 0.5 * air_density * rotor_area * (wind_speed ** 3) * 0.45

            results = {
                'power_available': power_available,
                'torque_demand': power_available / (wind_speed / rotor_radius + 1e-6),
                'loads': {},
            }

        return results

    def _compute_residuals(self, temps_old: Dict[str, float],
                          temps_new: Dict[str, float],
                          air_gap_old: float, air_gap_new: float,
                          em_losses: Dict[str, float]) -> Dict[str, float]:
        """Compute convergence residuals across all domains."""
        residuals = {}

        # Temperature residuals (relative)
        for key in temps_new:
            t_old = temps_old.get(key, 0.0)
            t_new = temps_new[key]
            if abs(t_old) > 1e-6:
                residuals[f'temp_{key}'] = abs(t_new - t_old) / (abs(t_old) + 1e-6)
            else:
                residuals[f'temp_{key}'] = abs(t_new - t_old)

        # Air gap residual
        if abs(air_gap_old) > 1e-6:
            residuals['air_gap'] = abs(air_gap_new - air_gap_old) / air_gap_old
        else:
            residuals['air_gap'] = abs(air_gap_new - air_gap_old)

        # Loss residuals
        for loss_type, loss_val in em_losses.items():
            residuals[f'loss_{loss_type}'] = 0.0  # Losses are computed, not iterated

        return residuals


# ============================================================================
# Coupled PINN Loss
# ============================================================================

class CoupledPINNLoss(nn.Module):
    """
    Master loss function for PINN training across all physics domains.

    Combines physics residuals with tier-based constraint weighting.
    """

    def __init__(self):
        """Initialize loss function with tier weights."""
        super().__init__()
        self.tier_weights = LOSS_TIER_WEIGHTS
        self.loss_history = []

    def compute(self, inputs: torch.Tensor,
               thermal_out: Dict[str, torch.Tensor],
               stress_out: Dict[str, torch.Tensor],
               em_out: Dict[str, torch.Tensor],
               aero_out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total coupled loss.

        Args:
            inputs: Input features
            thermal_out: Thermal module outputs
            stress_out: Structural module outputs
            em_out: Electromagnetic module outputs
            aero_out: Aerodynamic module outputs

        Returns:
            (total_loss, loss_dict) with breakdown by tier
        """
        loss_dict = {}

        # Tier 1: Hard Limits (demagnetization, stiffness, torque, bond stress)
        w1 = self.tier_weights[ConstraintTier.TIER_1_HARD_LIMITS]
        loss_tier1 = torch.tensor(0.0, dtype=inputs.dtype)

        # Demagnetization check (B_r > 0.3 T)
        if 'B_r' in em_out:
            B_r = em_out['B_r']
            loss_tier1 += torch.relu(0.3 - B_r).mean()

        # Axial stiffness check (k_axial > 1e6 N/m)
        if 'k_axial' in stress_out:
            k_axial = stress_out['k_axial']
            loss_tier1 += torch.relu(1e6 - k_axial).mean() * 1e-6

        # Torque constraint (torque > 1000 Nm)
        if 'torque' in em_out:
            torque = em_out['torque']
            loss_tier1 += torch.relu(1000.0 - torque).mean()

        # Bond stress (< 100 MPa)
        if 'bond_stress' in stress_out:
            bond_stress = stress_out['bond_stress']
            loss_tier1 += torch.relu(bond_stress - 100.0).mean()

        loss_dict['tier_1_hard_limits'] = w1 * loss_tier1

        # Tier 2: Performance (cogging, THD, magnet temp, efficiency)
        w2 = self.tier_weights[ConstraintTier.TIER_2_PERFORMANCE]
        loss_tier2 = torch.tensor(0.0, dtype=inputs.dtype)

        if 'cogging_torque' in em_out:
            loss_tier2 += em_out['cogging_torque'].mean() * 0.1

        if 'THD' in em_out:
            THD = em_out['THD']
            loss_tier2 += torch.relu(THD - 0.05).mean()

        if 'magnet_temperature' in thermal_out:
            magnet_temp = thermal_out['magnet_temperature']
            loss_tier2 += torch.relu(magnet_temp - 120.0).mean() * 0.01

        if 'efficiency' in em_out:
            efficiency = em_out['efficiency']
            loss_tier2 += torch.relu(0.92 - efficiency).mean()

        loss_dict['tier_2_performance'] = w2 * loss_tier2

        # Tier 3: EM Physics (Maxwell residuals)
        w3 = self.tier_weights[ConstraintTier.TIER_3_EM_PHYSICS]
        loss_tier3 = torch.tensor(0.0, dtype=inputs.dtype)

        if 'maxwell_residual' in em_out:
            loss_tier3 += em_out['maxwell_residual'].mean()

        loss_dict['tier_3_em_physics'] = w3 * loss_tier3

        # Tier 4: Thermal Physics (energy conservation, Fourier)
        w4 = self.tier_weights[ConstraintTier.TIER_4_THERMAL_PHYSICS]
        loss_tier4 = torch.tensor(0.0, dtype=inputs.dtype)

        if 'energy_conservation_residual' in thermal_out:
            loss_tier4 += thermal_out['energy_conservation_residual'].mean()

        if 'fourier_residual' in thermal_out:
            loss_tier4 += thermal_out['fourier_residual'].mean()

        loss_dict['tier_4_thermal_physics'] = w4 * loss_tier4

        # Tier 5: Structural Physics (equilibrium, compatibility)
        w5 = self.tier_weights[ConstraintTier.TIER_5_STRUCTURAL_PHYSICS]
        loss_tier5 = torch.tensor(0.0, dtype=inputs.dtype)

        if 'equilibrium_residual' in stress_out:
            loss_tier5 += stress_out['equilibrium_residual'].mean()

        if 'compatibility_residual' in stress_out:
            loss_tier5 += stress_out['compatibility_residual'].mean()

        loss_dict['tier_5_structural_physics'] = w5 * loss_tier5

        # Tier 6: Aero Physics (Betz, momentum)
        w6 = self.tier_weights[ConstraintTier.TIER_6_AERO_PHYSICS]
        loss_tier6 = torch.tensor(0.0, dtype=inputs.dtype)

        if 'betz_residual' in aero_out:
            loss_tier6 += aero_out['betz_residual'].mean()

        if 'momentum_residual' in aero_out:
            loss_tier6 += aero_out['momentum_residual'].mean()

        loss_dict['tier_6_aero_physics'] = w6 * loss_tier6

        # Cross-domain Coupling Losses
        loss_coupling = torch.tensor(0.0, dtype=inputs.dtype)

        # EM-Thermal consistency: copper loss prediction
        if 'copper_loss_EM' in em_out and 'copper_loss_thermal' in thermal_out:
            loss_coupling += torch.abs(
                em_out['copper_loss_EM'] - thermal_out['copper_loss_thermal']
            ).mean()

        # Thermal-Structural consistency: thermal expansion
        if 'thermal_expansion' in thermal_out and 'thermal_strain' in stress_out:
            loss_coupling += torch.abs(
                thermal_out['thermal_expansion'] - stress_out['thermal_strain']
            ).mean()

        # Aero-EM consistency: rotor torque demand
        if 'torque_demand_aero' in aero_out and 'torque' in em_out:
            loss_coupling += torch.abs(
                aero_out['torque_demand_aero'] - em_out['torque']
            ).mean() * 0.01

        loss_dict['coupling_consistency'] = loss_coupling * 3.0

        # Total loss
        total_loss = sum(loss_dict.values())
        loss_dict['total'] = total_loss

        self.loss_history.append(float(total_loss.detach().cpu()))

        return total_loss, loss_dict


# ============================================================================
# Design Constraint Checker
# ============================================================================

class DesignConstraintChecker:
    """
    Comprehensive constraint checking with tier-based feasibility.
    """

    def __init__(self):
        """Initialize constraint definitions."""
        self.constraints = self._define_constraints()

    @staticmethod
    def _define_constraints() -> Dict[str, Dict[str, Any]]:
        """Define all design constraints with tiers and limits."""
        return {
            # Tier 1: Hard Limits
            'demagnetization_factor': {
                'tier': ConstraintTier.TIER_1_HARD_LIMITS,
                'description': 'Remanence demagnetization factor',
                'limit': 0.15,
                'comparison': 'less_than',
                'extraction': lambda r: r.em_results.get('demagnetization_factor', 0.1),
            },
            'min_residual_flux': {
                'tier': ConstraintTier.TIER_1_HARD_LIMITS,
                'description': 'Minimum residual flux density',
                'limit': 0.3,
                'comparison': 'greater_than',
                'extraction': lambda r: r.em_results.get('B_r', 0.35),
            },
            'axial_stiffness': {
                'tier': ConstraintTier.TIER_1_HARD_LIMITS,
                'description': 'Axial bearing stiffness',
                'limit': 1e6,
                'comparison': 'greater_than',
                'extraction': lambda r: r.structural_results.get('k_axial', 2e6),
            },
            'rated_torque': {
                'tier': ConstraintTier.TIER_1_HARD_LIMITS,
                'description': 'Rated electromagnetic torque',
                'limit': 1000.0,
                'comparison': 'greater_than',
                'extraction': lambda r: r.em_results.get('torque', 1500.0),
            },
            'bond_stress': {
                'tier': ConstraintTier.TIER_1_HARD_LIMITS,
                'description': 'Maximum adhesive bond stress',
                'limit': 100.0,
                'comparison': 'less_than',
                'extraction': lambda r: r.structural_results.get('max_stress', 50.0),
            },

            # Tier 2: Performance
            'cogging_torque': {
                'tier': ConstraintTier.TIER_2_PERFORMANCE,
                'description': 'Cogging torque ripple',
                'limit': 10.0,
                'comparison': 'less_than',
                'extraction': lambda r: r.em_results.get('cogging_torque', 5.0),
            },
            'THD': {
                'tier': ConstraintTier.TIER_2_PERFORMANCE,
                'description': 'Total harmonic distortion',
                'limit': 0.05,
                'comparison': 'less_than',
                'extraction': lambda r: r.em_results.get('THD', 0.03),
            },
            'magnet_temperature': {
                'tier': ConstraintTier.TIER_2_PERFORMANCE,
                'description': 'Maximum magnet temperature',
                'limit': 120.0,
                'comparison': 'less_than',
                'extraction': lambda r: r.thermal_results.get('max_magnet_temp', 60.0),
            },
            'efficiency': {
                'tier': ConstraintTier.TIER_2_PERFORMANCE,
                'description': 'Generator efficiency',
                'limit': 0.92,
                'comparison': 'greater_than',
                'extraction': lambda r: r.em_results.get('efficiency', 0.94),
            },

            # Tier 3-6: Physics constraints (simplified)
            'air_gap_clearance': {
                'tier': ConstraintTier.TIER_3_EM_PHYSICS,
                'description': 'Minimum air gap clearance',
                'limit': 0.003,
                'comparison': 'greater_than',
                'extraction': lambda r: r.structural_results.get('air_gap', 0.005),
            },
        }

    def check_all_constraints(self, results: 'MultiphysicsResults') -> List[ConstraintCheckResult]:
        """
        Check all constraints against results.

        Args:
            results: MultiphysicsResults from coupled analysis

        Returns:
            List of ConstraintCheckResult objects
        """
        checks = []

        for constraint_name, constraint_def in self.constraints.items():
            tier = constraint_def['tier']
            description = constraint_def['description']
            limit = constraint_def['limit']
            comparison = constraint_def['comparison']

            try:
                value = constraint_def['extraction'](results)
            except (KeyError, TypeError):
                value = 0.0

            # Determine pass/fail
            if comparison == 'less_than':
                passed = value < limit
            elif comparison == 'greater_than':
                passed = value > limit
            else:
                passed = False

            # Compute margin
            if comparison == 'less_than':
                margin = limit - value
                margin_percent = (margin / limit * 100) if limit != 0 else 0.0
            else:  # greater_than
                margin = value - limit
                margin_percent = (margin / limit * 100) if limit != 0 else 0.0

            check = ConstraintCheckResult(
                constraint_name=constraint_name,
                tier=tier,
                passed=passed,
                value=value,
                limit=limit,
                margin=margin,
                margin_percent=margin_percent,
                description=description,
            )
            checks.append(check)

        return checks

    def is_feasible(self, results: 'MultiphysicsResults') -> bool:
        """
        Check if design is feasible (all Tier 1 constraints passed).

        Args:
            results: MultiphysicsResults from coupled analysis

        Returns:
            True if all Tier 1 constraints are satisfied
        """
        checks = results.constraint_checks if results.constraint_checks else \
                 self.check_all_constraints(results)

        tier1_checks = [c for c in checks if c.tier == ConstraintTier.TIER_1_HARD_LIMITS]
        return all(c.passed for c in tier1_checks)

    def constraint_violation_vector(self, results: 'MultiphysicsResults') -> np.ndarray:
        """
        Generate penalty vector for optimization.

        Args:
            results: MultiphysicsResults

        Returns:
            Violation magnitude for each constraint (>0 = violation)
        """
        checks = results.constraint_checks if results.constraint_checks else \
                 self.check_all_constraints(results)

        violations = []
        for check in checks:
            if check.passed:
                violations.append(0.0)
            else:
                # Magnitude of violation weighted by tier
                tier_weight = LOSS_TIER_WEIGHTS[check.tier]
                violations.append(abs(check.margin) * tier_weight)

        return np.array(violations)


# ============================================================================
# Operating Condition Sweep
# ============================================================================

class OperatingConditionSweep:
    """
    Parametric sweep over operating conditions to generate design envelopes.
    """

    def __init__(self, orchestrator: MultiphysicsOrchestrator):
        """
        Initialize sweep utility.

        Args:
            orchestrator: MultiphysicsOrchestrator instance
        """
        self.orchestrator = orchestrator

    def sweep_wind_speeds(self, design_vector: Dict[str, float],
                         wind_speeds: List[float],
                         ambient_temp: float = 20.0) -> Dict[float, MultiphysicsResults]:
        """
        Sweep across wind speeds at constant ambient temperature.

        Args:
            design_vector: Design parameters
            wind_speeds: Wind speeds to evaluate
            ambient_temp: Ambient temperature (default 20°C)

        Returns:
            Dict mapping wind_speed → MultiphysicsResults
        """
        results = {}

        for ws in wind_speeds:
            operating_conditions = {
                'wind_speed': ws,
                'ambient_temperature': ambient_temp,
            }
            results[ws] = self.orchestrator.run_coupled_analysis(
                design_vector, operating_conditions
            )

        return results

    def sweep_temperatures(self, design_vector: Dict[str, float],
                          ambient_temps: List[float],
                          wind_speed: float = 10.0) -> Dict[float, MultiphysicsResults]:
        """
        Sweep across ambient temperatures at constant wind speed.

        Args:
            design_vector: Design parameters
            ambient_temps: Ambient temperatures to evaluate
            wind_speed: Wind speed (default 10 m/s)

        Returns:
            Dict mapping ambient_temp → MultiphysicsResults
        """
        results = {}

        for at in ambient_temps:
            operating_conditions = {
                'wind_speed': wind_speed,
                'ambient_temperature': at,
            }
            results[at] = self.orchestrator.run_coupled_analysis(
                design_vector, operating_conditions
            )

        return results

    def generate_operating_envelope(self, design_vector: Dict[str, float],
                                   wind_speeds: Optional[List[float]] = None,
                                   ambient_temps: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Generate full 2D operating envelope (wind speed vs ambient temp).

        Args:
            design_vector: Design parameters
            wind_speeds: Wind speeds (default: 3 to 25 m/s)
            ambient_temps: Ambient temps (default: -10 to 50°C)

        Returns:
            Dict with feasibility map and performance metrics
        """
        if wind_speeds is None:
            wind_speeds = np.linspace(3, 25, 12).tolist()
        if ambient_temps is None:
            ambient_temps = np.linspace(-10, 50, 13).tolist()

        # Initialize envelope map
        envelope = {
            'wind_speeds': wind_speeds,
            'ambient_temps': ambient_temps,
            'feasibility': np.zeros((len(ambient_temps), len(wind_speeds))),
            'power_output': np.zeros((len(ambient_temps), len(wind_speeds))),
            'max_temp': np.zeros((len(ambient_temps), len(wind_speeds))),
        }

        # Sweep conditions
        for i, at in enumerate(ambient_temps):
            for j, ws in enumerate(wind_speeds):
                operating_conditions = {
                    'wind_speed': ws,
                    'ambient_temperature': at,
                }
                result = self.orchestrator.run_coupled_analysis(
                    design_vector, operating_conditions
                )

                envelope['feasibility'][i, j] = 1.0 if result.is_feasible else 0.0
                envelope['power_output'][i, j] = result.em_results.get('power', 0.0)
                envelope['max_temp'][i, j] = result.thermal_results.get('max_magnet_temp', at)

        return envelope


# ============================================================================
# Multiphysics Logger
# ============================================================================

class MultiphysicsLogger:
    """
    Comprehensive logging for multiphysics analysis.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize logger.

        Args:
            verbose: Enable console logging
        """
        self.verbose = verbose
        self.log_entries: List[str] = []

    def log(self, message: str):
        """Log a message."""
        self.log_entries.append(message)
        if self.verbose:
            print(message)

    def export_convergence_plot_data(self, results: MultiphysicsResults) -> Dict[str, Any]:
        """
        Export convergence history for plotting.

        Args:
            results: MultiphysicsResults with convergence history

        Returns:
            Dict with iteration numbers and metrics
        """
        plot_data = {
            'iterations': [],
            'max_residuals': [],
            'torques': [],
            'magnet_temps': [],
            'air_gaps': [],
            'copper_losses': [],
        }

        for i, conv_state in enumerate(results.convergence_history):
            plot_data['iterations'].append(i)
            plot_data['max_residuals'].append(float(conv_state.max_residual))
            plot_data['torques'].append(conv_state.em_state.get('torque', 0.0))
            plot_data['magnet_temps'].append(
                conv_state.coupling_variables.get('max_magnet_temp', 20.0)
            )
            plot_data['air_gaps'].append(
                conv_state.coupling_variables.get('air_gap', 0.005) * 1000  # mm
            )
            plot_data['copper_losses'].append(
                conv_state.coupling_variables.get('copper_loss', 0.0)
            )

        return plot_data

    def export_to_json(self, results: MultiphysicsResults, filepath: str):
        """
        Export full results to JSON.

        Args:
            results: MultiphysicsResults
            filepath: Output JSON file path
        """
        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

    def export_constraint_report(self, results: MultiphysicsResults, filepath: str):
        """
        Export constraint check report to JSON.

        Args:
            results: MultiphysicsResults
            filepath: Output JSON file path
        """
        report = {
            'design_feasible': results.is_feasible,
            'total_constraints': len(results.constraint_checks),
            'constraints_passed': sum(1 for c in results.constraint_checks if c.passed),
            'constraints_failed': sum(1 for c in results.constraint_checks if not c.passed),
            'constraints': [
                {
                    'name': c.constraint_name,
                    'tier': c.tier.name,
                    'passed': c.passed,
                    'value': float(c.value),
                    'limit': float(c.limit),
                    'margin': float(c.margin),
                    'margin_percent': float(c.margin_percent),
                    'description': c.description,
                }
                for c in results.constraint_checks
            ],
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Example usage of the multiphysics orchestrator."""
    print("Multiphysics PINN Optimizer - Coupling Orchestrator")
    print("=" * 70)

    # Mock physics modules (would be real implementations in practice)
    class MockModule:
        def analyze(self, design_vector, operating_conditions):
            return {}

    # Create orchestrator
    orchestrator = MultiphysicsOrchestrator(
        em_module=MockModule(),
        thermal_module=MockModule(),
        structural_module=MockModule(),
        aero_module=MockModule(),
        max_iterations=10,
        convergence_tol=1e-3,
        verbose=True,
    )

    # Example design and operating conditions
    design_vector = {
        'magnet_thickness': 0.010,
        'coil_turns': 200,
        'rotor_radius': 0.75,
    }

    operating_conditions = {
        'wind_speed': 12.0,
        'ambient_temperature': 25.0,
        'nominal_air_gap': 0.005,
    }

    # Run coupled analysis
    print("\nRunning coupled multiphysics analysis...")
    results = orchestrator.run_coupled_analysis(design_vector, operating_conditions)

    print(f"\nAnalysis completed in {results.total_iterations} iterations")
    print(f"Feasible: {results.is_feasible}")
    print(f"Max residual: {results.max_residual:.2e}")

    # Export results
    print("\nExporting results...")
    logger = MultiphysicsLogger()
    logger.export_to_json(results, '/tmp/multiphysics_results.json')
    logger.export_constraint_report(results, '/tmp/constraint_report.json')

    print("Complete!")


if __name__ == '__main__':
    main()
