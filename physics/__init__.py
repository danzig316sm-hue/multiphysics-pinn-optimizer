"""
Multiphysics PINN Physics Modules — Mobius-Nova Energy Platform

This package contains differentiable physics implementations for the Mobius-Nova
multiphysics PINN optimizer targeting the Bergey 15-kW direct-drive PMSG.

Physics Domains:
    - electromagnetics: Maxwell's equations, magnetic circuit, iron/copper losses
    - thermal: Lumped parameter network, convection, radiation, passive intake cooling
    - structural: Centrifugal stress, deformation, vibration, bond integrity
    - aerodynamics: BEM theory, wind resource, Betz limit, QBlade interface
    - multiphysics_orchestrator: Fixed-point coupled analysis, constraint checking

All loss classes return differentiable PyTorch tensors for PINN backpropagation.
"""

from .aerodynamics import (
    AerodynamicConstants,
    BEMSolver,
    AeroPINNLoss,
    WindResourceCalculator,
    QBladeInterface,
)

from .electromagnetics import (
    EMConstants,
    MagneticCircuitModel,
    IronLossModel,
    DemagnetizationChecker,
    EMPINNLoss,
)

from .thermal import (
    ThermalConstants,
    LumpedParameterThermalNetwork,
    ConvectionModel,
    RadiationModel,
    PassiveIntakeCoolingModel,
    ThermalPINNLoss,
    create_thermal_network_bergey,
)

from .structural import (
    StructuralConstants,
    CentrifugalStressModel,
    DeformationModel,
    GravitationalLoadModel,
    VibrationModel,
    StructuralPINNLoss,
)

from .multiphysics_orchestrator import (
    MultiphysicsCouplingMap,
    MultiphysicsOrchestrator,
    CoupledPINNLoss,
    DesignConstraintChecker,
    OperatingConditionSweep,
    MultiphysicsLogger,
)

__all__ = [
    # Aerodynamics
    "AerodynamicConstants",
    "BEMSolver",
    "AeroPINNLoss",
    "WindResourceCalculator",
    "QBladeInterface",
    # Electromagnetics
    "EMConstants",
    "MagneticCircuitModel",
    "IronLossModel",
    "DemagnetizationChecker",
    "EMPINNLoss",
    # Thermal
    "ThermalConstants",
    "LumpedParameterThermalNetwork",
    "ConvectionModel",
    "RadiationModel",
    "PassiveIntakeCoolingModel",
    "ThermalPINNLoss",
    "create_thermal_network_bergey",
    # Structural
    "StructuralConstants",
    "CentrifugalStressModel",
    "DeformationModel",
    "GravitationalLoadModel",
    "VibrationModel",
    "StructuralPINNLoss",
    # Multiphysics Orchestrator
    "MultiphysicsCouplingMap",
    "MultiphysicsOrchestrator",
    "CoupledPINNLoss",
    "DesignConstraintChecker",
    "OperatingConditionSweep",
    "MultiphysicsLogger",
]

__version__ = "0.2.0"
