"""
Mobius-Nova Agent Package
=========================
Specialized Claude API subagents for the multiphysics PINN pipeline.

Agent hierarchy:
    orchestrator.py             ← Top-level coordinator
        └── agents/
            ├── base_agent.py       ← Shared Claude API wrapper
            ├── pinn_agent.py       ← PINN training & physics residuals
            ├── geometry_agent.py   ← Bézier / Halbach geometry
            ├── physics_agent.py    ← CFD · Thermal · EM solver dispatch
            └── optimizer_agent.py  ← Bayesian design-space exploration
"""

from agents.base_agent import BaseAgent
from agents.pinn_agent import PINNAgent
from agents.geometry_agent import GeometryAgent
from agents.physics_agent import PhysicsAgent
from agents.optimizer_agent import OptimizerAgent

__all__ = [
    "BaseAgent",
    "PINNAgent",
    "GeometryAgent",
    "PhysicsAgent",
    "OptimizerAgent",
]
