"""
solvers/__init__.py
"""
from solvers.base_solver import GeometrySpec
from solvers.featool_solver import FEAToolSolver
from solvers.sw_verification import SolidWorksVerification

__all__ = ["GeometrySpec", "FEAToolSolver", "SolidWorksVerification"]
