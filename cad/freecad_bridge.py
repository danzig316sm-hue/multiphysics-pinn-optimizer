"""
FreeCAD FEM Bridge — structural and electromagnetic analysis via FreeCAD.

Loads STEP geometry into FreeCAD headlessly, sets up FEM meshes, applies
material properties and boundary conditions, runs CalculiX (structural) or
Elmer (EM), and extracts results as numpy arrays.

FreeCAD must be installed. Common Python lib paths:
  Linux:   /usr/lib/freecad/lib  or  /usr/lib/freecad-python3/lib
  macOS:   /Applications/FreeCAD.app/Contents/Resources/lib
  Windows: C:\\Program Files\\FreeCAD 0.21\\bin
"""

from __future__ import annotations

import os
import sys
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# FreeCAD import with path detection
# ---------------------------------------------------------------------------

_FREECAD_SEARCH_PATHS = [
    "/usr/lib/freecad/lib",
    "/usr/lib/freecad-python3/lib",
    "/usr/local/lib/freecad/lib",
    "/usr/share/freecad/lib",
    "/opt/freecad/lib",
    "/Applications/FreeCAD.app/Contents/Resources/lib",
    "C:\\Program Files\\FreeCAD 0.21\\bin",
    "C:\\Program Files\\FreeCAD 0.20\\bin",
]


def _find_freecad() -> Optional[str]:
    """Auto-detect FreeCAD lib path by trying known locations."""
    for path in _FREECAD_SEARCH_PATHS:
        if os.path.isdir(path):
            return path
    return None


def _import_freecad(lib_path: Optional[str] = None) -> bool:
    """
    Attempt to import FreeCAD. Returns True on success.

    Args:
        lib_path: Override path to FreeCAD lib directory. If None, auto-detects.
    """
    if lib_path is None:
        lib_path = _find_freecad()

    if lib_path and lib_path not in sys.path:
        sys.path.insert(0, lib_path)

    try:
        import FreeCAD  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Material:
    """Mechanical / EM material properties."""
    name: str
    youngs_modulus_pa: float = 210e9      # steel default
    poissons_ratio: float = 0.30
    density_kg_m3: float = 7850.0         # steel
    yield_strength_pa: float = 250e6
    # EM properties
    relative_permeability: float = 1.0    # 1 = non-magnetic
    conductivity_s_m: float = 6e7         # copper

    # Common presets
    @classmethod
    def steel(cls) -> "Material":
        return cls(name="Steel", youngs_modulus_pa=210e9, poissons_ratio=0.30,
                   density_kg_m3=7850.0, yield_strength_pa=250e6,
                   relative_permeability=100.0)

    @classmethod
    def aluminium(cls) -> "Material":
        return cls(name="Aluminium", youngs_modulus_pa=69e9, poissons_ratio=0.33,
                   density_kg_m3=2700.0, yield_strength_pa=270e6,
                   relative_permeability=1.0)

    @classmethod
    def copper(cls) -> "Material":
        return cls(name="Copper", youngs_modulus_pa=110e9, poissons_ratio=0.35,
                   density_kg_m3=8960.0, yield_strength_pa=210e6,
                   conductivity_s_m=5.96e7)

    @classmethod
    def pla(cls) -> "Material":
        return cls(name="PLA", youngs_modulus_pa=3.5e9, poissons_ratio=0.36,
                   density_kg_m3=1240.0, yield_strength_pa=50e6,
                   relative_permeability=1.0, conductivity_s_m=1e-14)


@dataclass
class BoundaryCondition:
    """FEM boundary condition spec."""
    kind: str           # "fixed", "force", "pressure", "symmetry"
    face_index: int     # 0-based face index on the shape
    value: float = 0.0  # magnitude (N for force, Pa for pressure)
    direction: tuple = (0.0, 0.0, -1.0)  # force direction unit vector


@dataclass
class FEMResult:
    """Results extracted from a FreeCAD FEM analysis."""
    analysis_type: str                # "structural" or "em"
    # Structural outputs
    von_mises: Optional[np.ndarray] = None          # Pa at each node
    displacement_magnitude: Optional[np.ndarray] = None  # m
    displacement_vectors: Optional[np.ndarray] = None    # (N, 3) m
    principal_stress_1: Optional[np.ndarray] = None
    # EM outputs
    magnetic_flux_density: Optional[np.ndarray] = None   # T at each node
    electric_potential: Optional[np.ndarray] = None      # V
    # Metadata
    n_nodes: int = 0
    n_elements: int = 0
    solver_converged: bool = True
    warnings: list[str] = field(default_factory=list)

    def max_von_mises(self) -> float:
        if self.von_mises is not None and len(self.von_mises) > 0:
            return float(self.von_mises.max())
        return 0.0

    def max_displacement(self) -> float:
        if self.displacement_magnitude is not None and len(self.displacement_magnitude) > 0:
            return float(self.displacement_magnitude.max())
        return 0.0

    def safety_factor(self, yield_strength_pa: float) -> float:
        """Von Mises safety factor vs. yield strength."""
        max_vm = self.max_von_mises()
        return yield_strength_pa / max_vm if max_vm > 0 else float("inf")

    def summary(self) -> dict:
        return {
            "analysis_type": self.analysis_type,
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "solver_converged": self.solver_converged,
            "max_von_mises_MPa": self.max_von_mises() / 1e6,
            "max_displacement_mm": self.max_displacement() * 1e3,
        }


# ---------------------------------------------------------------------------
# FreeCAD Bridge
# ---------------------------------------------------------------------------

class FreeCADBridge:
    """
    Headless FreeCAD FEM interface.

    Usage::

        from cad.freecad_bridge import FreeCADBridge, Material, BoundaryCondition

        bridge = FreeCADBridge()  # auto-detects FreeCAD path

        result = bridge.run_structural(
            step_file="output/coil_former.step",
            material=Material.aluminium(),
            boundary_conditions=[
                BoundaryCondition(kind="fixed", face_index=0),
                BoundaryCondition(kind="force", face_index=2,
                                  value=500.0, direction=(0, 0, -1)),
            ],
        )
        print(result.summary())
    """

    def __init__(self, freecad_lib_path: Optional[str] = None, work_dir: Optional[str] = None):
        """
        Args:
            freecad_lib_path: Path to FreeCAD lib directory. Auto-detected if None.
            work_dir:         Directory for temporary FEM files. Uses system temp if None.
        """
        self._available = _import_freecad(freecad_lib_path)
        if not self._available:
            print("[FreeCADBridge] FreeCAD not found. "
                  "Set freecad_lib_path or install FreeCAD.")
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="pinn_fem_")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_structural(
        self,
        step_file: str,
        material: Material,
        boundary_conditions: list[BoundaryCondition],
        mesh_size_mm: float = 5.0,
        solver: str = "ccx",           # CalculiX
    ) -> FEMResult:
        """
        Run a linear-elastic structural FEM analysis on a STEP file.

        Args:
            step_file:           Path to the STEP geometry file.
            material:            Material properties.
            boundary_conditions: List of BoundaryCondition objects.
            mesh_size_mm:        Target mesh element size in mm.
            solver:              FEM solver ("ccx" = CalculiX).

        Returns:
            FEMResult with von Mises stress and displacement fields.
        """
        if not self._available:
            return self._stub_result("structural", "FreeCAD unavailable")

        try:
            return self._run_structural_impl(
                step_file, material, boundary_conditions, mesh_size_mm, solver
            )
        except Exception as exc:
            result = FEMResult(analysis_type="structural", solver_converged=False)
            result.warnings.append(f"FEM failed: {exc}")
            print(f"[FreeCADBridge] Structural FEM error: {exc}")
            return result

    def run_magnetostatic(
        self,
        step_file: str,
        material: Material,
        current_density_a_m2: float,
        mesh_size_mm: float = 3.0,
    ) -> FEMResult:
        """
        Run a magnetostatic FEM analysis (requires Elmer solver).

        Args:
            step_file:               Path to the STEP geometry.
            material:                Material with relative_permeability set.
            current_density_a_m2:    Applied current density in A/m².
            mesh_size_mm:            Target mesh element size.

        Returns:
            FEMResult with magnetic_flux_density field.
        """
        if not self._available:
            return self._stub_result("em", "FreeCAD unavailable")

        try:
            return self._run_magnetostatic_impl(
                step_file, material, current_density_a_m2, mesh_size_mm
            )
        except Exception as exc:
            result = FEMResult(analysis_type="em", solver_converged=False)
            result.warnings.append(f"EM FEM failed: {exc}")
            print(f"[FreeCADBridge] Magnetostatic FEM error: {exc}")
            return result

    # ------------------------------------------------------------------
    # Structural implementation
    # ------------------------------------------------------------------

    def _run_structural_impl(
        self,
        step_file: str,
        material: Material,
        boundary_conditions: list[BoundaryCondition],
        mesh_size_mm: float,
        solver: str,
    ) -> FEMResult:
        import FreeCAD as App
        import Part
        import ObjectsFem
        import femmesh.gmshtools as gmsh_tools

        doc = App.newDocument("FEM_Structural")

        # Load STEP geometry
        shape = Part.Shape()
        shape.read(step_file)
        part_obj = doc.addObject("Part::Feature", "Geometry")
        part_obj.Shape = shape
        doc.recompute()

        # FEM analysis container
        analysis = ObjectsFem.makeAnalysis(doc, "Analysis")

        # Material
        mat_obj = ObjectsFem.makeMaterialSolid(doc, "Material")
        mat_obj.Material = {
            "Name": material.name,
            "YoungsModulus": f"{material.youngs_modulus_pa / 1e6} MPa",
            "PoissonRatio": str(material.poissons_ratio),
            "Density": f"{material.density_kg_m3} kg/m^3",
        }
        analysis.addObject(mat_obj)

        # Mesh via Gmsh
        mesh_obj = ObjectsFem.makeMeshGmsh(doc, "Mesh")
        mesh_obj.Part = part_obj
        mesh_obj.CharacteristicLengthMax = mesh_size_mm
        gmsh_runner = gmsh_tools.GmshTools(mesh_obj)
        gmsh_runner.create_mesh()
        analysis.addObject(mesh_obj)

        # Boundary conditions
        faces = list(shape.Faces)
        for bc in boundary_conditions:
            face_idx = min(bc.face_index, len(faces) - 1)
            face_ref = (part_obj, f"Face{face_idx + 1}")

            if bc.kind == "fixed":
                con = ObjectsFem.makeConstraintFixed(doc, "FixedConstraint")
                con.References = [face_ref]
                analysis.addObject(con)

            elif bc.kind == "force":
                con = ObjectsFem.makeConstraintForce(doc, "ForceConstraint")
                con.References = [face_ref]
                con.Force = bc.value
                con.DirectionVector = App.Vector(*bc.direction)
                analysis.addObject(con)

            elif bc.kind == "pressure":
                con = ObjectsFem.makeConstraintPressure(doc, "PressureConstraint")
                con.References = [face_ref]
                con.Pressure = bc.value
                analysis.addObject(con)

        # Solver
        if solver == "ccx":
            import femsolver.calculix.solver as ccx
            solver_obj = ccx.create(doc, "CalculiX")
        else:
            raise ValueError(f"Unknown solver: {solver}")

        analysis.addObject(solver_obj)
        doc.recompute()

        # Run
        from femtools import ccxtools
        fea = ccxtools.FemToolsCcx(analysis, solver_obj)
        fea.update_objects()
        fea.setup_working_dir()
        fea.setup_ccx()
        message = fea.run()

        return self._extract_structural_results(fea, message)

    def _extract_structural_results(self, fea, solver_message: str) -> FEMResult:
        """Pull results from a completed CalculiX run."""
        result = FEMResult(analysis_type="structural")

        if solver_message:
            result.warnings.append(solver_message)

        try:
            fea.load_results()
        except Exception as exc:
            result.solver_converged = False
            result.warnings.append(f"Result load failed: {exc}")
            return result

        for res_obj in fea.results:
            if hasattr(res_obj, "vonMises") and res_obj.vonMises:
                result.von_mises = np.array(res_obj.vonMises)
            if hasattr(res_obj, "DisplacementLengths") and res_obj.DisplacementLengths:
                result.displacement_magnitude = np.array(res_obj.DisplacementLengths)
            if hasattr(res_obj, "DisplacementVectors") and res_obj.DisplacementVectors:
                result.displacement_vectors = np.array(
                    [[v.x, v.y, v.z] for v in res_obj.DisplacementVectors]
                )
            if hasattr(res_obj, "NodeCount"):
                result.n_nodes = res_obj.NodeCount

        return result

    # ------------------------------------------------------------------
    # Magnetostatic implementation
    # ------------------------------------------------------------------

    def _run_magnetostatic_impl(
        self,
        step_file: str,
        material: Material,
        current_density: float,
        mesh_size_mm: float,
    ) -> FEMResult:
        """
        Magnetostatic analysis via FreeCAD + Elmer.
        Requires Elmer solver to be installed and on PATH.
        """
        import FreeCAD as App
        import Part
        import ObjectsFem

        doc = App.newDocument("FEM_EM")
        shape = Part.Shape()
        shape.read(step_file)
        part_obj = doc.addObject("Part::Feature", "Geometry")
        part_obj.Shape = shape
        doc.recompute()

        analysis = ObjectsFem.makeAnalysis(doc, "Analysis")

        mat_obj = ObjectsFem.makeMaterialSolid(doc, "Material")
        mat_obj.Material = {
            "Name": material.name,
            "RelativePermeability": str(material.relative_permeability),
            "Conductivity": f"{material.conductivity_s_m} S/m",
        }
        analysis.addObject(mat_obj)

        import ObjectsFem
        solver_obj = ObjectsFem.makeSolverElmer(doc, "Elmer")
        analysis.addObject(solver_obj)

        from femsolver.elmer import writer as elmer_writer
        w = elmer_writer.Writer()
        w.analysis = analysis
        w.solver = solver_obj
        w.write_solver_input(self.work_dir)

        # Elmer must be on PATH
        import subprocess
        elmer_result = subprocess.run(
            ["ElmerSolver", "case.sif"],
            cwd=self.work_dir,
            capture_output=True,
            text=True,
        )

        result = FEMResult(analysis_type="em")
        if elmer_result.returncode != 0:
            result.solver_converged = False
            result.warnings.append(elmer_result.stderr[:500])
        else:
            result = self._parse_elmer_results(result)

        return result

    def _parse_elmer_results(self, result: FEMResult) -> FEMResult:
        """Parse Elmer .vtu or .dat output files from work_dir."""
        vtu_files = [
            f for f in os.listdir(self.work_dir)
            if f.endswith(".vtu") or f.endswith(".dat")
        ]
        if not vtu_files:
            result.warnings.append("No Elmer output files found.")
            return result

        try:
            import meshio
            mesh = meshio.read(os.path.join(self.work_dir, vtu_files[0]))
            if "magnetic flux density" in mesh.point_data:
                B = mesh.point_data["magnetic flux density"]
                result.magnetic_flux_density = np.linalg.norm(B, axis=1)
            result.n_nodes = len(mesh.points)
        except ImportError:
            result.warnings.append("meshio not installed — cannot parse Elmer output. "
                                   "Install with: pip install meshio")

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stub_result(self, analysis_type: str, reason: str) -> FEMResult:
        result = FEMResult(analysis_type=analysis_type, solver_converged=False)
        result.warnings.append(reason)
        return result

    def cleanup(self):
        """Remove temporary work directory."""
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)
