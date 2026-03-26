# CAD Integration — CadQuery, FreeCAD FEM & Multi-Objective Optimizer

## Overview

The CAD stack closes the loop between parametric geometry and physics
optimization:

```
ParameterBound ranges
        │
        ▼
CoilDesigner / StructuralDesigner  (CadQuery geometry)
        │ STEP / STL
        ▼
FreeCADBridge  ──────────────────────── CalculiX (structural FEM)
        │                               Elmer    (magnetostatic FEM)
        ▼
MagneticAnalyzer  (Biot-Savart fast field computation)
        │
        ▼
ParametricOptimizer  (differential evolution / Bayesian / LHS)
        │  Pareto front
        ▼
Best design STEP export + FieldResult + FEMResult
```

---

## 1. Parametric Designer (`cad/parametric_designer.py`)

### Coil geometries

```python
from cad.parametric_designer import CoilDesigner, CoilParams

designer = CoilDesigner()

params = CoilParams(
    radius=40.0,          # mm
    height=80.0,          # mm
    turns=200,
    wire_diameter=1.0,    # mm
    core_radius=10.0,     # mm (air core bore)
    layer_count=2,
)

former = designer.solenoid_former(params)
designer.export_step(former, "output/solenoid.step")
designer.export_stl(former,  "output/solenoid.stl", tolerance=0.01)
```

### Structural formers

```python
from cad.parametric_designer import StructuralDesigner, StructuralParams

sdesigner = StructuralDesigner()   # StructuralDesigner is in CoilDesigner
params = StructuralParams(
    outer_radius=45.0, inner_radius=40.0,
    height=80.0, wall_thickness=5.0,
    n_ribs=6, rib_height=3.0
)
# Use CoilDesigner.hollow_cylinder()
```

### Parameter grid generation

```python
from cad.parametric_designer import generate_parameter_grid

grid = generate_parameter_grid(
    {"radius": (10.0, 100.0), "turns": (50, 500), "length": (20.0, 200.0)},
    n_samples=50,
    method="latin_hypercube",
)
```

---

## 2. Magnetic Analyzer (`cad/magnetic_analyzer.py`)

### Biot-Savart field computation

```python
from cad.magnetic_analyzer import MagneticAnalyzer, solenoid_path, helmholtz_path

analyzer = MagneticAnalyzer()

# Build coil path (SI units)
coil = solenoid_path(
    radius_m=0.04,
    length_m=0.08,
    turns=200,
    current=2.0,           # amperes
    n_points_per_turn=72,  # resolution
)

# Evaluate on axial line
z, Bz = analyzer.axial_field_profile(coil, -0.06, 0.06, n_points=100)

# Evaluate on volume grid
grid = MagneticAnalyzer.volume_grid(r_max=0.02, z_half=0.04, n=10)
result = analyzer.compute_field(coil, grid)

print(f"Peak field:  {result.peak_field()*1e3:.2f} mT")
print(f"Mean field:  {result.mean_field()*1e3:.2f} mT")
print(f"Uniformity:  {result.uniformity()*100:.1f}%")
```

### Efficiency metrics (for optimizer)

```python
metrics = analyzer.efficiency_metrics(
    coil,
    target_region_points=grid,
    wire_diameter_m=1e-3,
)
# Keys: mean_field_T, peak_field_T, uniformity,
#       wire_length_m, resistance_ohm, power_W,
#       field_per_watt, field_per_kg_wire
```

### Flux through a surface

```python
import numpy as np
# 10×10 grid over coil cross-section at z=0
surface = MagneticAnalyzer.midplane_grid(r_max=0.04, n_radial=10, n_angular=36)
flux = analyzer.flux_through_surface(
    coil, surface, normal=[0, 0, 1], area_element=1e-5
)
print(f"Flux: {flux*1e3:.4f} mWb")
```

---

## 3. FreeCAD Bridge (`cad/freecad_bridge.py`)

### Structural FEM

```python
from cad.freecad_bridge import FreeCADBridge, Material, BoundaryCondition

bridge = FreeCADBridge()

result = bridge.run_structural(
    step_file="output/solenoid.step",
    material=Material.aluminium(),
    boundary_conditions=[
        BoundaryCondition(kind="fixed", face_index=0),
        BoundaryCondition(kind="force", face_index=2,
                          value=1000.0, direction=(0, 0, -1)),
    ],
    mesh_size_mm=3.0,
)

print(result.summary())
print(f"Safety factor: {result.safety_factor(Material.aluminium().yield_strength_pa):.2f}")
```

### Magnetostatic FEM (requires Elmer)

```python
result = bridge.run_magnetostatic(
    step_file="output/core.step",
    material=Material.steel(),         # high permeability
    current_density_a_m2=5e6,
    mesh_size_mm=2.0,
)
```

### Available materials

| Preset | E (GPa) | ρ (kg/m³) | σ_y (MPa) | μᵣ |
|--------|---------|-----------|-----------|-----|
| `Material.steel()` | 210 | 7850 | 250 | 100 |
| `Material.aluminium()` | 69 | 2700 | 270 | 1 |
| `Material.copper()` | 110 | 8960 | 210 | 1 |
| `Material.pla()` | 3.5 | 1240 | 50 | 1 |

---

## 4. Multi-Objective Optimizer (`utils/optimizer.py`)

### Quick coil optimization

```python
from utils.optimizer import CoilOptimizer

opt = CoilOptimizer(
    target_z_half_m=0.04,
    target_radius_m=0.02,
    wire_diameter_m=1e-3,
    current_a=2.0,
)
result = opt.run(n_iterations=300, method="differential_evolution")
print(result.summary())
```

### Custom multi-objective optimization

```python
from utils.optimizer import ParametricOptimizer, ParameterBound, Objective

opt = ParametricOptimizer(
    bounds=[
        ParameterBound("radius_mm", 10.0, 100.0),
        ParameterBound("turns",     50,   500, integer=True),
        ParameterBound("wall_mm",   1.0,  10.0),
    ],
    objectives=[
        Objective("flux_Wb",      my_flux_fn,    minimize=False),
        Objective("mass_kg",      my_mass_fn,    minimize=True),
        Objective("max_stress_Pa",my_stress_fn,  minimize=True),
    ],
)
result = opt.run(n_iterations=500, method="bayesian")  # requires: pip install optuna
```

### Optimization methods

| Method | Speed | Quality | Extra deps |
|--------|-------|---------|------------|
| `differential_evolution` | Medium | High | none |
| `bayesian` | Medium | Highest | `pip install optuna` |
| `latin_hypercube` | Fast | Low | none |

### Reading the Pareto front

```python
for design in result.pareto_front:
    print(design.params)           # dict of parameter values
    print(design.objective_values) # dict of raw objective values
```

---

## Installation

```bash
pip install cadquery                # parametric CAD
pip install scipy numpy             # optimizer core
pip install optuna                  # optional: Bayesian search
pip install meshio                  # optional: parse Elmer FEM output
# FreeCAD: install system package (not pip)
# Ubuntu: sudo apt install freecad
# or download from https://www.freecad.org
```
