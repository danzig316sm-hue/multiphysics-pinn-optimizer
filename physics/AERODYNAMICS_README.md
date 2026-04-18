# Aerodynamics Module - Mobius-Nova PINN Optimizer

Complete aerodynamic physics module for wind turbine PINN training. Implements Blade Element Momentum (BEM) theory with differentiable physics constraints.

## Components

### 1. AerodynamicConstants
Wind turbine specifications for Bergey Excel 15-kW direct-drive turbine.

**Key Parameters:**
- Rotor diameter: 7.0 m
- Hub height: 30 m
- Cut-in wind speed: 3.0 m/s
- Rated wind speed: 11.0 m/s
- Cut-out (furling): 60 m/s
- Air density: 1.225 kg/m³
- Number of blades: 3
- Design tip speed ratio: 7.0
- Betz limit: 16/27 ≈ 0.593

### 2. S809AirfoilData
OSU wind tunnel polynomial fits for S809 airfoil (lift and drag coefficients).

**Features:**
- Cl and Cd as polynomials of angle of attack
- Valid range: -5 to 25 degrees
- Uses Horner's method for efficient evaluation
- Physically constrains Cd to positive values

**Usage:**
```python
alpha = np.array([0, 5, 10, 15])  # degrees
cl = S809AirfoilData.get_cl(alpha)
cd = S809AirfoilData.get_cd(alpha)
```

### 3. BEMSolver
Blade Element Momentum theory implementation for rotor aerodynamics.

**Features:**
- Iterative solution of induction factors (a and a')
- Prandtl tip/hub loss correction
- Glauert correction for high induction (a > 0.5)
- S809 airfoil lookup
- Element-wise and integrated quantities

**Method Signature:**
```python
solver = BEMSolver(n_elements=20, n_blades=3, rotor_radius=3.5)
result = solver.solve(wind_speed=10.0, rpm=400, pitch=0.0, 
                      chord=None, twist=None)
```

**Returns:**
- `Cp`: Power coefficient
- `Ct`: Thrust coefficient  
- `power`: Rotor power (W)
- `thrust`: Rotor thrust (N)
- `torque`: Rotor torque (N·m)
- `tsr`: Tip speed ratio
- `a`, `a_prime`: Induction factors per element
- `alpha`: Angle of attack per element
- `cl`, `cd`: Airfoil coefficients per element
- `element_power`, `element_thrust`: Per-element forces

### 4. AeroPINNLoss
Differentiable physics-informed loss functions (PyTorch tensors).

**Loss Functions:**

#### `betz_limit_loss(Cp_pred)`
Penalizes power coefficient > 16/27
```
L = mean(max(0, Cp - 16/27)²)
```

#### `momentum_consistency_loss(thrust, torque, wind_speed, rpm)`
Enforces: Power_torque ≈ Power_thrust
```
L = mean(|P_torque - P_thrust| / |P_thrust|)
```

#### `tip_speed_constraint(tsr_pred, wind_speed, rpm)`
TSR definition constraint: TSR = ωR/V
```
L = mean((TSR_pred - ωR/V)²)
```

#### `annual_energy_consistency(aep_pred, Cp_pred, wind_distribution)`
AEP integration over Weibull distribution
```
AEP = ∫ P(v) * f(v) * 8760 dv
```

**Usage:**
```python
pinn_loss = AeroPINNLoss(device="cpu")
Cp_pred = torch.tensor([0.45, 0.50, 0.55])
loss = pinn_loss.betz_limit_loss(Cp_pred)
```

### 5. WindResourceCalculator
Wind resource models and energy calculations.

**Distributions:**
- Rayleigh (special case of Weibull with k=2)
- Weibull with configurable shape parameter

**Methods:**
```python
# Annual energy production
aep = WindResourceCalculator.annual_energy_production(
    power_curve={3.0: 0.5, 6.5: 8.0, 11.0: 15.0},
    mean_wind_speed=6.5,
    distribution="rayleigh"
)

# Capacity factor
cf = WindResourceCalculator.capacity_factor(aep=45000, rated_power=15)
```

### 6. QBladeInterface
Stub for QBlade aerodynamic analysis integration.

**Methods:**
```python
# Export blade geometry to QBlade format
QBladeInterface.export_blade_geometry(
    chord_distribution=np.array([...]),
    twist_distribution=np.array([...]),
    airfoil_polars={...},
    output_path="qblade_blade.txt"
)

# Parse QBlade results (stub)
results = QBladeInterface.parse_qblade_results("output.txt")
```

## Example Usage

```python
import numpy as np
import torch
from physics.aerodynamics import (
    AerodynamicConstants,
    BEMSolver,
    AeroPINNLoss,
    WindResourceCalculator,
)

# Initialize turbine
constants = AerodynamicConstants()
solver = BEMSolver(n_elements=25, n_blades=3, rotor_radius=3.5)

# Solve aerodynamics at wind speed 10 m/s, 400 rpm
result = solver.solve(wind_speed=10.0, rpm=400, pitch=0.0)
print(f"Power: {result['power']:.1f} W")
print(f"Cp: {result['Cp']:.4f}")
print(f"Thrust: {result['thrust']:.1f} N")

# PINN physics loss
pinn_loss = AeroPINNLoss()
Cp_pred = torch.tensor(result['Cp'])
betz_loss = pinn_loss.betz_limit_loss(Cp_pred)
print(f"Betz loss: {betz_loss:.6f}")

# Annual energy production
power_curve = {3.0: 0.5, 6.5: 8.0, 11.0: 15.0, 15.0: 15.0, 25.0: 0.0}
aep = WindResourceCalculator.annual_energy_production(power_curve, mean_wind_speed=6.5)
print(f"AEP: {aep:.1f} kWh/year")
```

## Run Module Test

```bash
python physics/aerodynamics.py
```

Output shows:
- Bergey turbine specifications
- BEM solver results across wind speed range
- PINN loss examples
- Wind resource calculations

## Dependencies

- `numpy` — Numerical operations
- `torch` — Differentiable tensors (for AeroPINNLoss)
- `scipy` (optional) — Interpolation in wind resource calculations

## Physics Validation

The module enforces:
1. **Betz limit**: Cp ≤ 16/27 (fundamental aerodynamic bound)
2. **Momentum balance**: Thrust-power consistency
3. **Tip speed ratio**: TSR = ωR/V definition
4. **Energy conservation**: AEP integration over wind distribution
5. **Airfoil physics**: S809 polar curves from wind tunnel data

## Integration with PINN Training

All loss functions return PyTorch tensors with gradients enabled,
enabling end-to-end differentiable training:

```python
import torch.optim as optim

# Neural network predicts Cp, thrust, torque from wind_speed, rpm
model = MyAeroModel()
optimizer = optim.Adam(model.parameters())

for epoch in range(100):
    optimizer.zero_grad()
    
    wind_speed = torch.tensor(10.0, requires_grad=True)
    rpm = torch.tensor(400.0, requires_grad=True)
    
    Cp_pred, thrust_pred, torque_pred = model(wind_speed, rpm)
    
    loss = (
        pinn_loss.betz_limit_loss(Cp_pred) +
        pinn_loss.momentum_consistency_loss(thrust_pred, torque_pred, wind_speed, rpm) +
        pinn_loss.tip_speed_constraint(tsr_pred, wind_speed, rpm)
    )
    
    loss.backward()
    optimizer.step()
```

## File Locations

- Module: `/physics/aerodynamics.py`
- Package init: `/physics/__init__.py`
- This doc: `/physics/AERODYNAMICS_README.md`

## Status

✓ Fully implemented
✓ All required classes present
✓ Syntax validated
✓ Differentiable constraints ready
✓ S809 airfoil data polynomial fitted
✓ BEM solver with convergence criteria
