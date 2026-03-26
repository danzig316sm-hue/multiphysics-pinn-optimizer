# Physics Equations & Constraints

## Governing Physics Domains

### 1. Thermal Domain

Steady-state heat conduction (Laplace form):

```
∇·(k ∇T) + Q = 0
```

Where:
- `T` — temperature field
- `k` — thermal conductivity tensor
- `Q` — volumetric heat source

In the PINN the network learns `T(x; θ)` such that residuals of this PDE
are minimized alongside data fidelity.

### 2. Structural Domain (Stress)

Linear elasticity equilibrium:

```
∇·σ + f = 0
σ = C : ε
ε = ½(∇u + (∇u)ᵀ)
```

Where:
- `σ` — Cauchy stress tensor
- `ε` — infinitesimal strain tensor
- `u` — displacement field
- `C` — fourth-order elasticity tensor
- `f` — body force

Physical constraint: stress components must satisfy `σ ≥ 0` (compressive
material assumption). Implemented as a ReLU penalty on negative predictions.

### 3. Electromagnetic Domain

Time-harmonic Maxwell's equations (frequency domain):

```
∇ × H = J + jωεE
∇ × E = −jωμH
∇·(εE) = ρ
∇·(μH) = 0
```

Where:
- `E`, `H` — electric and magnetic field vectors
- `ε` — permittivity, `μ` — permeability
- `ρ` — charge density, `J` — current density
- `ω` — angular frequency

EM field smoothness (spatial continuity) is enforced as a finite-difference
penalty on consecutive predictions.

---

## Implemented Physics Constraints

Three constraints are currently active in `PINNModel.physics_loss()`:

### C1 — Energy Conservation

```python
physics_constraint_1 = mean(|thermal_pred + 0.1 × em_pred|)
```

Represents coupling between thermal and electromagnetic energy: the EM
field contributes Joule heating to the thermal balance. The factor 0.1
is a domain-coupling coefficient (tune per application).

**Target**: drive this to zero, meaning thermal and EM fields balance.

### C2 — Stress Non-Negativity

```python
physics_constraint_2 = mean(ReLU(-stress_pred))
```

Penalizes any predicted stress below zero. For compressive materials
stress should be non-negative in the sign convention used here.

**Target**: zero, meaning all predicted stresses satisfy `σ ≥ 0`.

### C3 — EM Smoothness

```python
physics_constraint_3 = mean(|diff(em_pred, dim=0)|)
```

First-order finite difference across the batch spatial dimension.
Enforces that the EM field varies smoothly without unphysical jumps.

**Target**: small value, indicating spatially smooth EM predictions.

---

## Combined Loss Function

```
L_total = L_data + w_phys × L_phys

L_data  = (MSE(thermal) + MSE(stress) + MSE(em)) / 3

L_phys  = (C1 + C2 + C3) / 3

w_phys  ∈ [physics_weight_init, physics_weight_max]   ← adapted by SelfCorrectionLoop
```

The self-correcting loop monitors C1, C2, C3 individually and boosts
`w_phys` when any constraint's moving-average residual exceeds the
configured tolerance (default 0.05).

---

## Adding New Physics Constraints

1. Compute the scalar residual in `PINNModel.physics_loss()`.
2. Add it to the `physics_loss` average.
3. Add its name to `PhysicsResidualTracker.CONSTRAINT_NAMES` in `utils/self_correction.py`.
4. Mirror the same computation in `SelfCorrectionLoop._extract_residuals()`.
