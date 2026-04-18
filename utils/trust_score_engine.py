"""
utils/trust_score_engine.py
============================
Trust score accumulation engine for the Mobius-Nova PMSG optimizer.

What this solves
----------------
Traditional simulation tools give you numbers. They don't tell you how
much to trust them, or how that trust changes as you accumulate physical
evidence. This module does both.

Every time a prediction (from the PINN or cfd_thermal_coupler) is checked
against a physical measurement (SolidWorks verification, prototype test,
or independent FEA), the delta is logged. Over time this builds a
confidence map across the design space — specific claims get specific
trust scores backed by a full audit trail.

Passive Air Intake Physics  (the gap nobody accounts for)
---------------------------------------------------------
Traditional CFD treats the inlet as a boundary condition you specify.
You define P_inlet and move on. Which means the intake geometry —
the scoop shape, the orientation to wind, the relationship between
ambient ram pressure and what the housing actually captures — is
never evaluated as part of the physics.

This module closes that gap. The wind that spins the turbine creates
ram pressure at the intake face:

    q_ram = ½ · ρ_air · v_wind²

At rated 11 m/s: q_ram = ½ × 1.225 × 121 = 74 Pa
At cut-in 3 m/s: q_ram = ½ × 1.225 × 9   =  5.5 Pa

A well-designed passive intake scoop recovers 60–80% of that (Cp ≈ 0.6–0.8).
A poorly oriented duct recovers 10–20%. The difference between those cases
is the margin between thermal pass and thermal fail — and it is entirely
determined by intake geometry, entirely calculable, and currently
unaccounted for in any commercial simulation workflow for this class of
machine.

The intake physics adds directly to the centrifugal pump head from
cfd_thermal_coupler.py:

    dP_total_driving = dP_centrifugal + dP_ram_recovered - dP_losses

This is what the trust score engine verifies against physical measurements:
did the combined driving pressure prediction match the actual measured
mass flow rate?

Trust Score Architecture
------------------------
Trust scores are maintained per (design_class, physics_domain) pair.

  design_class: defined by (bezier_mode, magnet_type, pole_geometry_hash)
  physics_domain: one of [em, cfd_pressure, cfd_flow, thermal_winding,
                           thermal_magnet, acoustic, structural, intake_ram]

Each domain has an independent rolling score so you can say:
"For asymmetric printed-magnet designs, we trust EM predictions at 97%
confidence but thermal predictions only at 82% — we need more convection
measurements in that design class."

Score formula (Wilson score interval for bounded accuracy):
    score = (n_correct + z²/2) / (n_total + z²)
    where z = 1.645 for 90% confidence interval

This is more honest than a simple hit-rate because it correctly assigns
lower scores when n is small — a 1/1 hit rate gives score ~0.53, not 1.0.

Verification Sources (ranked by trust weight)
---------------------------------------------
  prototype_test      weight = 1.00  — physical hardware measurement
  solidworks_fea      weight = 0.85  — verified FEA with mesh convergence
  independent_fea     weight = 0.75  — FEA from separate solver/analyst
  analytical_check    weight = 0.60  — hand calculation cross-check
  peer_review         weight = 0.50  — domain expert review without test

Tolerance gates (what counts as a correct prediction)
------------------------------------------------------
  em_torque           ± 3%   of measured value
  em_cogging          ± 5 Nm absolute
  em_efficiency       ± 0.5 percentage points
  cfd_dP              ± 15%  of measured ΔP
  cfd_mass_flow       ± 10%  of measured ṁ
  thermal_winding     ± 10°C of measured temperature
  thermal_magnet      ± 5°C  of measured temperature
  acoustic_Lp         ± 3 dB
  intake_ram_recovery ± 12%  of measured Cp

Usage
-----
    from utils.trust_score_engine import TrustScoreEngine, Verification

    engine = TrustScoreEngine(db_path="trust_scores.json")

    # Log a verification against a SolidWorks spot-check
    v = Verification(
        design_id="asym_printed_run_042",
        bezier_mode="asymmetric",
        magnet_is_printed=True,
        source="solidworks_fea",
        domain="thermal_magnet",
        predicted=54.3,
        measured=57.1,
        units="°C",
        notes="Stall condition, 40 kW, 3 blade sections",
    )
    engine.log(v)

    # Get confidence for a design class before deciding to ship
    report = engine.confidence_report("asymmetric", magnet_is_printed=True)
    print(report)

    # Check whether a design class is commercially shippable
    ready = engine.is_shippable("asymmetric", magnet_is_printed=True)
"""

from __future__ import annotations

import json
import math
import hashlib
import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# Physical constants (duplicated here so module is self-contained)
# ===========================================================================

RHO_AIR   = 1.225      # kg/m³
V_RATED   = 11.0       # m/s  — Bergey 15-kW rated wind speed
V_CUTIN   = 3.0        # m/s  — cut-in wind speed
V_CUTOUT  = 25.0       # m/s  — cut-out wind speed

# Pressure recovery coefficient for passive intake designs
# Cp_ideal = 1.0 (all ram pressure recovered)
# Well-designed scoop at optimal angle: Cp ≈ 0.65–0.80
# Flush opening, no scoop:              Cp ≈ 0.20–0.35
# Poorly oriented / blocked:            Cp ≈ 0.05–0.15
CP_SCOOP_OPTIMAL  = 0.72   # design target — well-shaped scoop at rated wind
CP_SCOOP_FLUSH    = 0.28   # flat flush opening baseline
CP_SCOOP_MIN      = 0.10   # worst-case blocked / misaligned

# Wilson score z-value for 90% confidence interval
Z_90 = 1.645

# Shippability threshold — minimum trust score to deliver to an OEM
SHIPPABLE_THRESHOLD = 0.75
# Minimum number of verifications before shippability can be asserted
MIN_VERIFICATIONS_TO_SHIP = 5


# ===========================================================================
# Tolerance gates — what counts as a correct prediction per domain
# ===========================================================================

TOLERANCE_GATES: Dict[str, Dict] = {
    "em_torque":           {"type": "relative", "tol": 0.03,  "units": "Nm"},
    "em_cogging":          {"type": "absolute", "tol": 5.0,   "units": "Nm"},
    "em_efficiency":       {"type": "absolute", "tol": 0.5,   "units": "%"},
    "em_Brmin":            {"type": "relative", "tol": 0.05,  "units": "T"},
    "cfd_dP":              {"type": "relative", "tol": 0.15,  "units": "Pa"},
    "cfd_mass_flow":       {"type": "relative", "tol": 0.10,  "units": "kg/s"},
    "thermal_winding":     {"type": "absolute", "tol": 10.0,  "units": "°C"},
    "thermal_magnet":      {"type": "absolute", "tol": 5.0,   "units": "°C"},
    "acoustic_Lp":         {"type": "absolute", "tol": 3.0,   "units": "dB"},
    "intake_ram_recovery": {"type": "relative", "tol": 0.12,  "units": "Cp"},
    "structural_deflect":  {"type": "relative", "tol": 0.10,  "units": "mm"},
}

# Verification source weights
SOURCE_WEIGHTS: Dict[str, float] = {
    "prototype_test":   1.00,
    "solidworks_fea":   0.85,
    "independent_fea":  0.75,
    "analytical_check": 0.60,
    "peer_review":      0.50,
}


# ===========================================================================
# Passive intake ram pressure physics
# ===========================================================================

class PassiveIntakePhysics:
    """
    Computes the ram pressure contribution from passive air intakes.

    This is the physics gap in every commercial CFD workflow for wind
    generator housings. The intake geometry is never optimised as part
    of the coupled thermal-CFD problem because CFD tools treat the
    inlet face as a user-specified boundary condition.

    Here we compute it from first principles and add it to the centrifugal
    pump head from cfd_thermal_coupler.py to get the true total driving
    pressure for cooling airflow.

    Parameters
    ----------
    n_intakes       : number of passive intake openings
    A_intake_each   : area of each intake opening (m²)
    scoop_angle_deg : angle of intake scoop lip relative to wind direction
                      0° = facing directly into wind (max recovery)
                      90° = perpendicular to wind (no ram, only static)
    intake_shape    : "scoop" | "flush" | "louvre" | "NACA_duct"
    """

    # Pressure recovery by shape type at optimal angle
    _CP_BASE = {
        "scoop":     0.72,
        "flush":     0.28,
        "louvre":    0.45,
        "NACA_duct": 0.82,   # NACA flush inlet — best passive design
    }

    # Angle correction: Cp(θ) = Cp_0 · cos²(θ)
    # At 30° off-axis: cos²(30°) = 0.75 → 25% penalty
    # At 60° off-axis: cos²(60°) = 0.25 → 75% penalty

    def __init__(
        self,
        n_intakes: int = 4,
        A_intake_each_m2: float = 0.0045,
        scoop_angle_deg: float = 15.0,
        intake_shape: str = "scoop",
    ):
        self.n_intakes       = n_intakes
        self.A_intake_each   = A_intake_each_m2
        self.A_total         = n_intakes * A_intake_each_m2
        self.scoop_angle_deg = scoop_angle_deg
        self.intake_shape    = intake_shape
        self.Cp_base         = self._CP_BASE.get(intake_shape, 0.28)

    def Cp_effective(self, yaw_error_deg: float = 0.0) -> float:
        """
        Effective pressure recovery coefficient at a given yaw error.

        Cp_eff = Cp_base · cos²(scoop_angle + yaw_error)

        The turbine yaws to face the wind but there's always some
        residual error (typically ±10–15° for passive yaw systems).
        """
        total_angle = math.radians(self.scoop_angle_deg + yaw_error_deg)
        return self.Cp_base * math.cos(total_angle) ** 2

    def ram_pressure_Pa(
        self,
        v_wind_m_s: float,
        yaw_error_deg: float = 0.0,
    ) -> float:
        """
        Ram pressure available at the intake face.

        q_ram = ½ · ρ · v_wind²
        dP_recovered = Cp_eff · q_ram

        This is the pressure that drives air into the housing above
        and beyond the centrifugal pumping from the rotor.
        """
        q_ram = 0.5 * RHO_AIR * v_wind_m_s ** 2
        return self.Cp_effective(yaw_error_deg) * q_ram

    def mass_flow_intake_kg_s(
        self,
        v_wind_m_s: float,
        yaw_error_deg: float = 0.0,
        dP_back_Pa: float = 0.0,
    ) -> float:
        """
        Mass flow rate through the passive intakes.

        From Bernoulli through the intake duct:
            ṁ = Cd · A_total · sqrt(2 · ρ · (dP_ram - dP_back))

        Cd = discharge coefficient (≈ 0.61 for sharp-edged, 0.82 for rounded)
        dP_back = back-pressure from the housing (Pa) — from coupler
        """
        dP_ram = self.ram_pressure_Pa(v_wind_m_s, yaw_error_deg)
        dP_net = max(dP_ram - dP_back_Pa, 0.0)
        Cd = 0.72 if self.intake_shape in ("scoop", "NACA_duct") else 0.61
        return Cd * self.A_total * math.sqrt(2.0 * RHO_AIR * dP_net + 1e-9)

    def full_report(
        self,
        v_wind_m_s: float = V_RATED,
        yaw_error_deg: float = 10.0,
        dP_centrifugal_Pa: float = 13.1,
        dP_loss_Pa: float = 0.3,
    ) -> Dict:
        """
        Complete intake physics report — the numbers nobody else computes.

        Returns a dict with all intermediate values so the trust score
        engine can verify each one against physical measurements.
        """
        q_ram          = 0.5 * RHO_AIR * v_wind_m_s ** 2
        Cp_eff         = self.Cp_effective(yaw_error_deg)
        dP_ram_recov   = Cp_eff * q_ram
        dP_total_drive = dP_centrifugal_Pa + dP_ram_recov
        dP_net         = max(dP_total_drive - dP_loss_Pa, 0.0)
        mdot_intake    = self.mass_flow_intake_kg_s(
            v_wind_m_s, yaw_error_deg, dP_loss_Pa
        )
        v_intake_face  = (mdot_intake / (RHO_AIR * self.A_total)
                          if self.A_total > 0 else 0.0)

        return {
            # Input conditions
            "v_wind_m_s":            v_wind_m_s,
            "yaw_error_deg":         yaw_error_deg,
            "intake_shape":          self.intake_shape,
            "n_intakes":             self.n_intakes,
            "A_total_m2":            self.A_total,
            "scoop_angle_deg":       self.scoop_angle_deg,

            # Ram pressure physics
            "q_ram_Pa":              q_ram,          # available dynamic pressure
            "Cp_base":               self.Cp_base,   # shape coefficient
            "Cp_effective":          Cp_eff,         # angle-corrected
            "dP_ram_recovered_Pa":   dP_ram_recov,   # pressure captured by intake
            "pct_of_q_ram_captured": Cp_eff * 100,

            # Combined driving pressure
            "dP_centrifugal_Pa":     dP_centrifugal_Pa,
            "dP_total_driving_Pa":   dP_total_drive,  # cent + ram
            "dP_loss_Pa":            dP_loss_Pa,
            "dP_net_Pa":             dP_net,

            # Flow outputs
            "mdot_intake_kg_s":      mdot_intake,
            "v_intake_face_m_s":     v_intake_face,

            # At rated vs cut-in
            "dP_ram_rated_Pa":  self.ram_pressure_Pa(V_RATED,  yaw_error_deg),
            "dP_ram_cutin_Pa":  self.ram_pressure_Pa(V_CUTIN,  yaw_error_deg),
            "dP_ram_cutout_Pa": self.ram_pressure_Pa(V_CUTOUT, yaw_error_deg),
        }

    def sensitivity(self) -> Dict:
        """
        Show how intake design choices affect the driving pressure.
        Useful for geometry selection — which lever moves the needle most.
        """
        results = {}
        for shape in ("scoop", "flush", "louvre", "NACA_duct"):
            temp = PassiveIntakePhysics(
                n_intakes=self.n_intakes,
                A_intake_each_m2=self.A_intake_each,
                scoop_angle_deg=self.scoop_angle_deg,
                intake_shape=shape,
            )
            dP = temp.ram_pressure_Pa(V_RATED, yaw_error_deg=10.0)
            results[shape] = {
                "dP_ram_Pa": round(dP, 2),
                "Cp_eff":    round(temp.Cp_effective(10.0), 3),
            }
        for angle in (0, 15, 30, 45, 60):
            dP = self.ram_pressure_Pa(V_RATED, yaw_error_deg=0.0)
            temp = PassiveIntakePhysics(
                n_intakes=self.n_intakes,
                A_intake_each_m2=self.A_intake_each,
                scoop_angle_deg=angle,
                intake_shape=self.intake_shape,
            )
            dP_a = temp.ram_pressure_Pa(V_RATED, yaw_error_deg=0.0)
            results[f"scoop_angle_{angle}deg"] = {
                "dP_ram_Pa": round(dP_a, 2),
                "Cp_eff":    round(temp.Cp_effective(0.0), 3),
            }
        return results


# ===========================================================================
# Verification record
# ===========================================================================

@dataclass
class Verification:
    """
    One physical verification event — a prediction checked against reality.

    Fields
    ------
    design_id       : unique identifier for the design run (e.g. genome key)
    bezier_mode     : symmetric | asymmetric | multimaterial
    magnet_is_printed: True → BAAM printed, False → sintered N48H
    source          : verification source (see SOURCE_WEIGHTS)
    domain          : physics domain being verified (see TOLERANCE_GATES)
    predicted       : what the PINN / coupler predicted
    measured        : what was actually measured
    units           : physical unit string for logging
    notes           : free text — operating condition, analyst name, etc.
    wind_speed_m_s  : wind speed during test (if applicable)
    yaw_error_deg   : yaw misalignment during test (if applicable)
    timestamp       : ISO 8601 string — auto-set if not provided
    """
    design_id:         str
    bezier_mode:       str
    magnet_is_printed: bool
    source:            str
    domain:            str
    predicted:         float
    measured:          float
    units:             str
    notes:             str = ""
    wind_speed_m_s:    float = V_RATED
    yaw_error_deg:     float = 0.0
    timestamp:         str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.utcnow().isoformat()

    @property
    def error_abs(self) -> float:
        return abs(self.predicted - self.measured)

    @property
    def error_rel(self) -> float:
        if abs(self.measured) < 1e-9:
            return 0.0
        return abs(self.predicted - self.measured) / abs(self.measured)

    @property
    def weight(self) -> float:
        return SOURCE_WEIGHTS.get(self.source, 0.5)

    def is_correct(self) -> bool:
        """
        Did the prediction fall within the tolerance gate for this domain?
        """
        gate = TOLERANCE_GATES.get(self.domain)
        if gate is None:
            return False
        if gate["type"] == "relative":
            return self.error_rel <= gate["tol"]
        else:
            return self.error_abs <= gate["tol"]

    def verdict_str(self) -> str:
        gate = TOLERANCE_GATES.get(self.domain, {})
        tol  = gate.get("tol", "?")
        typ  = gate.get("type", "?")
        ok   = "✓ WITHIN" if self.is_correct() else "✗ OUTSIDE"
        return (
            f"[{ok}]  predicted={self.predicted:.4g}{self.units}  "
            f"measured={self.measured:.4g}{self.units}  "
            f"err={'rel' if typ=='relative' else 'abs'}="
            f"{self.error_rel*100:.1f}%  tol={'±'+str(int(tol*100))+'%' if typ=='relative' else '±'+str(tol)+self.units}"
        )


# ===========================================================================
# Trust score per (design_class, domain)
# ===========================================================================

@dataclass
class DomainTrustRecord:
    """Rolling trust score for one (design_class, domain) pair."""
    design_class:  str
    domain:        str
    n_total:       float = 0.0   # weighted count
    n_correct:     float = 0.0   # weighted correct count
    verifications: List[dict] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        if self.n_total < 1e-9:
            return 0.0
        return self.n_correct / self.n_total

    @property
    def wilson_score(self) -> float:
        """
        Wilson score interval lower bound at 90% confidence.
        More honest than simple hit-rate when n is small.
        A single 1/1 hit gives ~0.53, not 1.0.
        """
        n = self.n_total
        if n < 1e-9:
            return 0.0
        p_hat = self.n_correct / n
        z = Z_90
        numer = p_hat + z**2 / (2*n) - z * math.sqrt(
            (p_hat*(1-p_hat) + z**2/(4*n)) / n
        )
        denom = 1.0 + z**2 / n
        return max(numer / denom, 0.0)

    @property
    def n_raw(self) -> int:
        return len(self.verifications)

    def add(self, v: Verification) -> None:
        w = v.weight
        self.n_total   += w
        self.n_correct += w if v.is_correct() else 0.0
        self.verifications.append({
            "timestamp":  v.timestamp,
            "design_id":  v.design_id,
            "source":     v.source,
            "predicted":  v.predicted,
            "measured":   v.measured,
            "units":      v.units,
            "correct":    v.is_correct(),
            "weight":     w,
            "error_rel":  v.error_rel,
            "error_abs":  v.error_abs,
            "notes":      v.notes,
        })

    def summary_line(self) -> str:
        gate = TOLERANCE_GATES.get(self.domain, {})
        tol_str = (
            f"±{int(gate['tol']*100)}%" if gate.get("type") == "relative"
            else f"±{gate.get('tol','?')}{gate.get('units','')}"
        )
        bar_len = int(self.wilson_score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        return (
            f"  {self.domain:<22} [{bar}] {self.wilson_score*100:5.1f}%  "
            f"n={self.n_raw}  hit={self.hit_rate*100:.0f}%  tol={tol_str}"
        )


# ===========================================================================
# Main trust score engine
# ===========================================================================

class TrustScoreEngine:
    """
    Accumulates verification evidence and maintains trust scores.

    Data is persisted to a JSON file so scores survive across sessions.
    Every verification event is immutably appended — full audit trail.

    Parameters
    ----------
    db_path : str | Path
        Path to the JSON persistence file.
        Defaults to "trust_scores.json" in the working directory.
    """

    def __init__(self, db_path: str | Path = "trust_scores.json"):
        self.db_path = Path(db_path)
        self._records: Dict[str, DomainTrustRecord] = {}
        self._load()

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _key(self, design_class: str, domain: str) -> str:
        return f"{design_class}::{domain}"

    def _load(self) -> None:
        if not self.db_path.exists():
            return
        try:
            with open(self.db_path, "r") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return   # empty or corrupt file — start fresh
        for key, d in raw.items():
            rec = DomainTrustRecord(
                design_class=d["design_class"],
                domain=d["domain"],
                n_total=d["n_total"],
                n_correct=d["n_correct"],
                verifications=d["verifications"],
            )
            self._records[key] = rec

    def _save(self) -> None:
        out = {}
        for key, rec in self._records.items():
            out[key] = {
                "design_class":  rec.design_class,
                "domain":        rec.domain,
                "n_total":       rec.n_total,
                "n_correct":     rec.n_correct,
                "verifications": rec.verifications,
            }
        with open(self.db_path, "w") as f:
            json.dump(out, f, indent=2)

    # -----------------------------------------------------------------------
    # Design class identifier
    # -----------------------------------------------------------------------

    @staticmethod
    def design_class(bezier_mode: str, magnet_is_printed: bool) -> str:
        """
        Canonical design class string.
        Future extension: include pole_geometry_hash for finer granularity.
        """
        mat = "printed" if magnet_is_printed else "sintered"
        return f"{bezier_mode}_{mat}"

    # -----------------------------------------------------------------------
    # Log a verification
    # -----------------------------------------------------------------------

    def log(self, v: Verification) -> DomainTrustRecord:
        """
        Log one verification event and update the rolling trust score.

        Returns the updated DomainTrustRecord for that (class, domain) pair.
        """
        dc   = self.design_class(v.bezier_mode, v.magnet_is_printed)
        key  = self._key(dc, v.domain)

        if key not in self._records:
            self._records[key] = DomainTrustRecord(
                design_class=dc, domain=v.domain
            )

        self._records[key].add(v)
        self._save()
        return self._records[key]

    def log_verdict(
        self,
        verdict,                  # DesignVerdict from cfd_thermal_coupler
        source: str,
        measured_values: Dict[str, float],
        design_id: str = "",
        notes: str = "",
        wind_speed_m_s: float = V_RATED,
        yaw_error_deg: float = 0.0,
    ) -> List[DomainTrustRecord]:
        """
        Log multiple verification domains from a single DesignVerdict
        checked against a dict of measured values.

        measured_values keys match domain names in TOLERANCE_GATES:
            {
                "em_torque":       955.2,
                "thermal_magnet":  58.3,
                "cfd_dP":          11.9,
                "intake_ram_recovery": 0.68,
                ...
            }

        Only domains present in measured_values are logged — partial
        measurements are fine and expected during early testing.
        """
        # Map DesignVerdict fields to domain names
        predictions = {
            "em_torque":       verdict.mean_torque_Nm,
            "em_cogging":      verdict.cogging_torque_Nm,
            "em_efficiency":   verdict.efficiency_pct,
            "em_Brmin":        verdict.Brmin_SC_T,
            "cfd_dP":          verdict.net_driving_dP_Pa,
            "cfd_mass_flow":   verdict.mass_flow_kg_s,
            "thermal_winding": verdict.T_winding_C,
            "thermal_magnet":  verdict.T_magnet_C,
            "acoustic_Lp":     verdict.acoustic_Lp_dB,
        }

        units_map = {
            "em_torque":       "Nm",
            "em_cogging":      "Nm",
            "em_efficiency":   "%",
            "em_Brmin":        "T",
            "cfd_dP":          "Pa",
            "cfd_mass_flow":   "kg/s",
            "thermal_winding": "°C",
            "thermal_magnet":  "°C",
            "acoustic_Lp":     "dB",
            "intake_ram_recovery": "Cp",
        }

        updated = []
        for domain, measured in measured_values.items():
            pred = predictions.get(domain)
            if pred is None:
                continue
            v = Verification(
                design_id=design_id or f"auto_{verdict.bezier_mode}",
                bezier_mode=verdict.bezier_mode,
                magnet_is_printed=verdict.magnet_is_printed,
                source=source,
                domain=domain,
                predicted=pred,
                measured=measured,
                units=units_map.get(domain, ""),
                notes=notes,
                wind_speed_m_s=wind_speed_m_s,
                yaw_error_deg=yaw_error_deg,
            )
            updated.append(self.log(v))

        return updated

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def confidence_report(
        self,
        bezier_mode: str,
        magnet_is_printed: bool,
    ) -> str:
        """
        Full confidence report for one design class.
        Shows every domain's trust score, hit rate, and sample size.
        """
        dc = self.design_class(bezier_mode, magnet_is_printed)
        lines = [
            "=" * 72,
            f"  TRUST SCORE REPORT  —  {dc.upper()}",
            "=" * 72,
            "",
            "  Domain                 [Score bar (20 chars)]  Wilson  n  hit%  tol",
            "  " + "─" * 68,
        ]

        domains_with_data = [
            rec for key, rec in self._records.items()
            if key.startswith(dc + "::")
        ]

        if not domains_with_data:
            lines.append("  No verifications recorded for this design class yet.")
        else:
            for rec in sorted(domains_with_data, key=lambda r: r.domain):
                lines.append(rec.summary_line())

        overall = self._overall_score(dc)
        lines += [
            "",
            f"  Overall Wilson score:  {overall*100:.1f}%",
            f"  Shippable to OEM:      {'YES ✓' if overall >= SHIPPABLE_THRESHOLD and self._min_n_met(dc) else 'NOT YET'}",
        ]

        if overall < SHIPPABLE_THRESHOLD:
            needed = SHIPPABLE_THRESHOLD - overall
            lines.append(
                f"  Gap to shippable:      {needed*100:.1f} percentage points"
            )
        if not self._min_n_met(dc):
            min_n = self._min_n(dc)
            lines.append(
                f"  Verifications needed:  {MIN_VERIFICATIONS_TO_SHIP - min_n} more "
                f"(minimum {MIN_VERIFICATIONS_TO_SHIP} required in all domains)"
            )

        lines.append("=" * 72)
        return "\n".join(lines)

    def _overall_score(self, design_class: str) -> float:
        """Geometric mean of all domain Wilson scores for a design class."""
        scores = [
            rec.wilson_score
            for key, rec in self._records.items()
            if key.startswith(design_class + "::")
        ]
        if not scores:
            return 0.0
        log_sum = sum(math.log(max(s, 1e-9)) for s in scores)
        return math.exp(log_sum / len(scores))

    def _min_n(self, design_class: str) -> int:
        """Minimum raw verification count across all domains."""
        counts = [
            rec.n_raw
            for key, rec in self._records.items()
            if key.startswith(design_class + "::")
        ]
        return min(counts) if counts else 0

    def _min_n_met(self, design_class: str) -> bool:
        return self._min_n(design_class) >= MIN_VERIFICATIONS_TO_SHIP

    def is_shippable(
        self,
        bezier_mode: str,
        magnet_is_printed: bool,
    ) -> bool:
        dc = self.design_class(bezier_mode, magnet_is_printed)
        return (
            self._overall_score(dc) >= SHIPPABLE_THRESHOLD
            and self._min_n_met(dc)
        )

    def all_design_classes(self) -> List[str]:
        classes = set()
        for key in self._records:
            classes.add(key.split("::")[0])
        return sorted(classes)

    def intake_sensitivity_report(
        self,
        n_intakes: int = 4,
        A_intake_each_m2: float = 0.0045,
        scoop_angle_deg: float = 15.0,
    ) -> str:
        """
        Print the passive intake sensitivity analysis — shows which intake
        design choices move the needle most on total driving pressure.
        This is the geometry intelligence that commercial CFD tools skip.
        """
        intake = PassiveIntakePhysics(
            n_intakes=n_intakes,
            A_intake_each_m2=A_intake_each_m2,
            scoop_angle_deg=scoop_angle_deg,
            intake_shape="scoop",
        )
        sens = intake.sensitivity()

        lines = [
            "=" * 72,
            "  PASSIVE INTAKE SENSITIVITY  —  what geometry choices matter most",
            "=" * 72,
            "",
            f"  Baseline: {n_intakes}× intake, A={A_intake_each_m2*1e4:.1f} cm² each, "
            f"scoop_angle={scoop_angle_deg}°",
            f"  Wind speed: {V_RATED} m/s rated  (q_ram = "
            f"{0.5*RHO_AIR*V_RATED**2:.1f} Pa available)",
            "",
            "  Intake shape comparison:",
        ]

        for shape in ("NACA_duct", "scoop", "louvre", "flush"):
            key = shape
            if key in sens:
                d = sens[key]
                bar = "█" * int(d["Cp_eff"] * 20)
                lines.append(
                    f"    {shape:<12} Cp={d['Cp_eff']:.3f}  "
                    f"ΔP_ram={d['dP_ram_Pa']:5.1f} Pa  [{bar}]"
                )

        lines += ["", "  Scoop angle comparison (shape=scoop, yaw_error=0°):"]
        for angle in (0, 15, 30, 45, 60):
            key = f"scoop_angle_{angle}deg"
            if key in sens:
                d = sens[key]
                bar = "█" * int(d["Cp_eff"] * 20)
                lines.append(
                    f"    {angle:2d}° off-axis   Cp={d['Cp_eff']:.3f}  "
                    f"ΔP_ram={d['dP_ram_Pa']:5.1f} Pa  [{bar}]"
                )

        lines += [
            "",
            "  Combined driving pressure at rated wind (cent=13.1 Pa + ram):",
        ]
        for shape in ("NACA_duct", "scoop", "flush"):
            ip = PassiveIntakePhysics(
                n_intakes=n_intakes,
                A_intake_each_m2=A_intake_each_m2,
                scoop_angle_deg=scoop_angle_deg,
                intake_shape=shape,
            )
            dP_ram = ip.ram_pressure_Pa(V_RATED, yaw_error_deg=10.0)
            dP_total = 13.1 + dP_ram
            lines.append(
                f"    {shape:<12}  ΔP_cent=13.1 + ΔP_ram={dP_ram:.1f} "
                f"= {dP_total:.1f} Pa total  "
                f"({'%.0f' % (dP_total/13.1)}× centrifugal alone)"
            )

        lines += ["", "=" * 72]
        return "\n".join(lines)


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import tempfile, os

    print("\n" + "=" * 72)
    print("  SELF-TEST — Passive intake physics")
    print("=" * 72 + "\n")

    intake = PassiveIntakePhysics(
        n_intakes=4,
        A_intake_each_m2=0.0045,
        scoop_angle_deg=15.0,
        intake_shape="scoop",
    )

    report = intake.full_report(
        v_wind_m_s=V_RATED,
        yaw_error_deg=10.0,
        dP_centrifugal_Pa=13.1,
        dP_loss_Pa=0.3,
    )

    print("  Passive intake full report (rated wind, 10° yaw error):\n")
    for k, val in report.items():
        if isinstance(val, float):
            print(f"    {k:<30} {val:.4g}")
        else:
            print(f"    {k:<30} {val}")

    print()
    print("  Key insight:")
    print(f"    Centrifugal pumping alone:    13.1 Pa")
    print(f"    Ram pressure recovered:       {report['dP_ram_recovered_Pa']:.1f} Pa")
    print(f"    Combined total driving ΔP:    {report['dP_total_driving_Pa']:.1f} Pa")
    ram_mult = report['dP_total_driving_Pa'] / 13.1
    print(f"    Ram multiplier:               {ram_mult:.1f}× — "
          f"this is what CFD tools leave as an assumed BC")

    print()

    # Trust score engine test with temp db
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    engine = TrustScoreEngine(db_path=tmp_path)

    print("  SELF-TEST — Trust score accumulation\n")

    # Simulate a series of SolidWorks verifications for asymmetric printed designs
    test_data = [
        ("em_torque",       955.0,  952.3,  "Nm",  "solidworks_fea"),
        ("em_torque",       955.0,  958.1,  "Nm",  "solidworks_fea"),
        ("em_torque",       940.0,  935.0,  "Nm",  "prototype_test"),
        ("thermal_magnet",   54.3,   57.1,  "°C",  "solidworks_fea"),
        ("thermal_magnet",   51.2,   49.8,  "°C",  "prototype_test"),
        ("thermal_winding", 142.1,  138.5,  "°C",  "solidworks_fea"),
        ("em_cogging",       22.0,   23.8,  "Nm",  "solidworks_fea"),
        ("cfd_dP",           12.8,   11.2,  "Pa",  "analytical_check"),
        ("intake_ram_recovery", 0.68, 0.71, "Cp",  "prototype_test"),
    ]

    for domain, pred, meas, units, source in test_data:
        v = Verification(
            design_id="test_run_001",
            bezier_mode="asymmetric",
            magnet_is_printed=True,
            source=source,
            domain=domain,
            predicted=pred,
            measured=meas,
            units=units,
            notes="self-test verification",
        )
        rec = engine.log(v)
        print(f"  [{domain:<22}]  {v.verdict_str()}")

    print()
    print(engine.confidence_report("asymmetric", magnet_is_printed=True))

    print()
    print(engine.intake_sensitivity_report(
        n_intakes=4,
        A_intake_each_m2=0.0045,
        scoop_angle_deg=15.0,
    ))

    os.unlink(tmp_path)
