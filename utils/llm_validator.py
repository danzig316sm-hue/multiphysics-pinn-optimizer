"""
LLM Physics Critic — Ollama Interface for Mobius-Nova PMSG Optimizer.
utils/llm_validator.py

Integrates a locally-hosted LLM (Llama 3.2 via Ollama) as a semantic
physics validation critic in the self-correction loop.

ROLE IN THE PIPELINE:
    1. Physics residual checker (self_correction.py) runs FIRST — hard constraint.
       If residual > threshold → flag for LLM analysis.
    2. LLM critic runs SECOND — semantic reasoning about the violation.
       Interprets what the residual means physically and recommends
       which loss component to adjust.
    3. Decision engine uses BOTH signals to trigger retraining or
       adjust physics weights.

This separation is intentional:
    - The residual checker is fast, mathematical, objective.
    - The LLM critic is slower, semantic, interpretive.
    - Neither replaces the other. Both are required.

SYSTEM PROMPT PHILOSOPHY:
    The LLM is not a generic physics assistant. It knows exactly:
    - Which machine it is optimizing (Bergey 15-kW, 60/50, 150 RPM)
    - Which constraints are hard vs soft
    - What the NREL/ORNL reference values are
    - That axial stiffness carries 2x weight (NREL conclusion vii)
    - That F = J × B is the governing equation that never changes

RESPONSE FORMAT:
    Structured — no prose, no explanation, exact fields only.
    STATUS: VALID | SUSPECT | INVALID
    VIOLATION: <what physical law or constraint is violated>
    ADJUSTMENT: physics_loss | boundary_loss | data_loss | none
    CONFIDENCE: 0.0-1.0
    PRIORITY: low | medium | high | critical

Usage:
    from utils.llm_validator import LLMPhysicsCritic, CriticVerdict

    critic = LLMPhysicsCritic()
    verdict = critic.analyze(
        prediction=physics_outputs,
        residual_error=0.0045,
        epoch=142,
        design_mode="asymmetric",
    )
    if verdict.status == "INVALID" and verdict.priority == "critical":
        trigger_retraining()
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import httpx
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generator constants — same as trust_score_engine.py (self-contained)
# ---------------------------------------------------------------------------

NREL_REFERENCE = {
    "torque_Nm":              955.0,   # rated torque at 150 RPM
    "cogging_Nm":              19.1,   # 2% of rated
    "efficiency_pct":          93.0,   # direct-drive PMSG target
    "flux_density_T":           1.35,  # rated operating flux
    "magnet_temp_C":           60.0,   # N48H hard limit
    "axial_deform_mm":          6.35,  # NREL Table 3 — binding constraint (2x weight)
    "radial_deform_mm":         0.38,  # NREL Table 3
    "back_emf_thd_pct":         3.0,   # grid-tie standard
    "total_magnet_mass_kg":    24.08,  # baseline
    "torque_density_Nm_kg":   351.28,  # NREL MADE3D baseline
}

# Ollama configuration
OLLAMA_HOST    = "http://127.0.0.1:11434"
OLLAMA_MODEL   = "llama3.2:latest"
OLLAMA_TIMEOUT = 30.0   # seconds — LLM calls are not instant
OLLAMA_CTX     = 4096


# ---------------------------------------------------------------------------
# System prompt — Mobius-Nova specific, not generic
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the physics validation critic for Mobius-Nova Energy's
PMSG optimization system. You validate PINN predictions for a 15-kW, 150 RPM,
60-slot 50-pole radial-flux outer-rotor permanent magnet synchronous generator,
modeled on the Bergey Windpower turbine and anchored to NREL/ORNL research
(Sethuraman et al., "Advanced Permanent Magnet Generator Topologies").

GOVERNING EQUATION: F = J × B (Lorentz force — this never changes)

REFERENCE VALUES (NREL/ORNL baseline):
  Rated torque:        955 Nm
  Cogging torque:      < 19.1 Nm (2% of rated)
  Efficiency:          > 93%
  Flux density:        1.35 T at rated conditions
  Magnet temperature:  < 60°C (N48H hard limit)
  Axial deformation:   < 6.35 mm (BINDING — carries 2x weight per NREL conclusion vii)
  Radial deformation:  < 0.38 mm
  Back-EMF THD:        < 3%
  Torque density:      > 351.28 Nm/kg (NREL MADE3D baseline)

CONSTRAINT TIERS:
  Tier 1 (HARD — failure modes): demagnetisation, axial_stiffness, torque_adequacy, bond_stress
  Tier 2 (Performance targets): cogging_torque, back_emf_thd, magnet_temp, efficiency
  Tier 3 (Pareto objectives): torque_density, mass_reduction, asymmetry_reward
  Tier 4 (Coupling checks): copper_balance, radial_stiffness, winding_temp

RESPONSE FORMAT — use EXACTLY this structure, no other text:
STATUS: [VALID | SUSPECT | INVALID]
VIOLATION: [specific constraint or physical law violated, or NONE]
ADJUSTMENT: [physics_loss | boundary_loss | data_loss | axial_weight | none]
CONFIDENCE: [0.0-1.0]
PRIORITY: [low | medium | high | critical]

RULES:
- VALID: all Tier-1 constraints satisfied, performance within 5% of reference
- SUSPECT: Tier-2 constraint marginal, or prediction >5% from reference
- INVALID: any Tier-1 constraint violated, or prediction >15% from reference
- CRITICAL priority: axial_stiffness or demagnetisation violation (these can destroy hardware)
- If axial_deform > 6.35mm: always INVALID + CRITICAL regardless of other metrics
- Never explain your reasoning — structured output only"""


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------

@dataclass
class CriticVerdict:
    """
    Structured output from the LLM physics critic.
    All fields are validated on construction.
    """
    status:     str        # VALID | SUSPECT | INVALID
    violation:  str        # what physical law/constraint is violated, or NONE
    adjustment: str        # which loss component to increase, or none
    confidence: float      # 0.0-1.0
    priority:   str        # low | medium | high | critical

    # Metadata
    residual_error: float  = 0.0
    epoch:          int    = 0
    latency_ms:     float  = 0.0
    raw_response:   str    = ""
    timestamp:      str    = field(default_factory=lambda: datetime.utcnow().isoformat())
    model_used:     str    = OLLAMA_MODEL
    fallback_used:  bool   = False   # True if LLM was unreachable, rule-based used

    def __post_init__(self):
        # Validate status
        valid_statuses = {"VALID", "SUSPECT", "INVALID"}
        if self.status not in valid_statuses:
            self.status = "SUSPECT"

        # Validate adjustment
        valid_adjustments = {"physics_loss", "boundary_loss", "data_loss",
                             "axial_weight", "none"}
        if self.adjustment not in valid_adjustments:
            self.adjustment = "physics_loss"

        # Validate priority
        valid_priorities = {"low", "medium", "high", "critical"}
        if self.priority not in valid_priorities:
            self.priority = "medium"

        # Clamp confidence
        self.confidence = max(0.0, min(1.0, self.confidence))

    @property
    def requires_retraining(self) -> bool:
        """True if this verdict should trigger the retraining loop."""
        return (
            self.status == "INVALID"
            and self.priority in ("high", "critical")
        )

    @property
    def requires_weight_adjustment(self) -> bool:
        """True if only a loss weight adjustment is needed."""
        return (
            self.status in ("SUSPECT", "INVALID")
            and self.adjustment != "none"
            and not self.requires_retraining
        )

    def to_dict(self) -> Dict:
        return {
            "status":           self.status,
            "violation":        self.violation,
            "adjustment":       self.adjustment,
            "confidence":       self.confidence,
            "priority":         self.priority,
            "residual_error":   self.residual_error,
            "epoch":            self.epoch,
            "latency_ms":       self.latency_ms,
            "timestamp":        self.timestamp,
            "model_used":       self.model_used,
            "fallback_used":    self.fallback_used,
            "requires_retraining":       self.requires_retraining,
            "requires_weight_adjustment": self.requires_weight_adjustment,
        }

    def log_line(self) -> str:
        icon = {"VALID": "✓", "SUSPECT": "⚠", "INVALID": "✗"}.get(self.status, "?")
        return (
            f"[LLM Critic] {icon} {self.status:<8} "
            f"priority={self.priority:<8} "
            f"adj={self.adjustment:<15} "
            f"conf={self.confidence:.2f} "
            f"residual={self.residual_error:.2e} "
            f"latency={self.latency_ms:.0f}ms"
            + (f" [FALLBACK]" if self.fallback_used else "")
        )


# ---------------------------------------------------------------------------
# Rule-based fallback critic
# — used when Ollama is unreachable or times out
# — deterministic, testable, always available
# ---------------------------------------------------------------------------

def _rule_based_verdict(
    prediction: np.ndarray,
    residual_error: float,
    epoch: int,
) -> CriticVerdict:
    """
    Deterministic rule-based physics critic.
    Implements the same logic as the LLM system prompt but in code.
    Used as fallback when Ollama is unreachable.
    """
    status     = "VALID"
    violation  = "NONE"
    adjustment = "none"
    priority   = "low"

    # Map prediction array to named outputs
    # Matches PHYSICS_OUTPUT_FIELDS in design_genome.py
    outputs = {}
    if len(prediction) >= 12:
        field_names = [
            "copper_loss_W", "iron_loss_W", "magnet_temp_C",
            "radial_deform_mm", "axial_deform_mm", "bond_stress_MPa",
            "torque_Nm", "cogging_Nm", "efficiency_pct",
            "flux_density_T", "back_emf_thd_pct", "torque_density_Nm_kg",
        ]
        for i, name in enumerate(field_names):
            if i < len(prediction):
                outputs[name] = float(prediction[i])

    # TIER 1 — Hard constraints (CRITICAL)
    # Axial stiffness carries 2x weight per NREL conclusion vii
    axial = outputs.get("axial_deform_mm", 0.0)
    if axial > NREL_REFERENCE["axial_deform_mm"]:
        status    = "INVALID"
        violation = f"axial_stiffness: {axial:.3f}mm > {NREL_REFERENCE['axial_deform_mm']}mm limit (2x weight)"
        adjustment = "axial_weight"
        priority   = "critical"
        return CriticVerdict(
            status=status, violation=violation, adjustment=adjustment,
            confidence=0.95, priority=priority,
            residual_error=residual_error, epoch=epoch,
            fallback_used=True,
        )

    mag_temp = outputs.get("magnet_temp_C", 0.0)
    if mag_temp > NREL_REFERENCE["magnet_temp_C"]:
        status    = "INVALID"
        violation = f"demagnetisation risk: T_magnet={mag_temp:.1f}°C > 60°C N48H limit"
        adjustment = "physics_loss"
        priority   = "critical"
        return CriticVerdict(
            status=status, violation=violation, adjustment=adjustment,
            confidence=0.93, priority=priority,
            residual_error=residual_error, epoch=epoch,
            fallback_used=True,
        )

    torque = outputs.get("torque_Nm", NREL_REFERENCE["torque_Nm"])
    if torque < NREL_REFERENCE["torque_Nm"] * 0.85:
        status    = "INVALID"
        violation = f"torque_adequacy: {torque:.1f}Nm < 85% of rated 955Nm"
        adjustment = "physics_loss"
        priority   = "high"
        return CriticVerdict(
            status=status, violation=violation, adjustment=adjustment,
            confidence=0.90, priority=priority,
            residual_error=residual_error, epoch=epoch,
            fallback_used=True,
        )

    # TIER 2 — Performance targets (SUSPECT/HIGH)
    eff = outputs.get("efficiency_pct", 93.0)
    if eff < NREL_REFERENCE["efficiency_pct"] * 0.95:
        status    = "SUSPECT"
        violation = f"efficiency: {eff:.1f}% below 95% of {NREL_REFERENCE['efficiency_pct']}% target"
        adjustment = "data_loss"
        priority   = "high"

    cogging = outputs.get("cogging_Nm", 0.0)
    if cogging > NREL_REFERENCE["cogging_Nm"] * 1.5:
        status    = "SUSPECT"
        violation = f"cogging_torque: {cogging:.1f}Nm > 1.5x {NREL_REFERENCE['cogging_Nm']}Nm limit"
        adjustment = "boundary_loss"
        priority   = "medium"

    # Residual-based assessment
    if residual_error > 1e-2:
        if status == "VALID":
            status    = "INVALID"
            violation = f"PDE residual {residual_error:.2e} >> 1e-2 threshold"
            adjustment = "physics_loss"
            priority   = "high"
    elif residual_error > 1e-3:
        if status == "VALID":
            status    = "SUSPECT"
            violation = f"PDE residual {residual_error:.2e} > 1e-3 threshold"
            adjustment = "physics_loss"
            priority   = "medium"

    return CriticVerdict(
        status=status, violation=violation, adjustment=adjustment,
        confidence=0.80, priority=priority,
        residual_error=residual_error, epoch=epoch,
        fallback_used=True,
    )


# ---------------------------------------------------------------------------
# LLM Physics Critic
# ---------------------------------------------------------------------------

class LLMPhysicsCritic:
    """
    Ollama-backed physics validation critic for the Mobius-Nova PINN optimizer.

    Calls the locally-hosted Llama 3.2 model with a Mobius-Nova-specific
    system prompt. Falls back to deterministic rule-based validation if
    Ollama is unreachable or times out.

    Parameters
    ----------
    ollama_host     : Ollama API base URL (default: http://127.0.0.1:11434)
    model           : Ollama model name (default: llama3.2:latest)
    timeout_s       : HTTP timeout for LLM calls (default: 30s)
    max_retries     : Number of retry attempts before fallback (default: 2)
    log_dir         : Directory to persist verdict history (default: critic_log/)
    verbose         : Print verdict log lines (default: True)
    """

    def __init__(
        self,
        ollama_host: str  = OLLAMA_HOST,
        model:       str  = OLLAMA_MODEL,
        timeout_s:   float = OLLAMA_TIMEOUT,
        max_retries: int  = 2,
        log_dir:     str  = "critic_log",
        verbose:     bool = True,
    ):
        self.host        = ollama_host
        self.model       = model
        self.timeout_s   = timeout_s
        self.max_retries = max_retries
        self.log_dir     = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose     = verbose

        self._history: List[CriticVerdict] = []
        self._failure_count: int = 0
        self._consecutive_invalid: int = 0

        # Test connectivity on init
        self._ollama_available = self._test_connection()
        if self.verbose:
            status = "✓ ONLINE" if self._ollama_available else "✗ OFFLINE (fallback mode)"
            print(f"[LLMPhysicsCritic] Ollama {status}  model={self.model}")

    # ------------------------------------------------------------------ #
    #  Primary interface                                                   #
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        prediction:      np.ndarray,
        residual_error:  float,
        epoch:           int = 0,
        design_mode:     str = "asymmetric",
        extra_context:   Optional[str] = None,
    ) -> CriticVerdict:
        """
        Analyze a PINN prediction and residual error.

        Parameters
        ----------
        prediction      : (12,) physics output array from PINN
                          [copper_loss, iron_loss, magnet_temp, radial_deform,
                           axial_deform, bond_stress, torque, cogging, efficiency,
                           flux_density, back_emf_thd, torque_density]
        residual_error  : PDE residual ||N[u]||² at collocation points
        epoch           : current training epoch
        design_mode     : bezier mode (symmetric/asymmetric/multimaterial)
        extra_context   : optional additional context string

        Returns
        -------
        CriticVerdict with status, violation, adjustment, confidence, priority
        """
        t_start = time.time()

        # Always run rule-based check first — instant, no network
        rule_verdict = _rule_based_verdict(prediction, residual_error, epoch)

        # Tier-1 violations (critical/high): rule-based wins immediately
        # LLM cannot override hard physics constraints
        if rule_verdict.status == "INVALID" and rule_verdict.priority in ("critical", "high"):
            rule_verdict.latency_ms = (time.time() - t_start) * 1000
            if self.verbose:
                print(rule_verdict.log_line())
            self._record(rule_verdict)
            self._update_consecutive_invalid(rule_verdict)
            return rule_verdict

        # Try LLM for nuanced Tier-2/3 analysis
        if self._ollama_available:
            verdict = self._call_ollama(
                prediction, residual_error, epoch, design_mode, extra_context
            )
            if verdict is not None:
                # LLM cannot downgrade a rule-based INVALID to VALID
                if rule_verdict.status == "INVALID" and verdict.status == "VALID":
                    verdict.status = "SUSPECT"
                    verdict.violation = rule_verdict.violation + " (LLM downgrade blocked)"
                verdict.latency_ms = (time.time() - t_start) * 1000
                if self.verbose:
                    print(verdict.log_line())
                self._record(verdict)
                self._update_consecutive_invalid(verdict)
                return verdict

        # Fallback: use rule-based
        rule_verdict.latency_ms = (time.time() - t_start) * 1000
        if self.verbose:
            print(rule_verdict.log_line())
        self._record(rule_verdict)
        self._update_consecutive_invalid(rule_verdict)
        return rule_verdict

    @property
    def consecutive_invalid_count(self) -> int:
        """Number of consecutive INVALID verdicts — used by 3-strike trigger."""
        return self._consecutive_invalid

    @property
    def should_trigger_retraining(self) -> bool:
        """
        Three-strike rule: trigger retraining after 3 consecutive INVALID verdicts.
        Matches the decision_engine threshold from the architecture spec.
        """
        return self._consecutive_invalid >= 3

    def reset_strike_count(self):
        """Call after successful retraining to reset the counter."""
        self._consecutive_invalid = 0

    # ------------------------------------------------------------------ #
    #  Ollama API call                                                     #
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        prediction:    np.ndarray,
        residual_error: float,
        epoch:         int,
        design_mode:   str,
        extra_context: Optional[str],
    ) -> str:
        """Build the user message — structured physics context, no prose."""
        field_names = [
            "copper_loss_W", "iron_loss_W", "magnet_temp_C",
            "radial_deform_mm", "axial_deform_mm", "bond_stress_MPa",
            "torque_Nm", "cogging_Nm", "efficiency_pct",
            "flux_density_T", "back_emf_thd_pct", "torque_density_Nm_kg",
        ]

        # Format prediction values with NREL reference comparison
        pred_lines = []
        for i, name in enumerate(field_names):
            if i < len(prediction):
                val = float(prediction[i])
                ref = NREL_REFERENCE.get(name)
                if ref:
                    delta_pct = (val - ref) / abs(ref) * 100
                    pred_lines.append(
                        f"  {name:<28} = {val:>10.4f}  "
                        f"(NREL ref: {ref}, delta: {delta_pct:+.1f}%)"
                    )
                else:
                    pred_lines.append(f"  {name:<28} = {val:>10.4f}")

        pred_block = "\n".join(pred_lines)

        prompt = f"""PINN PREDICTION ANALYSIS REQUEST
Epoch: {epoch}
Design mode: {design_mode}
PDE residual ||N[u]||²: {residual_error:.4e}  (threshold: 1e-3)

Predicted physics outputs vs NREL/ORNL reference:
{pred_block}

{('Additional context: ' + extra_context) if extra_context else ''}

Analyze this prediction. Respond in the exact structured format specified."""

        return prompt

    def _call_ollama(
        self,
        prediction:    np.ndarray,
        residual_error: float,
        epoch:         int,
        design_mode:   str,
        extra_context: Optional[str],
    ) -> Optional[CriticVerdict]:
        """
        Call Ollama API and parse the response.
        Returns None on failure (caller uses fallback).
        """
        prompt = self._build_prompt(
            prediction, residual_error, epoch, design_mode, extra_context
        )

        payload = {
            "model":  self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.1,   # low temperature — we want deterministic output
                "num_ctx":     OLLAMA_CTX,
                "stop":        ["\n\n"],
            },
        }

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.post(
                        f"{self.host}/api/generate",
                        json=payload,
                    )
                resp.raise_for_status()
                raw = resp.json().get("response", "")
                return self._parse_response(raw, residual_error, epoch)

            except httpx.TimeoutException:
                logger.warning(f"[LLMPhysicsCritic] Timeout on attempt {attempt+1}")
                if attempt == self.max_retries - 1:
                    self._ollama_available = False
            except httpx.RequestError as e:
                logger.warning(f"[LLMPhysicsCritic] Connection error: {e}")
                self._ollama_available = False
                break
            except Exception as e:
                logger.warning(f"[LLMPhysicsCritic] Unexpected error: {e}")
                break

        return None

    def _parse_response(
        self,
        raw: str,
        residual_error: float,
        epoch: int,
    ) -> Optional[CriticVerdict]:
        """
        Parse the structured LLM response into a CriticVerdict.
        Expected format:
            STATUS: VALID
            VIOLATION: NONE
            ADJUSTMENT: none
            CONFIDENCE: 0.92
            PRIORITY: low
        """
        lines = {
            line.split(":")[0].strip().upper(): ":".join(line.split(":")[1:]).strip()
            for line in raw.strip().split("\n")
            if ":" in line
        }

        try:
            return CriticVerdict(
                status=lines.get("STATUS", "SUSPECT").strip().upper(),
                violation=lines.get("VIOLATION", "parse_error").strip(),
                adjustment=lines.get("ADJUSTMENT", "physics_loss").strip().lower(),
                confidence=float(lines.get("CONFIDENCE", "0.5").strip()),
                priority=lines.get("PRIORITY", "medium").strip().lower(),
                residual_error=residual_error,
                epoch=epoch,
                raw_response=raw,
                model_used=self.model,
                fallback_used=False,
            )
        except (ValueError, KeyError) as e:
            logger.warning(f"[LLMPhysicsCritic] Parse error: {e}  raw={raw[:100]}")
            return None

    # ------------------------------------------------------------------ #
    #  Internal state management                                           #
    # ------------------------------------------------------------------ #

    def _test_connection(self) -> bool:
        """Ping Ollama to check availability."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def _update_consecutive_invalid(self, verdict: CriticVerdict):
        if verdict.status == "INVALID":
            self._consecutive_invalid += 1
        else:
            # Decay: one valid verdict reduces count by 1
            self._consecutive_invalid = max(0, self._consecutive_invalid - 1)

    def _record(self, verdict: CriticVerdict):
        """Persist verdict to history and append to log file."""
        self._history.append(verdict)
        log_file = self.log_dir / "critic_verdicts.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(verdict.to_dict()) + "\n")

    # ------------------------------------------------------------------ #
    #  Reporting                                                           #
    # ------------------------------------------------------------------ #

    def session_summary(self) -> Dict:
        """Summary statistics for this session's verdicts."""
        if not self._history:
            return {"n_verdicts": 0}
        statuses = [v.status for v in self._history]
        return {
            "n_verdicts":           len(self._history),
            "n_valid":              statuses.count("VALID"),
            "n_suspect":            statuses.count("SUSPECT"),
            "n_invalid":            statuses.count("INVALID"),
            "pct_valid":            statuses.count("VALID") / len(statuses) * 100,
            "retraining_triggered": sum(1 for v in self._history if v.requires_retraining),
            "mean_confidence":      sum(v.confidence for v in self._history) / len(self._history),
            "ollama_calls":         sum(1 for v in self._history if not v.fallback_used),
            "fallback_calls":       sum(1 for v in self._history if v.fallback_used),
            "mean_latency_ms":      sum(v.latency_ms for v in self._history) / len(self._history),
        }

    def print_session_summary(self):
        s = self.session_summary()
        if s["n_verdicts"] == 0:
            print("[LLMPhysicsCritic] No verdicts this session.")
            return
        print("\n" + "=" * 60)
        print("  LLM CRITIC SESSION SUMMARY")
        print("=" * 60)
        print(f"  Total verdicts:     {s['n_verdicts']}")
        print(f"  Valid:              {s['n_valid']}  ({s['pct_valid']:.1f}%)")
        print(f"  Suspect:            {s['n_suspect']}")
        print(f"  Invalid:            {s['n_invalid']}")
        print(f"  Retraining triggers:{s['retraining_triggered']}")
        print(f"  Mean confidence:    {s['mean_confidence']:.3f}")
        print(f"  Ollama calls:       {s['ollama_calls']}")
        print(f"  Fallback calls:     {s['fallback_calls']}")
        print(f"  Mean latency:       {s['mean_latency_ms']:.0f}ms")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running LLMPhysicsCritic self-test...\n")

    critic = LLMPhysicsCritic(verbose=True)

    # Test 1: Good prediction — should be VALID
    good_pred = np.array([
        120.0,   # copper_loss_W
        80.0,    # iron_loss_W
        52.0,    # magnet_temp_C    (< 60 ✓)
        0.25,    # radial_deform_mm (< 0.38 ✓)
        4.8,     # axial_deform_mm  (< 6.35 ✓)
        22.0,    # bond_stress_MPa
        960.0,   # torque_Nm        (≈ 955 ✓)
        18.0,    # cogging_Nm       (< 19.1 ✓)
        93.5,    # efficiency_pct   (> 93 ✓)
        1.36,    # flux_density_T   (≈ 1.35 ✓)
        2.8,     # back_emf_thd_pct (< 3 ✓)
        358.0,   # torque_density   (> 351.28 ✓)
    ], dtype=np.float32)

    print("Test 1: Good prediction (should be VALID)")
    v1 = critic.analyze(good_pred, residual_error=2e-4, epoch=50)
    print(f"  Result: {v1.status} | {v1.violation[:50]}")

    # Test 2: Axial stiffness violation — should be INVALID + CRITICAL
    bad_pred = good_pred.copy()
    bad_pred[4] = 8.5   # axial_deform_mm > 6.35 limit

    print("\nTest 2: Axial stiffness violation (should be INVALID/CRITICAL)")
    v2 = critic.analyze(bad_pred, residual_error=5e-3, epoch=75)
    print(f"  Result: {v2.status} | priority={v2.priority} | {v2.violation[:60]}")
    assert v2.status == "INVALID", f"Expected INVALID, got {v2.status}"
    assert v2.priority == "critical", f"Expected critical, got {v2.priority}"

    # Test 3: High residual — should be SUSPECT or INVALID
    med_pred = good_pred.copy()
    print("\nTest 3: High PDE residual (should be SUSPECT/INVALID)")
    v3 = critic.analyze(med_pred, residual_error=5e-3, epoch=100)
    print(f"  Result: {v3.status} | adj={v3.adjustment} | {v3.violation[:50]}")

    # Test 4: Three-strike trigger
    print("\nTest 4: Three-strike trigger")
    critic2 = LLMPhysicsCritic(verbose=False)
    for i in range(3):
        bad = good_pred.copy()
        bad[4] = 9.0  # axial violation
        critic2.analyze(bad, residual_error=0.05, epoch=i)
    print(f"  Consecutive invalids: {critic2.consecutive_invalid_count}")
    print(f"  Should trigger retraining: {critic2.should_trigger_retraining}")
    assert critic2.should_trigger_retraining, "Three-strike trigger failed"

    critic.print_session_summary()
    print("Self-test complete. ✓")
