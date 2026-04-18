"""
Design Genome Database for Mobius-Nova Energy PMSG Optimizer.
utils/design_genome.py

THE CORE PRINCIPLE:
    Every reading has a complete formula behind it.
    Every number stored knows exactly where it came from,
    what physics produced it, what constraints were active,
    and how it compares to every previous result.

    If a torque density of 387 Nm/kg is logged, the genome stores:
        - The exact 40-element Bezier design vector that produced it
        - The full 14-constraint residual ledger at that epoch
        - The mass accounting (sintered kg, printed kg, core kg, total)
        - The PINN prediction AND any FEA/SolidWorks validation delta
        - The physics_weight active during that run
        - Whether all Tier-1 hard limits were satisfied
        - The parent design it mutated from (lineage tree)
        - Timestamp, run ID, session ID

STORAGE LAYOUT:
    design_genome/
        genome_index.json          fast metadata index for all designs
        pareto_front.json          current Pareto-optimal design IDs
        trust_log.json             SolidWorks / FEA delta history
        lineage.json               parent-to-child mutation tree
        vectors/{design_id}.npz    numpy arrays per design
        reports/{design_id}.json   full accounting report per design

QUERY CAPABILITIES:
    genome.top_k_by(metric, k)
    genome.similarity_search(vec, k)
    genome.pareto_front(objectives)
    genome.designs_meeting(**filters)
    genome.lineage_of(design_id)
    genome.delta_report(id_a, id_b)
    genome.full_accounting(design_id)
    genome.trust_score_summary
    genome.print_summary()
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Field name registries
# ---------------------------------------------------------------------------

PHYSICS_OUTPUT_FIELDS = [
    "copper_loss_W",       # thermal 0
    "iron_loss_W",         # thermal 1
    "magnet_temp_C",       # thermal 2
    "radial_deform_mm",    # stress  3
    "axial_deform_mm",     # stress  4
    "bond_stress_MPa",     # stress  5
    "torque_Nm",           # em      6
    "cogging_Nm",          # em      7
    "efficiency_pct",      # em      8
    "flux_density_T",      # em      9
    "back_emf_thd_pct",    # em      10
    "torque_density_Nm_kg", # em     11
]

CONSTRAINT_FIELDS = [
    # Tier 1 — Hard limits (failure modes)
    "demagnetisation", "axial_stiffness", "torque_adequacy", "bond_stress",
    # Tier 2 — Performance targets
    "cogging_torque", "back_emf_thd", "magnet_temp", "efficiency_target",
    # Tier 3 — Pareto objectives
    "torque_density", "mass_reduction", "asymmetry_reward", "flux_saturation",
    # Tier 4 — Coupling checks
    "copper_balance", "radial_stiffness", "winding_temp",
]

MASS_FIELDS = [
    "sintered_mass_kg", "printed_mass_kg", "total_magnet_mass_kg",
    "rotor_core_mass_kg", "total_active_mass_kg", "torque_density_Nm_kg",
    "mass_reduction_pct", "asymmetry_index", "cost_index_usd",
]

CONSTRAINT_TIER = {
    "demagnetisation": 1, "axial_stiffness": 1,
    "torque_adequacy": 1, "bond_stress": 1,
    "cogging_torque": 2,  "back_emf_thd": 2,
    "magnet_temp": 2,     "efficiency_target": 2,
    "torque_density": 3,  "mass_reduction": 3,
    "asymmetry_reward": 3,"flux_saturation": 3,
    "copper_balance": 4,  "radial_stiffness": 4, "winding_temp": 4,
}

# NREL/ORNL reference values — the ground truth every reading is measured against
NREL_REFERENCE = {
    "torque_density_Nm_kg":    351.28,  # NREL MADE3D baseline IEA 15-MW
    "total_magnet_mass_kg":     24.08,  # NREL/ORNL paper Table 5 baseline
    "mass_reduction_pct":       27.0,   # NREL Case IV asymmetric best result
    "efficiency_pct":           93.0,   # direct-drive PMSG target
    "cogging_Nm":               19.1,   # 2% of 955 Nm rated torque
    "back_emf_thd_pct":          3.0,   # grid-tie standard
    "magnet_temp_C":            60.0,   # NREL N48H hard limit
    "axial_deform_mm":           6.35,  # NREL Table 3 — binding structural constraint
    "radial_deform_mm":          0.38,  # NREL Table 3
    "bond_stress_MPa":          32.0,   # ORNL printed magnet tensile lower bound
    "flux_density_T":            1.35,  # NREL MADE3D rated operating flux
}


# ---------------------------------------------------------------------------
# DesignRecord — one complete, provenance-complete design entry
# ---------------------------------------------------------------------------

class DesignRecord:
    """
    Complete, self-contained record of one evaluated PMSG design.

    Every field has provenance. Every number traces back to the formula,
    constraint, and NREL/ORNL reference that produced it.
    No partial readings. No missing variables.
    """

    def __init__(
        self,
        design_vector:        np.ndarray,
        physics_outputs:      Optional[np.ndarray],
        constraint_residuals: Optional[Dict[str, float]],
        mass_accounting:      Optional[Dict[str, float]],
        bezier_mode:          str = "asymmetric",
        physics_weight:       float = 0.1,
        epoch:                int = 0,
        run_id:               Optional[str] = None,
        session_id:           Optional[str] = None,
        parent_id:            Optional[str] = None,
        sw_validation:        Optional[Dict[str, float]] = None,
        fea_validation:       Optional[Dict[str, float]] = None,
        notes:                str = "",
    ):
        self.design_id  = self._hash_vector(design_vector)
        self.timestamp  = datetime.now().isoformat()
        self.run_id     = run_id or str(uuid.uuid4())[:8]
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.parent_id  = parent_id

        self.design_vector        = design_vector.astype(np.float32)
        self.physics_outputs      = (
            physics_outputs.astype(np.float32) if physics_outputs is not None
            else np.zeros(12, dtype=np.float32)
        )
        self.constraint_residuals = constraint_residuals or {}
        self.mass_accounting      = mass_accounting or {}

        self.bezier_mode    = bezier_mode
        self.physics_weight = physics_weight
        self.epoch          = epoch
        self.notes          = notes
        self.sw_validation  = sw_validation
        self.fea_validation = fea_validation

        # Derived on construction
        self.composite_score = self._score()
        self.tier1_all_clear = self._tier1_ok()
        self.trust_delta     = self._trust_delta()

    @staticmethod
    def _hash_vector(dv: np.ndarray) -> str:
        return "dv_" + hashlib.sha256(dv.astype(np.float32).tobytes()).hexdigest()[:12]

    def _score(self) -> float:
        """Rubric composite score 0-10. Identical to PMSGPINNModel.design_score()."""
        p = self.physics_outputs
        if len(p) < 12:
            return 0.0
        torque, cogging, eff, _, _, _, td = (
            float(p[6]), float(p[7]), float(p[8]),
            float(p[9]), float(p[10]), float(p[2]), float(p[11])
        )
        mag_temp   = float(p[2])
        axial      = float(p[4])
        s_eff    = min(max((eff - 80.0) / 20.0 * 10.0,    0.0), 10.0)
        s_td     = min(max(td / (351.28 * 1.5) * 10.0,    0.0), 10.0)
        cog_pct  = cogging / (torque + 1e-8) * 100.0
        s_cog    = min(max((1.0 - cog_pct / 2.0) * 10.0,  0.0), 10.0)
        s_therm  = min(max((60.0 - mag_temp) / 60.0 * 10.0, 0.0), 10.0)
        s_struct = min(max((6.35 - axial) / 6.35 * 10.0,  0.0), 10.0)
        return round(
            0.30*s_eff + 0.25*s_td + 0.20*s_cog + 0.15*s_therm + 0.10*s_struct, 3
        )

    def _tier1_ok(self) -> bool:
        tier1 = ["demagnetisation", "axial_stiffness", "torque_adequacy", "bond_stress"]
        return all(self.constraint_residuals.get(c, 1.0) < 1e-3 for c in tier1)

    def _trust_delta(self) -> Optional[float]:
        val = self.sw_validation or self.fea_validation
        if not val:
            return None
        deltas = []
        for field, ext_val in val.items():
            if field in PHYSICS_OUTPUT_FIELDS:
                idx = PHYSICS_OUTPUT_FIELDS.index(field)
                pinn_val = float(self.physics_outputs[idx])
                if abs(ext_val) > 1e-8:
                    deltas.append(abs(pinn_val - ext_val) / abs(ext_val) * 100.0)
        return round(float(np.mean(deltas)), 3) if deltas else None

    def get_output(self, field: str) -> Optional[float]:
        if field in PHYSICS_OUTPUT_FIELDS:
            return float(self.physics_outputs[PHYSICS_OUTPUT_FIELDS.index(field)])
        return self.mass_accounting.get(field)

    def full_accounting(self) -> Dict[str, Any]:
        """
        The complete formula audit — every number, every label,
        every NREL reference, every delta. Nothing missing.
        """
        report = {
            "identity": {
                "design_id":   self.design_id,
                "parent_id":   self.parent_id,
                "timestamp":   self.timestamp,
                "run_id":      self.run_id,
                "session_id":  self.session_id,
                "bezier_mode": self.bezier_mode,
                "epoch":       self.epoch,
                "notes":       self.notes,
            },
            "summary": {
                "composite_score": self.composite_score,
                "tier1_all_clear": self.tier1_all_clear,
                "trust_delta_pct": self.trust_delta,
                "physics_weight":  self.physics_weight,
            },
            "physics_outputs": {},
            "constraint_residuals": {},
            "mass_accounting": self.mass_accounting,
            "vs_nrel_reference": {},
            "validation": {
                "sw_validation":   self.sw_validation,
                "fea_validation":  self.fea_validation,
                "trust_delta_pct": self.trust_delta,
            },
            "design_vector": {
                "r_gap_control_points":  self.design_vector[0:11].tolist(),
                "r_rear_control_points": self.design_vector[11:22].tolist(),
                "r_core_control_points": self.design_vector[22:33].tolist(),
                "ratio_param":           float(self.design_vector[33]),
                "hm1_m":                 float(self.design_vector[34]),
                "vol_pct_norm":          float(self.design_vector[35]),
                "wall_t_m":              float(self.design_vector[36]),
                "n_fins":                float(self.design_vector[37]),
                "fin_h_m":               float(self.design_vector[38]),
                "fin_t_m":               float(self.design_vector[39]),
            },
        }

        # Physics outputs — value + NREL reference + delta
        for i, field in enumerate(PHYSICS_OUTPUT_FIELDS):
            val = float(self.physics_outputs[i]) if i < len(self.physics_outputs) else None
            ref = NREL_REFERENCE.get(field)
            report["physics_outputs"][field] = {
                "value":         round(val, 4) if val is not None else None,
                "nrel_ref":      ref,
                "delta_vs_ref":  round(val - ref, 4) if (val and ref) else None,
            }

        # Constraint residuals — value + tier + pass/fail
        for c in CONSTRAINT_FIELDS:
            val = self.constraint_residuals.get(c)
            report["constraint_residuals"][c] = {
                "residual":  round(val, 6) if val is not None else None,
                "tier":      CONSTRAINT_TIER.get(c, 0),
                "satisfied": (val < 1e-3) if val is not None else None,
            }

        # VS NREL reference — every measurable metric
        for field, ref in NREL_REFERENCE.items():
            actual = self.get_output(field)
            if actual is not None:
                report["vs_nrel_reference"][field] = {
                    "actual":   round(actual, 4),
                    "nrel_ref": ref,
                    "delta":    round(actual - ref, 4),
                    "pct_diff": round((actual - ref) / (abs(ref) + 1e-8) * 100, 2),
                }

        return report

    def to_index_entry(self) -> Dict:
        """Flat dict for genome_index.json — fast lookup without loading arrays."""
        return {
            "design_id":             self.design_id,
            "parent_id":             self.parent_id,
            "timestamp":             self.timestamp,
            "run_id":                self.run_id,
            "session_id":            self.session_id,
            "bezier_mode":           self.bezier_mode,
            "epoch":                 self.epoch,
            "physics_weight":        self.physics_weight,
            "composite_score":       self.composite_score,
            "tier1_all_clear":       self.tier1_all_clear,
            "trust_delta_pct":       self.trust_delta,
            "notes":                 self.notes,
            # Key metrics for fast filtering
            "torque_Nm":             self.get_output("torque_Nm"),
            "efficiency_pct":        self.get_output("efficiency_pct"),
            "torque_density_Nm_kg":  self.get_output("torque_density_Nm_kg"),
            "cogging_Nm":            self.get_output("cogging_Nm"),
            "magnet_temp_C":         self.get_output("magnet_temp_C"),
            "axial_deform_mm":       self.get_output("axial_deform_mm"),
            "flux_density_T":        self.get_output("flux_density_T"),
            "total_magnet_mass_kg":  self.mass_accounting.get("total_magnet_mass_kg"),
            "mass_reduction_pct":    self.mass_accounting.get("mass_reduction_pct"),
            "asymmetry_index":       self.mass_accounting.get("asymmetry_index"),
            "cost_index_usd":        self.mass_accounting.get("cost_index_usd"),
        }


# ---------------------------------------------------------------------------
# DesignGenome — the persistent accumulating database
# ---------------------------------------------------------------------------

class DesignGenome:
    """
    Every run compounds. Nothing is ever lost.
    Every number has a formula. Every formula has a reference.
    """

    def __init__(self, genome_dir: str = "design_genome"):
        self.genome_dir   = Path(genome_dir)
        self._vec_dir     = self.genome_dir / "vectors"
        self._rep_dir     = self.genome_dir / "reports"

        for d in [self.genome_dir, self._vec_dir, self._rep_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._idx_path     = self.genome_dir / "genome_index.json"
        self._pareto_path  = self.genome_dir / "pareto_front.json"
        self._trust_path   = self.genome_dir / "trust_log.json"
        self._lineage_path = self.genome_dir / "lineage.json"

        self._index:   Dict[str, Dict] = self._load(self._idx_path,     {})
        self._pareto:  List[str]        = self._load(self._pareto_path,  [])
        self._trust:   List[Dict]       = self._load(self._trust_path,   [])
        self._lineage: Dict[str, List]  = self._load(self._lineage_path, {})

        print(
            f"[DesignGenome] '{genome_dir}' — "
            f"{len(self._index)} designs, "
            f"{len(self._pareto)} Pareto-optimal, "
            f"{len(self._trust)} SW/FEA validations."
        )

    # ── Ingestion ──────────────────────────────────────────────────────────

    def log(self, record: DesignRecord) -> str:
        """
        Persist one design record. Idempotent on design_id.
        Returns design_id.
        """
        did = record.design_id

        if did in self._index:
            return did  # already stored — no duplicate writes

        # Arrays
        np.savez_compressed(
            self._vec_dir / f"{did}.npz",
            design_vector=record.design_vector,
            physics_outputs=record.physics_outputs,
            constraint_residuals=np.array(
                [record.constraint_residuals.get(c, 0.0) for c in CONSTRAINT_FIELDS],
                dtype=np.float32,
            ),
            mass_accounting=np.array(
                [record.mass_accounting.get(f, 0.0) for f in MASS_FIELDS],
                dtype=np.float32,
            ),
        )

        # Full report JSON
        with open(self._rep_dir / f"{did}.json", "w") as f:
            json.dump(record.full_accounting(), f, indent=2, default=str)

        # Index
        self._index[did] = record.to_index_entry()
        self._save(self._idx_path, self._index)

        # Lineage
        if record.parent_id:
            self._lineage.setdefault(record.parent_id, []).append(did)
            self._save(self._lineage_path, self._lineage)

        # Trust log
        if record.trust_delta is not None:
            self._trust.append({
                "design_id":       did,
                "timestamp":       record.timestamp,
                "trust_delta_pct": record.trust_delta,
                "source":          "sw" if record.sw_validation else "fea",
            })
            self._save(self._trust_path, self._trust)

        # Pareto
        self._refresh_pareto()

        return did

    def log_sw_validation(
        self,
        design_id: str,
        sw_results: Dict[str, float],
    ) -> float:
        """
        Record SolidWorks FEA results for an existing design.
        Computes PINN vs. SW delta and updates the trust log.
        Returns trust_delta_pct.
        """
        if design_id not in self._index:
            raise KeyError(f"Design {design_id} not in genome.")

        npz = np.load(self._vec_dir / f"{design_id}.npz")
        pinn = npz["physics_outputs"]

        deltas = []
        for field, sw_val in sw_results.items():
            if field in PHYSICS_OUTPUT_FIELDS:
                idx = PHYSICS_OUTPUT_FIELDS.index(field)
                if abs(sw_val) > 1e-8:
                    deltas.append(abs(float(pinn[idx]) - sw_val) / abs(sw_val) * 100.0)

        delta = round(float(np.mean(deltas)), 3) if deltas else 0.0

        self._trust.append({
            "design_id":       design_id,
            "timestamp":       datetime.now().isoformat(),
            "trust_delta_pct": delta,
            "source":          "solidworks",
            "sw_results":      sw_results,
        })
        self._save(self._trust_path, self._trust)

        self._index[design_id]["trust_delta_pct"] = delta
        self._save(self._idx_path, self._index)

        print(f"[DesignGenome] SW validation {design_id}: PINN vs SW = {delta:.2f}%")
        return delta

    # ── Queries ────────────────────────────────────────────────────────────

    def top_k_by(
        self,
        metric: str,
        k: int = 10,
        higher_is_better: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Top-K designs by any indexed metric, with optional pre-filters."""
        cands = list(self._index.values())
        if filters:
            cands = [c for c in cands if all(c.get(fk) == fv for fk, fv in filters.items())]
        valid = [c for c in cands if c.get(metric) is not None]
        return sorted(valid, key=lambda x: x[metric], reverse=higher_is_better)[:k]

    def similarity_search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        K nearest designs in Bézier geometry space (first 33 dims).
        Use before evaluating a new candidate — avoid re-exploring.
        """
        q = query_vector[:33].astype(np.float32)
        dists = []
        for did in self._index:
            try:
                stored = np.load(self._vec_dir / f"{did}.npz")["design_vector"][:33]
                dists.append((did, float(np.linalg.norm(q - stored))))
            except Exception:
                continue
        return sorted(dists, key=lambda x: x[1])[:k]

    def designs_meeting(
        self,
        min_efficiency_pct:       Optional[float] = None,
        max_cogging_Nm:           Optional[float] = None,
        min_mass_reduction_pct:   Optional[float] = None,
        min_torque_density_Nm_kg: Optional[float] = None,
        max_magnet_temp_C:        Optional[float] = None,
        max_axial_deform_mm:      Optional[float] = None,
        tier1_all_clear:          bool = True,
    ) -> List[Dict]:
        """
        Physics-criteria filter.
        Returns designs meeting ALL specified thresholds, sorted by composite score.
        """
        out = []
        for e in self._index.values():
            if tier1_all_clear and not e.get("tier1_all_clear"):
                continue
            if min_efficiency_pct       and (e.get("efficiency_pct")       or 0)   < min_efficiency_pct:
                continue
            if max_cogging_Nm           and (e.get("cogging_Nm")           or 999) > max_cogging_Nm:
                continue
            if min_mass_reduction_pct   and (e.get("mass_reduction_pct")   or 0)   < min_mass_reduction_pct:
                continue
            if min_torque_density_Nm_kg and (e.get("torque_density_Nm_kg") or 0)   < min_torque_density_Nm_kg:
                continue
            if max_magnet_temp_C        and (e.get("magnet_temp_C")        or 999) > max_magnet_temp_C:
                continue
            if max_axial_deform_mm      and (e.get("axial_deform_mm")      or 999) > max_axial_deform_mm:
                continue
            out.append(e)
        return sorted(out, key=lambda x: x.get("composite_score", 0), reverse=True)

    def pareto_front(self) -> List[Dict]:
        """Current Pareto-optimal designs (efficiency, mass reduction, torque density)."""
        return [self._index[d] for d in self._pareto if d in self._index]

    def lineage_of(self, design_id: str, depth: int = 20) -> List[str]:
        """Full ancestry chain from root to this design."""
        chain, current = [design_id], design_id
        for _ in range(depth):
            parent = self._index.get(current, {}).get("parent_id")
            if not parent or parent == current:
                break
            chain.insert(0, parent)
            current = parent
        return chain

    def delta_report(self, id_a: str, id_b: str) -> Dict:
        """Before/after comparison of every physics metric between two designs."""
        if id_a not in self._index or id_b not in self._index:
            raise KeyError("One or both design IDs not in genome.")
        a, b = self._index[id_a], self._index[id_b]
        metrics = [
            "composite_score", "torque_Nm", "efficiency_pct",
            "torque_density_Nm_kg", "cogging_Nm", "magnet_temp_C",
            "axial_deform_mm", "flux_density_T",
            "total_magnet_mass_kg", "mass_reduction_pct", "asymmetry_index",
        ]
        changes = {}
        for m in metrics:
            va, vb = a.get(m), b.get(m)
            if va is not None and vb is not None:
                diff = vb - va
                pct  = diff / (abs(va) + 1e-8) * 100.0
                changes[m] = {
                    "before":     round(va, 4),
                    "after":      round(vb, 4),
                    "delta":      round(diff, 4),
                    "pct_change": round(pct, 2),
                    "direction":  "improved" if diff > 0 else "degraded",
                }
        return {"design_a": id_a, "design_b": id_b, "changes": changes}

    def full_accounting(self, design_id: str) -> Dict:
        """Load the complete formula audit for one design."""
        path = self._rep_dir / f"{design_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        raise FileNotFoundError(f"No accounting report for {design_id}")

    # ── Trust score ────────────────────────────────────────────────────────

    @property
    def trust_score_summary(self) -> Dict:
        """
        Running accuracy summary across all external validations.
        This is the commercial credibility number.
        """
        if not self._trust:
            return {"n_validations": 0}
        deltas = [t["trust_delta_pct"] for t in self._trust]
        return {
            "n_validations":    len(deltas),
            "mean_delta_pct":   round(float(np.mean(deltas)), 3),
            "max_delta_pct":    round(float(np.max(deltas)), 3),
            "min_delta_pct":    round(float(np.min(deltas)), 3),
            "pct_within_5pct":  round(sum(d < 5.0 for d in deltas) / len(deltas) * 100, 1),
            "pct_within_10pct": round(sum(d < 10.0 for d in deltas) / len(deltas) * 100, 1),
        }

    # ── Reporting ──────────────────────────────────────────────────────────

    def print_summary(self):
        n = len(self._index)
        t1_clean = sum(1 for d in self._index.values() if d.get("tier1_all_clear"))
        print("\n" + "=" * 70)
        print("  DESIGN GENOME — COMPLETE ACCOUNTING SUMMARY")
        print("=" * 70)
        print(f"  Total designs logged     : {n}")
        print(f"  Tier-1 all-clear         : {t1_clean}  ({t1_clean/max(n,1)*100:.1f}%)")
        print(f"  Pareto-optimal           : {len(self._pareto)}")
        print(f"  SW/FEA validations       : {len(self._trust)}")

        ts = self.trust_score_summary
        if ts.get("n_validations", 0) > 0:
            print(f"\n  ── Commercial Trust Score ───────────────────────────────────")
            print(f"  Validated designs        : {ts['n_validations']}")
            print(f"  Mean PINN vs. SW delta   : {ts['mean_delta_pct']:.2f}%")
            print(f"  Within 5%                : {ts['pct_within_5pct']:.1f}% of runs")
            print(f"  Within 10%               : {ts['pct_within_10pct']:.1f}% of runs")

        top5 = self.top_k_by("composite_score", k=5, filters={"tier1_all_clear": True})
        if top5:
            print(f"\n  ── Top 5 Designs (Tier-1 Clean) ─────────────────────────────")
            hdr = f"  {'Design ID':<14} {'Score':>6} {'Eff%':>6} {'Nm/kg':>8} {'Mass-Red%':>10} {'Cog Nm':>8} {'Mode':<14}"
            print(hdr)
            print(f"  {'-'*65}")
            for d in top5:
                print(
                    f"  {d['design_id']:<14} "
                    f"{d.get('composite_score',0):>6.2f} "
                    f"{d.get('efficiency_pct') or 0:>6.1f} "
                    f"{d.get('torque_density_Nm_kg') or 0:>8.1f} "
                    f"{d.get('mass_reduction_pct') or 0:>9.1f}% "
                    f"{d.get('cogging_Nm') or 0:>8.2f} "
                    f"{d.get('bezier_mode','?'):<14}"
                )

        # NREL reference comparison on best design
        if top5:
            best = top5[0]
            print(f"\n  ── Best Design vs. NREL/ORNL Reference ─────────────────────")
            metrics = [
                ("torque_density_Nm_kg", "Torque density", "Nm/kg", True),
                ("efficiency_pct",       "Efficiency",     "%",     True),
                ("mass_reduction_pct",   "Mass reduction", "%",     True),
                ("cogging_Nm",           "Cogging torque", "Nm",    False),
                ("magnet_temp_C",        "Magnet temp",    "°C",    False),
                ("axial_deform_mm",      "Axial deform",   "mm",    False),
            ]
            for field, label, unit, higher_better in metrics:
                actual = best.get(field)
                ref    = NREL_REFERENCE.get(field)
                if actual is not None and ref is not None:
                    diff = actual - ref
                    pct  = diff / abs(ref) * 100
                    arrow = ("↑" if diff > 0 else "↓")
                    status = ("✓" if (diff > 0) == higher_better else "–")
                    print(
                        f"  {label:<22} {actual:>8.2f} {unit:<6} "
                        f"vs NREL {ref:>8.2f}  {arrow}{abs(pct):5.1f}%  {status}"
                    )

        print("=" * 70 + "\n")

    # ── Internal ───────────────────────────────────────────────────────────

    def _refresh_pareto(self):
        objs = ["efficiency_pct", "mass_reduction_pct", "torque_density_Nm_kg"]
        cands = {
            did: e for did, e in self._index.items()
            if e.get("tier1_all_clear") and all(e.get(o) is not None for o in objs)
        }
        if not cands:
            return
        ids    = list(cands.keys())
        matrix = np.array([[cands[i][o] for o in objs] for i in ids])
        dominated = np.zeros(len(ids), dtype=bool)
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i == j:
                    continue
                if np.all(matrix[j] >= matrix[i]) and np.any(matrix[j] > matrix[i]):
                    dominated[i] = True
                    break
        self._pareto = [ids[i] for i in range(len(ids)) if not dominated[i]]
        self._save(self._pareto_path, self._pareto)

    @staticmethod
    def _load(path: Path, default: Any) -> Any:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return default

    @staticmethod
    def _save(path: Path, data: Any):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# SelfCorrectionLoop integration helper
# ---------------------------------------------------------------------------

def log_epoch_to_genome(
    genome:               DesignGenome,
    design_vector:        np.ndarray,
    physics_outputs:      np.ndarray,
    constraint_residuals: Dict[str, float],
    mass_accounting:      Dict[str, float],
    epoch:                int,
    physics_weight:       float,
    bezier_mode:          str = "asymmetric",
    parent_id:            Optional[str] = None,
    session_id:           Optional[str] = None,
) -> str:
    """Drop-in call from SelfCorrectionLoop.run() to log each epoch's best design."""
    record = DesignRecord(
        design_vector=design_vector,
        physics_outputs=physics_outputs,
        constraint_residuals=constraint_residuals,
        mass_accounting=mass_accounting,
        bezier_mode=bezier_mode,
        physics_weight=physics_weight,
        epoch=epoch,
        parent_id=parent_id,
        session_id=session_id,
    )
    return genome.log(record)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    print("Running DesignGenome smoke test...")

    with tempfile.TemporaryDirectory() as tmp:
        genome = DesignGenome(genome_dir=tmp)
        rng    = np.random.default_rng(0)
        ids    = []

        for i in range(6):
            dv = rng.random(40).astype(np.float32)
            po = np.array([
                120+i*10, 80, 42+i*3,
                0.15, 4.0-i*0.5, 18.0,
                955+i*20, 16-i*1.5, 91+i*0.5,
                1.37, 2.4, 355+i*8
            ], dtype=np.float32)
            cr = {c: max(0.0, 0.008 - i*0.001) for c in CONSTRAINT_FIELDS}
            ma = {
                "sintered_mass_kg": 5.0, "printed_mass_kg": 13-i,
                "total_magnet_mass_kg": 18-i, "rotor_core_mass_kg": 34.0,
                "total_active_mass_kg": 52-i, "torque_density_Nm_kg": 355+i*8,
                "mass_reduction_pct": 20+i*2, "asymmetry_index": 0.04+i*0.01,
                "cost_index_usd": 1100-i*40,
            }
            rec = DesignRecord(
                design_vector=dv, physics_outputs=po,
                constraint_residuals=cr, mass_accounting=ma,
                epoch=i*10, parent_id=ids[-1] if ids else None,
            )
            ids.append(genome.log(rec))

        genome.print_summary()

        print("Delta report (first vs last):")
        dr = genome.delta_report(ids[0], ids[-1])
        for m, v in dr["changes"].items():
            print(f"  {m:<30} {v['before']:>8.2f} → {v['after']:>8.2f}  ({v['pct_change']:+.1f}%  {v['direction']})")

    print("\nSmoke test complete.")
