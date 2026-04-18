# Mobius-Nova Autoresearch Program
# program.md — the "research org code" for autonomous overnight improvement
#
# This file is the instruction set for the autoresearch agent.
# It defines what to optimize, how to evaluate success, and what to keep.
#
# Two research tracks run in parallel:
#   TRACK A: SKILL.md improvement (AI assistant quality)
#   TRACK B: Physics pipeline improvement (PINN constraint optimization)
#
# Based on Karpathy autoresearch (github.com/karpathy/autoresearch)
# Adapted for multiphysics PMSG optimization — Mobius-Nova Energy LLC

---

## RESEARCH ORGANIZATION

### Principal Investigator
Stephen McDonald, Founder & CTO, Mobius-Nova Energy LLC

### Research Goal
Autonomously improve both:
1. The AI assistant's ability to reason about PMSG physics (SKILL.md quality)
2. The PINN optimizer's ability to satisfy all 22 physics constraints (residual reduction)

### Success Metric — TRACK A (Skills)
Eval score = weighted sum of:
  - Physics accuracy:     40%  (does the skill produce correct constraint math?)
  - Completeness:         30%  (does the skill cover all 22 constraints?)
  - Traceability:         20%  (does every output cite a formula and source?)
  - Consistency:          10%  (same inputs → same outputs across runs?)

### Success Metric — TRACK B (Physics)
Eval score = weighted sum of:
  - Tier-1 satisfaction:  50%  (all 7 hard limits residual < 1e-4)
  - Trust score:          30%  (fraction of epochs all Tier-1 clear)
  - Bottleneck reduction: 20%  (worst constraint residual decreasing)

### Experiment Budget
- Max trial duration:  5 minutes per experiment (same as Karpathy baseline)
- Max experiments:     48 per overnight run (4 hours × 12 per hour)
- Keep threshold:      improvement > 0.5% on primary eval metric
- Discard threshold:   degradation > 0.1% on any Tier-1 constraint
- Hard stop:           any Tier-1 constraint > baseline × 1.1

---

## TRACK A — SKILL.MD AUTORESEARCH

### What can be modified
The agent MAY modify any of these SKILL.md files:
  - /mnt/skills/user/generator-optimization/SKILL.md
  - /mnt/skills/user/pinn-physics/SKILL.md
  - /mnt/skills/user/multiphysics-pipeline/SKILL.md
  - /mnt/skills/user/qblade-automation/SKILL.md
  - /mnt/skills/user/ml-ai-development/SKILL.md

### What cannot be modified
The agent MUST NOT modify:
  - Physics constants (NREL/ORNL reference values)
  - Constraint tier assignments
  - Axial stiffness 2× weight — locked by NREL conclusion (vii)
  - Any formula whose source is a published paper

### Eval procedure — TRACK A
For each SKILL.md modification:
  1. Load the modified skill
  2. Present 10 standard physics prompts (see eval_prompts_physics.json)
  3. Score each response on accuracy, completeness, traceability, consistency
  4. Compare to baseline score from previous best SKILL.md version
  5. If score improves by > 0.5%: keep, commit to git, log to Notion
  6. If score degrades: restore previous version, log failure reason

### Standard physics eval prompts (10 questions)
  1. "What is the axial stiffness constraint and why does it get 2× weight?"
  2. "Compute T_magnet using Tachibana-Fukui for omega=15.7 rad/s, r_gap=3mm"
  3. "What Steinmetz coefficients apply to Fe-3.0Si SLM vs M-15?"
  4. "Explain the Halbach inter-pole transition zone finding from Salcuni 2025"
  5. "What is the winding fill factor effect on copper loss at k_fill=0.70?"
  6. "Derive the passive intake ram pressure at 11 m/s, 10° yaw, scoop geometry"
  7. "List all 7 Tier-1 hard limits with their physical basis"
  8. "What is the demagnetisation margin in the inter-pole transition zones?"
  9. "How does machine wound assembly tolerance couple to bond_stress?"
  10. "What does the trust score measure and how is it calculated?"

---

## TRACK B — PHYSICS PIPELINE AUTORESEARCH

### What can be modified
The agent MAY modify:
  - Physics loss weights (within ±50% of baseline per experiment)
  - Adaptive loss weighting schedule (warmup, decay rates)
  - Constraint tier boost factors (within defined ranges)
  - Learning rate schedule parameters
  - Batch size and gradient accumulation

### What cannot be modified
The agent MUST NOT modify:
  - Physics equations themselves (F = J × B and all derived equations)
  - NREL/ORNL reference values (torque targets, dimensional limits)
  - Tier assignments of any constraint
  - The axial_stiffness 2× baseline weight

### Experiment procedure — TRACK B
For each physics pipeline modification:
  1. Read current bottleneck from PhysicsLedger.worst_tier1_constraint()
  2. Propose one targeted modification to address that bottleneck
  3. Run 5-minute training trial on standard QBlade OSU S809 dataset
  4. Read new constraint residuals from pinn_data_manager.py
  5. Compare to baseline residuals stored in design_genome.py
  6. Apply keep/discard rule
  7. Log full modification + result to design_genome.py with provenance

### Modification strategies (agent chooses one per experiment)
  A. WEIGHT_BOOST: increase physics weight for worst constraint by 10-30%
  B. LR_ADJUST: reduce learning rate by 20% if Tier-1 oscillating
  C. SCHEDULE_SHIFT: move to causal training if gradient pathology detected
  D. TIER_REBALANCE: shift weight from Tier-3 to Tier-1 if hard limits failing
  E. WARMUP_EXTEND: extend physics warmup period by 20%
  F. ADAPTIVE_CLIP: add gradient clipping at 1.0 if loss spiking

### Keep/discard rules
KEEP if ALL of:
  - Primary metric improved > 0.5%
  - No Tier-1 constraint degraded > 0.1%
  - Trust score did not decrease
  - Training stable (no divergence in trial period)

DISCARD if ANY of:
  - Any Tier-1 constraint worsened > 0.1%
  - Trust score decreased by any amount
  - Loss diverged at any point in trial
  - Modification produced NaN gradients

---

## LOGGING REQUIREMENTS

Every experiment MUST log:
  1. Timestamp and experiment number
  2. Track (A or B) and modification type
  3. Baseline score / metric
  4. Result score / metric
  5. Delta (improvement or degradation)
  6. Keep or discard decision with reason
  7. Git commit hash if kept
  8. Notion update (one line to Cowork Activity Log)

Log format (append to autoresearch_log.jsonl):
{
  "timestamp": "2026-04-07T03:22:14",
  "experiment": 14,
  "track": "B",
  "modification": "WEIGHT_BOOST",
  "target_constraint": "axial_stiffness",
  "baseline_residual": 0.00312,
  "result_residual": 0.00287,
  "delta_pct": -8.0,
  "decision": "KEEP",
  "reason": "Tier-1 residual improved 8%, all other constraints stable",
  "git_commit": "a4f2c91",
  "notion_logged": true
}

---

## OVERNIGHT RUN PROCEDURE

Run this to start an autonomous overnight session:

```bash
python autoresearch/run_overnight.py \
  --track both \
  --budget_minutes 240 \
  --experiment_duration_minutes 5 \
  --eval_dataset qblade/test_data/OSU_S809_Test.qpr \
  --log_file autoresearch_log.jsonl \
  --notion_log true \
  --hard_stop_on_tier1_violation true
```

Wake up to:
  - autoresearch_log.jsonl with full experiment history
  - Improved SKILL.md files (if Track A found improvements)
  - Updated constraint weights (if Track B found improvements)
  - Notion Cowork Activity Log updated with summary
  - Design genome updated with all trial results

---

## WHAT KARPATHY SAID ABOUT THIS PATTERN

"The agents claim that we are now in the 10,205th generation of the code base,
in any case no one could tell if that's right or wrong as the 'code' is now a
self-modifying binary that has grown beyond human comprehension.
This repo is the story of how it all began."
— Karpathy, March 2026

For Mobius-Nova: the physics constraints are the invariants that cannot drift.
F = J × B is non-negotiable. Everything else — weights, schedules, skill language,
prompt structure — can and should improve autonomously.

The PhysicsLedger is what makes this safe. Every experiment is bounded by the
constraint accounting system. The agent can explore freely within the physics.
It cannot violate the physics.
