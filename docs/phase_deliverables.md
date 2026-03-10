# Phase Deliverables

## Phase 1: Conceptual design space, assumptions, and experiment plan
- Defined in `docs/research_plan.md`.
- Includes explicit assumptions, hypotheses, and disciplined experiment protocol.

## Phase 2: Project structure and implementation
- Implemented modular toolkit under `mnist_hybrid/`:
  - `config.py`, `data.py`
  - `models/` (MLP, CNN, factory, intervention-capable forward)
  - `memory/` (retrieval, datastore, target construction, forgetting, refresh, intervention engine)
  - `training/trainer.py` (unified runner for baselines + hybrid variants)
  - `evaluation/` (metrics + embedding KNN baseline)
  - `analysis/` (diagnostic plotting)
- CLI scripts in `scripts/` for running experiments, ablations, analysis, and report stub generation.

## Phase 3: Evaluation + analysis framework and ablation matrix
- Baseline and hybrid configs in `configs/`.
- Ablation/control matrix in `configs/ablation_matrix.yaml`.
- Supports controlled comparisons for:
  - insertion layer
  - intervention mode
  - query representation
  - value type and target construction
  - forgetting/refresh policies
  - train-only vs inference-only memory
  - random-memory and inactive-memory controls

## Phase 4: Report template and interpretation structure
- Full report structure in `docs/report_template.md`.
- Executive summary template in `docs/executive_summary_template.md`.
- Open problems and next steps in `docs/open_questions_next_steps.md`.

## Phase 5: Plausible outcomes and failure modes (without fake certainty)
Likely patterns to test rigorously (not claims):
- Gated interventions may be more stable than hard overwrite.
- Late/penultimate insertion may outperform early insertion.
- Gradient-based deltas may outperform naive absolute hidden targets.
- Helpfulness-aware bounded memory may reduce stale-memory damage.
- Random-memory controls may erase most gains if retrieval quality is causal.

Key failure modes to monitor:
- train/test mismatch from memory usage mode
- stale entries under representation drift
- retrieval collapse to same-class memorization
- hard-overwrite corruption in early layers
- apparent gains disappearing under stronger baselines
