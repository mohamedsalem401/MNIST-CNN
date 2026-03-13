# MNIST Hybrid + Growing-NN Memory Research Toolkit

This repository is a research toolkit for evaluating **hidden-state memory retrieval interventions** in MNIST models, including fixed architectures and progressively growing MLPs.

## Current Research Status
- Prior staged ablation study report: `docs/final_scientific_report.md`
- New growing-architecture study report: `docs/growing_scientific_report.md`
- Latest growing study outcome: growing+memory is currently **unsupported** under control- and baseline-separation criteria.

## What This Toolkit Supports
- Baseline methods:
  - Plain NN (`nn`)
  - Larger NN (`nn_large`)
  - NN + extra parametric layer (`nn_extra_layer`)
  - Embedding-space KNN baseline (`embedding_knn`)
- Hybrid memory-intervention methods:
  - Active hybrid (`hybrid`)
  - Inactive memory control (`hybrid_inactive`)
  - Random-memory control (`random_memory`)
- Intervention controls:
  - Layer placement (`early` / `middle` / `late` / `penultimate` / explicit layer)
  - Query source (`full`, `untouched`, `projection`)
  - Value target type (`delta`, `absolute`)
  - Mode (`gated`, `residual`, `overwrite`)
  - Train-time / inference-time memory usage toggles
- Memory system:
  - L2/cosine retrieval
  - Top-k weighted retrieval
  - Forgetting/eviction policies (`none`, `ttl`, `fifo`, `reservoir`, `usage`, `helpfulness`, `helpfulness_age`)
  - Optional refresh updates
- Growing MLP architecture:
  - Width/depth/combined growth modes
  - Epoch/plateau/performance/fixed-step growth schedules
  - Growth event logging and parameter-count timeline
- Diagnostics:
  - Accuracy/loss/ECE
  - Helped/harmed + strong-help/strong-harm fractions
  - Retrieval purity/distance/top-1 agreement
  - Gate statistics and intervention magnitude
  - Throughput, training time, inference time, memory footprint

## Repository Layout
- `mnist_hybrid/config.py`: experiment schema + YAML loading + CLI overrides
- `mnist_hybrid/data.py`: MNIST split/data loader utilities
- `mnist_hybrid/models/`: MLP/CNN intervenable models
- `mnist_hybrid/memory/`: KNN memory, retrieval, forgetting/refreshing, target construction
- `mnist_hybrid/training/trainer.py`: unified training/evaluation runner
- `mnist_hybrid/analysis/analysis.py`: plotting/diagnostic utilities
- `scripts/run_experiment.py`: single config multi-seed runner
- `scripts/run_matrix.py`: ablation matrix runner
- `scripts/analyze_results.py`: generate plots from run outputs
- `scripts/summarize_matrix.py`: tabular summary export
- `scripts/prepare_growth_stage_b.py`: Stage B promotion + config generation for growth study
- `scripts/build_growth_analysis.py`: growth-study consolidation, pairwise stats, and report writer
- `configs/`: baseline/hybrid configs, ablation matrix, and growth-study matrices
- `docs/research_plan.md`: hypotheses and experiment discipline
- `docs/growth_extension_plan.md`: growth-study plan, matrix, and default hyperparameters

## Results Artifact Map
- Original staged analysis:
  - `results/final_analysis/consolidated_metrics.csv`
  - `results/final_analysis/comparison_tests.csv`
  - `docs/final_scientific_report.md`
- Growing-architecture analysis:
  - `results/growth_analysis/consolidated_metrics.csv`
  - `results/growth_analysis/comparison_tests.csv`
  - `results/growth_analysis/stage_policy.json`
  - `docs/growing_scientific_report.md`

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Single Experiments
```bash
# Plain NN
python scripts/run_experiment.py --config configs/baseline_nn.yaml

# Hybrid with memory
python scripts/run_experiment.py --config configs/base_hybrid.yaml
```

## Run Original Ablation Matrix
```bash
python scripts/run_matrix.py --base-config configs/base_hybrid.yaml --matrix configs/ablation_matrix.yaml
python scripts/summarize_matrix.py --summary results/matrix/matrix_summary.json
```

## Run Growing-Architecture Study (Stage A -> Stage B)
```bash
# Stage A screening
python scripts/run_matrix.py --base-config configs/growth_base_stage_a.yaml --matrix configs/growth_matrix_stage_a.yaml

# Build Stage B promoted configs from Stage A validation only
python scripts/prepare_growth_stage_b.py

# Stage B multi-seed confirmation
python scripts/run_matrix.py --base-config results/growth_analysis/configs/growth_base_stage_b_seed11.yaml --matrix results/growth_analysis/configs/growth_matrix_stage_b_seed11.yaml
python scripts/run_matrix.py --base-config results/growth_analysis/configs/growth_base_stage_b_seed22.yaml --matrix results/growth_analysis/configs/growth_matrix_stage_b_seed22.yaml
python scripts/run_matrix.py --base-config results/growth_analysis/configs/growth_base_stage_b_seed33.yaml --matrix results/growth_analysis/configs/growth_matrix_stage_b_seed33.yaml

# Consolidate and write report
python scripts/build_growth_analysis.py
```

## Analyze a Single Run
```bash
python scripts/analyze_results.py --experiment-dir results/hybrid_base_seed11
```

## Reproducibility Conventions
- Promotion decisions in staged studies use validation metrics only.
- Every run writes:
  - `metrics.json` (config + environment + final/toggle metrics)
  - `epoch_logs.csv` (epoch-level training/validation trace)
  - `details.pt` (sample-level evaluation diagnostics)
  - `analysis/memory_state.pt` when memory snapshots are enabled
- Growth-capable runs additionally log:
  - growth event timeline
  - parameter-count timeline
  - pre/post-growth validation deltas

## Reporting Guidance
- Separate evidence-backed conclusions from interpretation and open questions.
- Report negative/null findings explicitly, especially for active-memory vs random/inactive controls.
- Do not claim success unless confidence intervals exclude zero for the required causal comparisons.

## Legacy files
Legacy visualization/training scripts remain available:
- `train.py`
- `visualize_forward_pass.py`
