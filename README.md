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

## Project layout
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
- `configs/`: baseline/hybrid configs and ablation matrix
- `docs/research_plan.md`: hypotheses and experiment discipline
- `docs/report_template.md`: full research report structure

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run a baseline
```bash
python scripts/run_experiment.py --config configs/baseline_nn.yaml
```

## Run a hybrid experiment
```bash
python scripts/run_experiment.py --config configs/base_hybrid.yaml
```

## Run full ablation matrix
```bash
python scripts/run_matrix.py --base-config configs/base_hybrid.yaml --matrix configs/ablation_matrix.yaml
python scripts/summarize_matrix.py --summary results/matrix/matrix_summary.json
```

## Analyze one run
```bash
python scripts/analyze_results.py --experiment-dir results/hybrid_base_seed11
```

## Reproducibility notes
- Every run saves:
  - Config snapshot via `metrics.json`
  - Epoch logs (`epoch_logs.csv`)
  - Detailed per-sample artifacts (`details.pt`)
  - Optional memory snapshot (`analysis/memory_state.pt`)
- Multi-seed aggregation is exported to `aggregate_metrics.json`.

## Research workflow
1. Finalize hypotheses in `docs/research_plan.md`.
2. Execute baseline and control configs.
3. Run targeted ablations from `configs/ablation_matrix.yaml`.
4. Generate diagnostics with `scripts/analyze_results.py`.
5. Write the report with `docs/report_template.md` and `docs/executive_summary_template.md`.

## Legacy files
Legacy visualization/training scripts remain available:
- `train.py`
- `visualize_forward_pass.py`
