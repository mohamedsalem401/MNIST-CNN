# MNIST Hybrid NN + KNN-Memory Research Toolkit

This repository now provides a full experimental toolkit for studying **hidden-state retrieval interventions** in MNIST classifiers.

## What this toolkit supports
- Baselines:
  - Plain NN (`nn`)
  - Larger NN (`nn_large`)
  - NN + extra parametric layer (`nn_extra_layer`)
  - Final-layer embedding KNN (`embedding_knn`)
- Hybrid NN + memory model with configurable:
  - Intervention layer (`early` / `middle` / `late` / `penultimate` or explicit layer name)
  - Number/selection of affected hidden dimensions
  - Query representation (`full`, `untouched`, `projection`)
  - Value type (`absolute`, `delta`)
  - Intervention mode (`overwrite`, `residual`, `gated`)
  - Train-time vs inference-time memory usage
- Memory retrieval and management:
  - Euclidean/cosine KNN
  - top-k and weighted aggregation
  - Staleness-aware policies: none, TTL, FIFO, reservoir, usage, helpfulness, helpfulness+age
  - Refresh updates via EMA
- Controls:
  - Memory inactive sanity check
  - Random-memory retrieval control
- Evaluation and analysis:
  - Loss/accuracy/ECE
  - Per-class metrics and confusion matrices
  - Helped/harmed sample fractions via memory on/off toggles
  - Retrieval purity/distance stats
  - Intervention magnitude, memory age/usefulness/usage stats
  - Overhead and footprint tracking

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
