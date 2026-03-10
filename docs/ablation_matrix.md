# Ablation Matrix Design

This matrix is defined in `configs/ablation_matrix.yaml` and is intentionally compact but question-oriented.

## Core causal checks
- `a0_baseline_nn`: plain parametric baseline.
- `a19_baseline_nn_large`: capacity control.
- `a20_baseline_nn_extra_layer`: extra-depth control.
- `a21_baseline_embedding_knn`: retrieval-at-output baseline.
- `a18_hybrid_inactive`: memory object present but no intervention.
- `a17_random_memory_control`: retrieval signal destroyed.

## Intervention mechanics
- `a1` vs `a2` vs `a3`: gated vs overwrite vs residual at same layer.
- `a1` vs `a4` vs `a5`: penultimate vs middle vs early placement.
- `a1` vs `a7` vs `a8`: query source (full vs untouched vs projection).
- `a1` vs `a6`: delta targets vs absolute targets.

## Memory management
- `a9`: no forgetting.
- `a10`: FIFO bounded.
- `a11`: reservoir bounded.
- `a12`: usage-based eviction.
- `a13`: helpfulness-based eviction.
- `a14`: helpfulness+age + refresh.

## Train vs inference decomposition
- `a15`: train-only memory.
- `a16`: inference-only memory.

## Suggested run order
1. Baselines and controls (`a0`, `a19`, `a20`, `a21`, `a18`, `a17`).
2. Core hybrid intervention mechanics (`a1`-`a8`).
3. Memory management/staleness (`a9`-`a14`).
4. Train/inference decomposition (`a15`, `a16`).
