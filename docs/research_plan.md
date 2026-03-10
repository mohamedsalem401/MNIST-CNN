# Hybrid NN + KNN-Memory on MNIST: Research Plan

## Phase 1: Conceptual Design Space

### Core mechanism
- A base neural network (MLP or small CNN) computes hidden activations.
- At one selected intervention layer, a memory module receives a query from the pre-intervention hidden state.
- The memory returns a retrieved value from top-k nearest neighbors.
- The model applies one of three intervention modes to a selected subset of hidden dimensions:
  - `overwrite`
  - `residual`
  - `gated`

### Memory key/value definitions
- Key: hidden activation before intervention (or query projection/subset).
- Value options:
  - Absolute target hidden vector.
  - Delta correction vector.

### Target construction strategies
- `current_hidden` (heuristic reference; may be weak and identity-like).
- `grad_delta` (backprop-derived delta: `-eta * dL/dh`).
- `grad_target` (absolute target: `h - eta * dL/dh`).
- `class_centroid` (class-conditional running latent target).

### Memory management variables
- Forgetting:
  - none
  - TTL
  - FIFO bounded
  - reservoir sampling
  - usage-based eviction
  - helpfulness-based eviction
  - helpfulness+age hybrid eviction
- Refresh:
  - optional nearest-entry EMA key/value update.

### Retrieval variables
- Query representation:
  - full hidden
  - untouched dimensions only
  - random projection subset
- Distance metric:
  - Euclidean
  - cosine
- Neighbor aggregation:
  - uniform
  - inverse distance
  - softmax weighting

## Explicit assumptions
- The intervention is not end-to-end differentiable through hard neighbor selection.
- Hidden targets are heuristic; no unique “perfect” hidden state exists.
- Memory entries become stale due to representation drift.
- Strong baselines are required to attribute gains to memory retrieval rather than capacity or compute.
- Train-time and inference-time memory effects are separable and should be evaluated independently.

## Main research questions
- Does hidden-state retrieval improve accuracy, calibration, and/or robustness proxies?
- Which intervention mode is best (`overwrite` vs `residual` vs `gated`)?
- Is `delta` value storage superior to `absolute` targets?
- Which insertion layer works best and why?
- Does memory help boundary/hard examples or mostly memorize train distribution?
- How harmful is staleness, and which forgetting/refresh policy controls it best?
- Is retrieval itself causal, or are gains reproduced by stronger parametric baselines?

## Hypotheses (pre-experiment)
- H1: `gated` interventions will be more stable than hard overwrite.
- H2: `grad_delta` values will outperform naive absolute values.
- H3: Later-layer interventions (late/penultimate) will be safer than early-layer interventions.
- H4: Bounded memory with helpfulness-aware retention will outperform no-forgetting under drift.
- H5: Random-memory controls will remove most gains if retrieval quality is the key driver.

## Experiment plan

### Splits and discipline
- Train/val/test using MNIST train split partitioned into train/val and official test split.
- No test leakage into model fitting, memory insertion, or tuning decisions.

### Reproducibility
- Multi-seed runs (default 3 seeds; can scale to 5+).
- All hyperparameters captured via YAML config and logged per run.
- Structured metrics + per-sample diagnostics exported.

### Baselines and controls
- `nn` plain model.
- `nn_large` larger hidden capacity baseline.
- `nn_extra_layer` extra parametric layer baseline.
- `embedding_knn` final-layer embedding-space KNN baseline.
- `hybrid` retrieval intervention variants.
- `hybrid_inactive` memory present but not used.
- `random_memory` shuffled/random retrieval control.

### Primary metrics
- Validation/test accuracy.
- Validation/test loss.
- ECE calibration.
- Helped/harmed fraction under memory on/off toggle on same trained weights.

### Secondary metrics
- Per-class precision/recall/F1.
- Confusion matrix.
- Retrieval purity and distance.
- Intervention magnitude.
- Memory age/usage/usefulness statistics.
- Throughput and memory footprint overhead.

### Ablation matrix
See `configs/ablation_matrix.yaml` for a structured run list covering layer choice, intervention mode, query representation, value target type, forgetting/refresh policies, and train-vs-inference memory usage.

## Interpretation protocol
- Separate supported claims from speculation.
- Report negative results and null effects.
- Tie observed gains to controls (especially random-memory and stronger parametric baselines).
- Explicitly discuss where gains may reflect memorization rather than generalization.
