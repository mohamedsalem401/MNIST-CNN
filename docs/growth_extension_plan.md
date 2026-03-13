# Research Extension Plan: Growing NN + Hidden-State Memory

## Objective
Test whether progressive network growth improves the causal usefulness of hidden-state memory retrieval versus fixed-size and prior strong baselines.

## Stage design
- Stage A (screening): single seed, compact budget, full required matrix + optional stress variants.
- Stage B (confirmation): promoted subset with multi-seed replication (validation-only promotion).
- Primary endpoint: `test_accuracy` in Stage B, with paired seed-level comparisons.
- Promotion metric: `val_accuracy` only.

## Proposed matrix
Required groups:
1. `g0_fixed_small_no_memory`
2. `g1_fixed_small_with_memory`
3. `g2_fixed_large_no_memory_capacity_matched`
4. `g3_fixed_large_with_memory_capacity_matched`
5. `g4_growing_no_memory`
6. `g5_growing_with_memory`
7. `g6_growing_random_memory_control`
8. `g7_growing_inactive_memory_control`
9. `g8_prior_nn_large`
10. `g9_prior_embedding_knn`

Optional add-ons:
11. `g10_growing_with_memory_overwrite`
12. `g11_growing_with_memory_layer_early`
13. `g12_growing_with_memory_uncertainty_gating`

## Explicit hyperparameters and defaults
A. Growth architecture
- `initial_width=48`
- `initial_depth=2`
- `growth_enabled=false` (overridden per run)
- `growth_mode=width_and_depth`
- `growth_schedule=epoch_based`
- `growth_interval=1`
- `growth_amount_width=16`
- `growth_amount_depth=1`
- `max_width=96`
- `max_depth=4`
- `growth_warmup_epochs=1`
- `growth_stop_epoch=3` (Stage A)
- `capacity_match_baseline=false` (overridden in capacity-matched runs)

B. Memory system
- `memory_enabled=true`
- `memory_layer=penultimate`
- `memory_k=8`
- `memory_size=20000`
- `memory_update_policy=always`
- `memory_forgetting_policy=helpfulness_age`
- `memory_distance=l2`
- `memory_query_source=full`
- `memory_value_target=delta`
- `memory_intervention_mode=gated`
- `memory_train_enabled=true`
- `memory_inference_enabled=true`

C. Intervention/gating
- `gate_temperature=1.0`
- `gate_bias_init=0.0`
- `gate_regularization=0.0`
- `intervention_strength=1.0`
- `intervention_clip=0.0`
- `uncertainty_aware_gating=false`

D. Optimization/training
- `seed=11` (Stage A), Stage B seeds generated automatically
- `epochs=3` (Stage A), `5` (Stage B)
- `batch_size=128`
- `learning_rate=0.001`
- `weight_decay=1e-5`
- `optimizer=adam`
- `scheduler=none`
- `early_stopping=false`
- `train_subset_size=8000` (A), `12000` (B)
- `val_subset_size=2000` (A), `3000` (B)
- `test_subset_size=5000` (A), `10000` (B)

E. Evaluation/analysis
- `n_seeds=1` (A metadata), Stage B analysis uses 3 seeds
- `primary_metric=val_accuracy` for promotion, `test_accuracy` for final paired checks
- `multiple_comparison_correction=fdr_bh`
- `promotion_rule=validation_only_staged`
- `compute_budget_tier=stage_a_screen` / `stage_b_confirmation`

## Implementation plan
1. Add growth-capable MLP with reproducible width/depth expansion and parameter-count logging.
2. Add trainer growth scheduler, growth event timeline, and memory reset handling after architecture changes.
3. Add gating and mechanistic diagnostics (strong-help/harm, gate statistics, retrieval top-1 agreement).
4. Add Stage B promotion script using validation-only rules and fixed anchors.
5. Run Stage A matrix, then Stage B promoted multi-seed runs.
6. Build consolidated tables + paired tests (+ p-value correction) and generate final report.

## Predefined critical causal checks
- `g5` vs `g6` (active vs random memory)
- `g5` vs `g7` (active vs inactive memory)
- `g5` vs `g4` (memory gain within growing architecture)
- `g5` vs fixed-memory best promoted (`g1` or `g3`)
- `g5` vs prior strong baselines (`g8`, `g9`)
- parameter-count contextualization for capacity effects
