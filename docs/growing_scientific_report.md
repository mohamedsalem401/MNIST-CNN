# Scientific Report: Growing NN + Hidden-State Memory (MNIST)

## Motivation
This stage evaluates whether progressive network growth makes hidden-state retrieval interventions more causally useful than fixed-size alternatives.

## Pre-registered hypotheses
- H1: Growing NN without memory can match/exceed fixed NN at similar effective capacity.
- H2: Growing NN with memory benefits more from retrieval than fixed NN with memory.
- H3: Memory usefulness depends on attachment layer in growing models.
- H4: Growth schedule and memory schedule interact; some combinations help, others add noise.
- H5: If growth improves memory usefulness, active memory should separate from random/inactive controls.
- H6: If not separated from controls, growth+memory is not yet a convincing causal improvement.

## Experimental protocol
- Stage A screening: single-seed matrix over required groups + optional stress variants.
- Stage B confirmation: promoted subset with seeds 11/22/33.
- Promotion criterion: validation accuracy only (see `results/growth_analysis/stage_policy.json`).
- Multiple-comparison correction in pairwise claims: `fdr_bh`.

## Run status
- Stage A successful runs: 13, failures: 0
- Stage B successful runs: 27, failures: 0

## Stage B main table
| Run | n | Val acc mean | Test acc mean +/- std | Test acc 95% CI | Params mean | Growth events | Helped | Harmed | Strong help | Strong harm | Gate mean | Retrieval top1 | Thr ratio | Memory MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g4_growing_no_memory | 3 | 0.9353 | 0.9367 +/- 0.0044 | [0.9258, 0.9476] | 83050 | 2.00 | NA | NA | NA | NA | NA | NA | NA | NA |
| g5_growing_with_memory | 3 | 0.9347 | 0.9364 +/- 0.0035 | [0.9276, 0.9452] | 83050 | 2.00 | 0.1789 | 0.1283 | 0.0000 | 0.0000 | 0.4000 | 0.9167 | 0.7337 | 12.7411 |
| g6_growing_random_memory_control | 3 | 0.9302 | 0.9358 +/- 0.0022 | [0.9303, 0.9413] | 83050 | 2.00 | 0.2099 | 0.2232 | 0.0000 | 0.0000 | 0.4000 | 0.1875 | 0.9904 | 12.7411 |
| g7_growing_inactive_memory_control | 3 | 0.9350 | 0.9358 +/- 0.0044 | [0.9250, 0.9466] | 83050 | 2.00 | 0.1792 | 0.1354 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5996 | 12.7411 |
| g8_prior_nn_large | 3 | 0.9570 | 0.9570 +/- 0.0042 | [0.9466, 0.9675] | 535818 | 0.00 | NA | NA | NA | NA | NA | NA | NA | NA |
| g9_prior_embedding_knn | 3 | 0.9606 | 0.9623 +/- 0.0027 | [0.9557, 0.9690] | 235146 | 0.00 | NA | NA | NA | NA | NA | NA | NA | NA |
| g2_fixed_large_no_memory_capacity_matched | 3 | 0.9453 | 0.9479 +/- 0.0018 | [0.9433, 0.9524] | 104266 | 0.00 | NA | NA | NA | NA | NA | NA | NA | NA |
| g3_fixed_large_with_memory_capacity_matched | 3 | 0.9402 | 0.9414 +/- 0.0072 | [0.9234, 0.9594] | 104266 | 0.00 | 0.1322 | 0.0976 | 0.0000 | 0.0000 | 0.4000 | 0.9167 | 0.8106 | 15.1825 |
| g12_growing_with_memory_uncertainty_gating | 3 | 0.9362 | 0.9369 +/- 0.0041 | [0.9268, 0.9470] | 83050 | 2.00 | 0.1119 | 0.1096 | 0.0000 | 0.0000 | 0.1127 | 0.9375 | 0.6877 | 12.7411 |

## Critical causal/statistical checks
| Question | A | B | n | Diff A-B | 95% CI | p | p adj |
|---|---|---|---|---|---|---|---|
| growing_mem_vs_growing_no_mem | g5_growing_with_memory | g4_growing_no_memory | 3 | -0.0003 | [-0.0154, 0.0148] | 0.9396 | 0.9396 |
| growing_mem_vs_random_control | g5_growing_with_memory | g6_growing_random_memory_control | 3 | 0.0006 | [-0.0120, 0.0131] | 0.8642 | 0.9396 |
| growing_mem_vs_inactive_control | g5_growing_with_memory | g7_growing_inactive_memory_control | 3 | 0.0006 | [-0.0021, 0.0034] | 0.4242 | 0.5939 |
| growing_mem_vs_prior_nn_large | g5_growing_with_memory | g8_prior_nn_large | 3 | -0.0206 | [-0.0279, -0.0134] | 0.0066 | 0.0332 |
| growing_mem_vs_prior_embedding_knn | g5_growing_with_memory | g9_prior_embedding_knn | 3 | -0.0259 | [-0.0369, -0.0150] | 0.0095 | 0.0332 |
| growing_mem_vs_best_fixed_mem | g5_growing_with_memory | g3_fixed_large_with_memory_capacity_matched | 3 | -0.0050 | [-0.0222, 0.0122] | 0.3371 | 0.5899 |
| growing_no_mem_vs_best_fixed_no_mem | g4_growing_no_memory | g2_fixed_large_no_memory_capacity_matched | 3 | -0.0112 | [-0.0254, 0.0031] | 0.0778 | 0.1815 |

## Hypothesis-by-hypothesis readout
- H1: g4 - g2_fixed_large_no_memory_capacity_matched = -0.0112
- H2: memory_gain_growing=-0.0003, memory_gain_fixed=-0.0065
- H3: Layer-placement evidence only from promoted optional variant if selected; otherwise unresolved.
- H4: Growth/memory interaction evidence is limited to compact schedule sweep in this compute tier.
- H5/H6: judged primarily by `g5` vs `g6` and `g7` pairwise tests.

## Evidence-backed conclusions
- Claims are limited to Stage B paired comparisons and their confidence intervals.
- If CIs overlap zero, no decisive separation is claimed.
- Growth effects are interpreted jointly with parameter count and control runs.

## Mechanistic diagnostics
- Reported diagnostics include helped/harmed and strong-help/strong-harm fractions, intervention magnitude, gate statistics, retrieval top-1 agreement, throughput ratio, training time, memory footprint, parameter count, growth events, and pre/post-growth validation deltas.

## Limitations
- CPU-only budget and staged promotion limit matrix-wide multi-seed certainty.
- Some optional mechanism variants may remain single-seed if not promoted.
- Conclusions are benchmark-local to MNIST; no external validity claim is made.

## Final recommendation
The growing-memory direction is **currently unsupported** under the current evidence.
- Decision rule: no success claim unless growing+memory beats strong baselines and memory controls with uncertainty excluding zero.
- If controls are not separated, prioritize intervention/target redesign before scaling compute.

## Artifact map
- Consolidated metrics: `results/growth_analysis/consolidated_metrics.csv`
- Pairwise tests: `results/growth_analysis/comparison_tests.csv`
- Promotion policy: `results/growth_analysis/stage_policy.json`
- Stage B promoted runs: `results/growth_analysis/stage_b_promoted_runs.json`
