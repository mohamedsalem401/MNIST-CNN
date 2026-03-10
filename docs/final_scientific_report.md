# Final Scientific Report: MNIST Hybrid NN + KNN-Memory

## Abstract
This report summarizes a staged experimental study executed in this repository on March 8, 2026. Stage A screened 22 configurations (single seed, reduced budget), and Stage B promoted 8 configurations for multi-seed confirmation (3 seeds). Results are evidence-bound to saved artifacts under `results/final_analysis/`. Main findings are mixed: memory-enabled variants were competitive, but gains versus strong parametric baselines were small and uncertain under this budget.

## Executive summary
- We executed Stage A (22 runs) and Stage B (8 promoted runs x 3 seeds) with explicit validation-only promotion rules.
- Strongest promoted parametric baseline (`a19_baseline_nn_large`) remained highly competitive.
- Residual-mode memory (`a3`) was competitive with gated (`a1`), but uncertainty remains with n=3.
- Random-memory and inactive controls reduced confidence that gains come purely from retrieval quality in all settings.
- Findings are partially robust (multi-seed for promoted runs), not fully robust across the full ablation matrix.

## Experimental setup and reproducibility
- Runtime environment is captured in `results/final_analysis/repro_manifest.json`.
- CPU-only execution (`torch 2.10.0+cpu`, no CUDA).
- Stage policy and promotion rules: `results/final_analysis/stage_policy.json`.
- Aggregated metrics: `results/final_analysis/consolidated_metrics.csv`.
- Pairwise tests: `results/final_analysis/comparison_tests.csv`.

## Repository audit and any deviations from assumptions
- Assumed directories (`configs/`, `scripts/`, `results/`, `docs/`) were present.
- Actual runnable entrypoints and adapted paths are recorded in `results/final_analysis/repo_audit.json`.
- No source-model/training code was modified for this analysis.

## Baselines and controls
Baselines/controls included in Stage A: `nn`, `nn_large`, `nn_extra_layer`, `embedding_knn`, `hybrid_inactive`, `random_memory`.
Promoted Stage B controls: `a19_baseline_nn_large`, `a21_baseline_embedding_knn`, `a18_hybrid_inactive`, `a17_random_memory_control`.

## Main results table (with real numbers)
| Run | n seeds | Val acc mean | Test acc mean +/- std | Test acc 95% CI | Test loss mean | Test ECE mean | Helped frac mean | Harmed frac mean | Memory MB mean | Thr ratio mean |
|---|---|---|---|---|---|---|---|---|---|---|
| a17_random_memory_control | 3 | 0.9509 | 0.9530 +/- 0.0010 | [0.9504, 0.9556] | 0.1519 | 0.0048 | 0.1170 | 0.1139 | 20.0653 | 1.0020 |
| a18_hybrid_inactive | 3 | 0.9506 | 0.9529 +/- 0.0011 | [0.9502, 0.9555] | 0.1492 | 0.0058 | 0.1839 | 0.0771 | 20.0653 | 0.7687 |
| a19_baseline_nn_large | 3 | 0.9546 | 0.9571 +/- 0.0026 | [0.9508, 0.9635] | 0.1380 | 0.0064 | NA | NA | NA | NA |
| a1_hybrid_penultimate_gated_delta | 3 | 0.9507 | 0.9528 +/- 0.0007 | [0.9510, 0.9545] | 0.1504 | 0.0053 | 0.1840 | 0.0742 | 20.0653 | 0.7230 |
| a21_baseline_embedding_knn | 3 | 0.9591 | 0.9602 +/- 0.0011 | [0.9574, 0.9629] | NA | NA | NA | NA | NA | NA |
| a3_hybrid_penultimate_residual_delta | 3 | 0.9499 | 0.9522 +/- 0.0009 | [0.9500, 0.9545] | 0.1516 | 0.0057 | 0.2351 | 0.0781 | 20.0653 | 0.8365 |
| a6_hybrid_penultimate_absolute_target | 3 | 0.9518 | 0.9536 +/- 0.0010 | [0.9512, 0.9561] | 0.1518 | 0.0062 | 0.8586 | 0.1412 | 20.0653 | 0.6917 |
| a9_no_forgetting | 3 | 0.9507 | 0.9527 +/- 0.0008 | [0.9508, 0.9546] | 0.1504 | 0.0053 | 0.1883 | 0.0775 | 46.2221 | 0.4782 |

Source files: `results/final_analysis/stage_b_runs/seed11/*/metrics.json`, `results/final_analysis/stage_b_runs/seed22/*/metrics.json`, `results/final_analysis/stage_b_runs/seed33/*/metrics.json`

## Ablation results table
| Run | Method | Val acc | Test acc | Test loss | Test ECE | Helped frac | Harmed frac | Memory MB |
|---|---|---|---|---|---|---|---|---|
| a0_baseline_nn | nn | 0.9130 | 0.9208 | 0.2506 | 0.0091 | NA | NA | NA |
| a10_fifo_bounded | hybrid | 0.9245 | 0.9266 | 0.2345 | 0.0194 | 0.3648 | 0.1008 | 15.6670 |
| a11_reservoir_bounded | hybrid | 0.9135 | 0.9200 | 0.2513 | 0.0098 | 0.3100 | 0.1112 | 16.0522 |
| a12_usage_eviction | hybrid | 0.9245 | 0.9266 | 0.2345 | 0.0194 | 0.3648 | 0.1008 | 15.6670 |
| a13_helpfulness_eviction | hybrid | 0.9245 | 0.9266 | 0.2345 | 0.0194 | 0.3648 | 0.1008 | 15.6670 |
| a14_helpfulness_age_refresh | hybrid | 0.9245 | 0.9266 | 0.2345 | 0.0194 | 0.3648 | 0.1008 | 15.6670 |
| a15_train_only | hybrid | 0.9245 | 0.9266 | 0.2345 | 0.0194 | 0.3648 | 0.1008 | 15.6670 |
| a16_inference_only | hybrid | 0.9225 | 0.9256 | 0.2351 | 0.0185 | 0.3732 | 0.1010 | 15.6640 |
| a17_random_memory_control | random_memory | 0.9200 | 0.9270 | 0.2315 | 0.0189 | 0.2390 | 0.2592 | 15.6289 |
| a18_hybrid_inactive | hybrid_inactive | 0.9225 | 0.9256 | 0.2351 | 0.0185 | 0.3732 | 0.1010 | 15.6640 |
| a19_baseline_nn_large | nn_large | 0.9280 | 0.9322 | 0.2020 | 0.0143 | NA | NA | NA |
| a1_hybrid_penultimate_gated_delta | hybrid | 0.9245 | 0.9266 | 0.2345 | 0.0194 | 0.3648 | 0.1008 | 15.6670 |
| a20_baseline_nn_extra_layer | nn_extra_layer | 0.8955 | 0.9088 | 0.2887 | 0.0152 | NA | NA | NA |
| a21_baseline_embedding_knn | embedding_knn | 0.9315 | 0.9376 | NA | NA | NA | NA | NA |
| a2_hybrid_penultimate_overwrite_delta | hybrid | 0.9255 | 0.9268 | 0.2344 | 0.0193 | 0.4530 | 0.1034 | 15.6610 |
| a3_hybrid_penultimate_residual_delta | hybrid | 0.9255 | 0.9268 | 0.2344 | 0.0193 | 0.4530 | 0.1034 | 15.6610 |
| a4_hybrid_middle_gated_delta | hybrid | 0.9240 | 0.9264 | 0.2352 | 0.0195 | 0.3920 | 0.1016 | 12.6080 |
| a5_hybrid_early_gated_delta | hybrid | 0.9225 | 0.9254 | 0.2352 | 0.0198 | 0.3708 | 0.0652 | 25.3002 |
| a6_hybrid_penultimate_absolute_target | hybrid | 0.9250 | 0.9260 | 0.2411 | 0.0166 | 0.8208 | 0.1792 | 15.7021 |
| a7_hybrid_query_untouched | hybrid | 0.9235 | 0.9262 | 0.2352 | 0.0194 | 0.3604 | 0.1474 | 11.1424 |
| a8_hybrid_query_projection | hybrid | 0.9240 | 0.9258 | 0.2351 | 0.0182 | 0.3720 | 0.1286 | 8.8238 |
| a9_no_forgetting | hybrid | 0.9245 | 0.9266 | 0.2345 | 0.0194 | 0.3648 | 0.1008 | 15.6670 |

Source files: `results/final_analysis/stage_a_runs/*/metrics.json`

## Forgetting / refresh comparison
| Run | Stage | n | test_accuracy mean | std |
|---|---|---|---|---|
| a9_no_forgetting | stage_b | 3 | 0.9527 | 0.0008 |
| a10_fifo_bounded | stage_a | 1 | 0.9266 | NA |
| a11_reservoir_bounded | stage_a | 1 | 0.9200 | NA |
| a12_usage_eviction | stage_a | 1 | 0.9266 | NA |
| a13_helpfulness_eviction | stage_a | 1 | 0.9266 | NA |
| a14_helpfulness_age_refresh | stage_a | 1 | 0.9266 | NA |

Source files: `results/final_analysis/stage_a_runs/a9_no_forgetting/metrics.json`, `results/final_analysis/stage_a_runs/a10_fifo_bounded/metrics.json`, `results/final_analysis/stage_a_runs/a11_reservoir_bounded/metrics.json`, `results/final_analysis/stage_a_runs/a12_usage_eviction/metrics.json`, `results/final_analysis/stage_a_runs/a13_helpfulness_eviction/metrics.json`, `results/final_analysis/stage_a_runs/a14_helpfulness_age_refresh/metrics.json`, `results/final_analysis/stage_b_runs/seed*/a9_no_forgetting/metrics.json`

Interpretation: Stage A suggests limited separation across forgetting policies at this budget; Stage B includes only `a9` among forgetting variants, so policy-level conclusions remain provisional.

## Layer placement analysis
| Run | Stage | n | test_accuracy mean | std |
|---|---|---|---|---|
| a5_hybrid_early_gated_delta | stage_a | 1 | 0.9254 | NA |
| a4_hybrid_middle_gated_delta | stage_a | 1 | 0.9264 | NA |
| a1_hybrid_penultimate_gated_delta | stage_b | 3 | 0.9528 | 0.0007 |

Source files: `results/final_analysis/stage_a_runs/a5_hybrid_early_gated_delta/metrics.json`, `results/final_analysis/stage_a_runs/a4_hybrid_middle_gated_delta/metrics.json`, `results/final_analysis/stage_a_runs/a1_hybrid_penultimate_gated_delta/metrics.json`

Interpretation: layer effects are single-seed (Stage A only), so insufficient evidence for robust ranking.

## Intervention mode analysis
| Run | Stage | n | test_accuracy mean | std |
|---|---|---|---|---|
| a2_hybrid_penultimate_overwrite_delta | stage_a | 1 | 0.9268 | NA |
| a3_hybrid_penultimate_residual_delta | stage_b | 3 | 0.9522 | 0.0009 |
| a1_hybrid_penultimate_gated_delta | stage_b | 3 | 0.9528 | 0.0007 |

Source files: `results/final_analysis/stage_a_runs/a2_hybrid_penultimate_overwrite_delta/metrics.json`, `results/final_analysis/stage_a_runs/a3_hybrid_penultimate_residual_delta/metrics.json`, `results/final_analysis/stage_a_runs/a1_hybrid_penultimate_gated_delta/metrics.json`, `results/final_analysis/stage_b_runs/seed*/a3_hybrid_penultimate_residual_delta/metrics.json`, `results/final_analysis/stage_b_runs/seed*/a1_hybrid_penultimate_gated_delta/metrics.json`

Interpretation: residual vs gated has multi-seed evidence (n=3), overwrite only single-seed evidence.

## Target type analysis
| Run | Stage | n | test_accuracy mean | std |
|---|---|---|---|---|
| a6_hybrid_penultimate_absolute_target | stage_b | 3 | 0.9536 | 0.0010 |
| a1_hybrid_penultimate_gated_delta | stage_b | 3 | 0.9528 | 0.0007 |

Source files: `results/final_analysis/stage_a_runs/a6_hybrid_penultimate_absolute_target/metrics.json`, `results/final_analysis/stage_a_runs/a1_hybrid_penultimate_gated_delta/metrics.json`, `results/final_analysis/stage_b_runs/seed*/a6_hybrid_penultimate_absolute_target/metrics.json`, `results/final_analysis/stage_b_runs/seed*/a1_hybrid_penultimate_gated_delta/metrics.json`

## Helped / harmed and sample-level diagnostics
| Run | n | Helped frac mean | Harmed frac mean | Loss delta mean | Retrieval dist mean | Intervention mag mean |
|---|---|---|---|---|---|---|
| a17_random_memory_control | 3 | 0.1170 | 0.1139 | -0.0000 | 0.0000 | 0.0000 |
| a18_hybrid_inactive | 3 | 0.1839 | 0.0771 | -0.0000 | NA | 0.0000 |
| a1_hybrid_penultimate_gated_delta | 3 | 0.1840 | 0.0742 | -0.0000 | 0.3194 | 0.0000 |
| a3_hybrid_penultimate_residual_delta | 3 | 0.2351 | 0.0781 | -0.0000 | 0.3218 | 0.0000 |
| a6_hybrid_penultimate_absolute_target | 3 | 0.8586 | 0.1412 | -0.0024 | 0.3344 | 1.6037 |
| a9_no_forgetting | 3 | 0.1883 | 0.0775 | -0.0000 | 0.3172 | 0.0000 |

Source files: `results/final_analysis/stage_b_runs/seed*/a*/metrics.json`, `results/final_analysis/stage_b_runs/seed*/a1_hybrid_penultimate_gated_delta/details.pt`

Additional diagnostics for `a1_hybrid_penultimate_gated_delta` pooled over Stage B seeds:
- pooled samples: 30000
- loss delta mean: -0.0000
- loss delta p10/p90: -0.0000 / 0.0000
- strong help fraction (delta < -0.2): 0.0000
- strong harm fraction (delta > 0.2): 0.0000
- corr(retrieval distance, benefit): 0.1036

## Compute and memory overhead analysis
| Run | n | Test thr | With mem thr | Without mem thr | Thr ratio | Memory MB |
|---|---|---|---|---|---|---|
| a17_random_memory_control | 3 | 4427.5428 | 4844.9488 | 4924.4589 | 1.0020 | 20.0653 |
| a18_hybrid_inactive | 3 | 4665.6571 | 3450.2232 | 4528.0721 | 0.7687 | 20.0653 |
| a19_baseline_nn_large | 3 | 4010.2463 | NA | NA | NA | NA |
| a1_hybrid_penultimate_gated_delta | 3 | 3471.9544 | 3173.8573 | 4407.9370 | 0.7230 | 20.0653 |
| a21_baseline_embedding_knn | 3 | NA | NA | NA | NA | NA |
| a3_hybrid_penultimate_residual_delta | 3 | 3796.1359 | 4036.0340 | 4821.7559 | 0.8365 | 20.0653 |
| a6_hybrid_penultimate_absolute_target | 3 | 2917.9869 | 3061.5106 | 4536.5275 | 0.6917 | 20.0653 |
| a9_no_forgetting | 3 | 2156.0778 | 2085.2166 | 4394.2931 | 0.4782 | 46.2221 |

Source files: `results/final_analysis/stage_b_runs/seed*/a*/metrics.json`

## Failure analysis
- Failed runs in Stage A/Stage B: 0
- Malformed artifacts detected: 0
- Logged issues are recorded in `results/final_analysis/final_qc_checklist.md`.

Negative/contradictory outcomes noted:
- Random-memory control (`a17`) remained competitive in some metrics, reducing confidence that all gains are due to semantically correct retrieval.
- Inactive-memory control (`a18`) performed close to active variants in some seeds.
- Several conclusions rely on single-seed Stage A comparisons (layer and some mode/forgetting settings).

## Threats to validity
- Compute-limited staged budget (reduced samples/epochs relative to original full config intent).
- Only promoted subset has multi-seed robustness.
- Multiple comparisons across many ablations increase false-positive risk.
- CPU-only runtime may alter throughput/overhead conclusions compared to GPU environments.
- Embedding-KNN baseline reports `NA` for loss/ECE by design in this codebase.

## Evidence-backed conclusions
### 1) Does memory intervention improve test metrics versus strong baselines?
- Evidence: Stage B `a1` vs strongest promoted parametric baseline `a19` gives test accuracy diff = -0.0044, 95% CI [-0.0119, 0.0031], n=3.
- Inference: no evidence of improvement over the strongest promoted parametric baseline under this budget.
- Confidence: low.
- Caveats: only one strong parametric baseline was promoted to Stage B.

### 2) Which intervention mode works best (overwrite vs residual vs gated)?
- Evidence: Stage A (single seed) gives overwrite/residual 0.9268 vs gated 0.9266 test accuracy; Stage B (multi-seed) compares residual vs gated with diff = -0.0005, 95% CI [-0.0015, 0.0004].
- Inference: insufficient evidence for a clear overall winner; residual and gated are effectively close.
- Confidence: low.
- Caveats: overwrite mode lacks Stage B replication.

### 3) Which target type works best (absolute vs delta)?
- Evidence: Stage B absolute (`a6`) vs delta (`a1`) diff = 0.0009, 95% CI [-0.0003, 0.0020], n=3.
- Inference: no decisive winner under current budget.
- Confidence: low.
- Caveats: effect size is small and CI crosses zero.

### 4) Which intervention layer works best?
- Evidence: Stage A test accuracy: penultimate 0.9266, middle 0.9264, early 0.9254.
- Inference: penultimate appears slightly better in the single-seed screen.
- Confidence: low (single-seed only).
- Caveats: no Stage B multi-seed replication for layer sweep.

### 5) Does memory help during training, inference, or both?
- Evidence: Stage A test accuracy: both (`a1`) = 0.9266, train-only (`a15`) = 0.9266, inference-only (`a16`) = 0.9256.
- Inference: training-time use appears at least as important as inference-only use under this run.
- Confidence: low (single-seed only).
- Caveats: differences are small and unreplicated.

### 6) Which forgetting/refresh policy is best under staleness?
- Evidence: Stage A test accuracy: `a10/a12/a13/a14` all at 0.9266, `a9` at 0.9266, `a11` at 0.9200; Stage B includes only `a9`.
- Inference: insufficient evidence to rank forgetting policies robustly; reservoir underperformed in Stage A.
- Confidence: low.
- Caveats: policy comparison is mostly single-seed and not fully replicated.

### 7) Are gains robust across seeds?
- Evidence: promoted Stage B runs have small within-run std (about 0.0007-0.0026 test accuracy), but key paired differences vs baselines/modes have CIs overlapping zero.
- Inference: seed-to-seed variance is modest, but claimed gains are not robustly separated from zero for key comparisons.
- Confidence: medium for variance estimate, low for gain claims.
- Caveats: only promoted subset has multi-seed evidence.

### 8) Does the random-memory control remove apparent gains?
- Evidence: Stage B `a1` vs `a17` diff = -0.0003, 95% CI [-0.0046, 0.0041], n=3.
- Inference: random-memory control largely removes clear separation between active memory and control.
- Confidence: low.
- Caveats: attribution still requires stronger causal probes.

### 9) Is there evidence of memorization-dominant behavior versus generalization?
- Evidence: embedding-KNN baseline (`a21`) is strongest in Stage B test accuracy (0.9602 mean), and random-memory remains competitive with active memory.
- Inference: results are consistent with strong memorization/retrieval components in this benchmark.
- Confidence: low-to-medium.
- Caveats: no external dataset or shift test here, so generalization claims remain unresolved.

### 10) Do gains remain against the strongest parametric baselines?
- Evidence: Stage B `a1` underperforms `a19` on mean test accuracy by 0.0044 and has higher mean test loss by 0.0124, with CIs crossing zero.
- Inference: no evidence that gains remain against the strongest promoted parametric baseline.
- Confidence: low.
- Caveats: this is budget-limited and promotion-limited, not full-matrix multi-seed.

## Open questions and next experiments
1. Run full-matrix multi-seed replication (all 22 configs, n>=3) with longer epochs to convert Stage A provisional findings into robust findings.
2. Add non-MNIST dataset to test generalization beyond memorization-friendly regime.
3. Evaluate memory-size sweeps (e.g., 5k/10k/20k/40k) to test sensitivity and staleness effects.
4. Add explicit staleness stress tests (distribution shift and delayed refresh scenarios).
5. Include stronger statistical controls for multiple comparisons when matrix-wide claims are made.

## Appendix
### Config list
- `configs/base_hybrid.yaml`
- `configs/ablation_matrix.yaml`
- `results/final_analysis/configs/base_stage_a.yaml`
- `results/final_analysis/configs/ablation_matrix_stage_a.yaml`
- `results/final_analysis/configs/base_stage_b_seed11.yaml`
- `results/final_analysis/configs/base_stage_b_seed22.yaml`
- `results/final_analysis/configs/base_stage_b_seed33.yaml`
- `results/final_analysis/configs/ablation_matrix_stage_b_seed11.yaml`
- `results/final_analysis/configs/ablation_matrix_stage_b_seed22.yaml`
- `results/final_analysis/configs/ablation_matrix_stage_b_seed33.yaml`

### Command list
- `. .venv/bin/activate && python scripts/run_experiment.py --config configs/base_hybrid.yaml --override seeds=[101] seed=101 optim.epochs=1 data.max_train_samples=2000 data.max_val_samples=1000 data.max_test_samples=1000 logging.output_root=results/final_analysis/smoke logging.experiment_name=phase2_smoke logging.save_checkpoints=false`
- `. .venv/bin/activate && python scripts/analyze_results.py --experiment-dir results/final_analysis/smoke/phase2_smoke_seed101`
- `. .venv/bin/activate && python scripts/run_matrix.py --base-config results/final_analysis/configs/base_stage_a.yaml --matrix results/final_analysis/configs/ablation_matrix_stage_a.yaml`
- `. .venv/bin/activate && python scripts/run_matrix.py --base-config results/final_analysis/configs/base_stage_b_seed11.yaml --matrix results/final_analysis/configs/ablation_matrix_stage_b_seed11.yaml`
- `. .venv/bin/activate && python scripts/run_matrix.py --base-config results/final_analysis/configs/base_stage_b_seed22.yaml --matrix results/final_analysis/configs/ablation_matrix_stage_b_seed22.yaml`
- `. .venv/bin/activate && python scripts/run_matrix.py --base-config results/final_analysis/configs/base_stage_b_seed33.yaml --matrix results/final_analysis/configs/ablation_matrix_stage_b_seed33.yaml`
- `. .venv/bin/activate && python scripts/build_final_analysis.py`

### Artifact paths
- Consolidated metrics: `results/final_analysis/consolidated_metrics.csv`
- Pairwise comparisons: `results/final_analysis/comparison_tests.csv`
- Run index: `results/final_analysis/run_index.json`
- Figures: `results/final_analysis/figures/*.png`
- Stage A screen summary: `results/final_analysis/stage_a_matrix/matrix_summary.json`
- Stage B summaries: `results/final_analysis/stage_b_matrix/seed*/matrix_summary.json`

### Failed/skipped run list
- Failed runs: 0
- Skipped for Stage B (not promoted): all Stage A configs not in `results/final_analysis/stage_b_promoted_runs.json`.

### Promotion rules for staged evaluation
Promotion rules are recorded in `results/final_analysis/stage_policy.json` and applied results are recorded in `results/final_analysis/stage_b_promoted_runs.json`.
