# Final Executive Summary

## Scope
Executed full Stage A screening (22 configs) and Stage B promoted multi-seed confirmation (8 configs x 3 seeds) in this repository.

## Strongest evidence-backed takeaways
- `a19_baseline_nn_large` remained a strong comparator; memory variants did not show a decisive win under this budget.
- `a3_hybrid_penultimate_residual_delta` and `a1_hybrid_penultimate_gated_delta` were close in Stage B.
- `a6_hybrid_penultimate_absolute_target` vs `a1` showed no decisive separation with n=3.
- Random/inactive controls reduce certainty that gains are purely due to useful retrieval in all settings.

## Confidence
- Multi-seed promoted-run findings: medium-to-low confidence (n=3, some wide CIs).
- Full ablation matrix conclusions: low confidence where only Stage A single-seed evidence exists.

## Unresolved questions
- Layer-placement ranking is Stage A single-seed only.
- Most forgetting-policy variants were not re-evaluated in Stage B multi-seed.
- No non-MNIST external validity evidence is available.

## Status of robustness
Results are **partially robust**: robust enough for promoted-run directional insights, but not fully robust for complete matrix-wide claims.

## Key artifact pointers
- Report: `docs/final_scientific_report.md`
- Metrics: `results/final_analysis/consolidated_metrics.csv`
- Comparisons: `results/final_analysis/comparison_tests.csv`
- Traceability map: `results/final_analysis/run_index.json`
