# [Title]

## 1. Abstract
- Problem:
- Method:
- Main empirical findings:
- Main limitations:

## 2. Executive Summary
- What was tested:
- What worked:
- What failed:
- What is still uncertain:

## 3. Problem Statement
- Task and scope (MNIST classification).
- Why hybrid NN + retrieval-memory at hidden states.

## 4. Motivation and Intuition
- Why nonparametric memory may help.
- Why it may fail (staleness, mismatch, memorization).

## 5. Research Questions and Hypotheses
- RQ1...
- RQ2...
- H1...
- H2...

## 6. Method Overview
- Base model(s):
- Memory intervention:
- Training/inference usage modes:

## 7. Model and Memory Design
### 7.1 Base architectures
- MLP details
- CNN details

### 7.2 Intervention point and dimensions
- Candidate layer set:
- Affected dimensions:
- Query representation:

### 7.3 Retrieval
- Distance metric:
- Top-k:
- Weighting:

### 7.4 Value construction
- Absolute targets
- Delta targets
- Gradient-derived variants

### 7.5 Memory management
- Insertion
- Forgetting
- Refresh

## 8. Experimental Design
- Split protocol and leakage controls.
- Hyperparameter fairness budget.
- Seed protocol.

## 9. Baselines and Controls
- Plain NN.
- Larger NN.
- Extra parametric layer.
- Embedding KNN.
- Memory inactive control.
- Random memory control.

## 10. Main Results
- Table: val/test loss, acc, ECE.
- Multi-seed mean ± std.
- Compute overhead and memory footprint.

## 11. Ablation Results
- Layer position.
- Intervention mode.
- Query mode.
- Target construction.

## 12. Forgetting vs Refreshing Comparison
- Policy-by-policy comparison.
- Staleness sensitivity.

## 13. Effect of KNN Placement / Layer Choice
- Early vs middle vs late vs penultimate.

## 14. Overwrite vs Residual vs Gated
- Stability and performance tradeoffs.

## 15. Target Type / Delta Type
- `absolute` vs `delta`.
- `grad_target` vs `grad_delta` vs alternatives.

## 16. Failure Analysis
- Helped vs harmed sample analysis.
- Retrieval purity/distance relation to benefit.
- Corruption cases and failure clusters.

## 17. Interpretation: What Memory Is Doing
- Evidence-backed interpretation.
- Speculative interpretation (clearly marked).

## 18. Threats to Validity / Limitations
- Hidden target non-uniqueness.
- Representation drift and staleness.
- Potential memorization effects.
- External validity beyond MNIST.

## 19. Conclusion
- What is supported.
- What is not supported.

## 20. Open Problems and Future Work
- Better target definitions.
- Differentiable retrieval approximations.
- Robustness and OOD analysis.
- Scaling to richer datasets.

## 21. Reproducibility Appendix
- Exact config files.
- Commit hash.
- Seeds.
- Hardware/software environment.

## 22. Additional Plots and Diagnostics
- Confusion matrices.
- Per-class metrics.
- Memory age/usefulness stats.
- Latent projections.
