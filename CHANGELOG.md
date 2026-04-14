# Changelog

## 0.1.0 - 2026-04-14

Initial EML Lab release.

### Added

- Faithful complex-valued `eml_exact(x, y) = exp(x) - log(y)` operator.
- Training-safe `eml_train` helper with explicit stabilization metadata.
- Differentiable `SoftEMLTree`, deterministic snapping, raw exact verification, and
  shallow benchmark artifacts.
- CLI and Streamlit app for training, snapping, benchmarking, paper fixtures, campaign
  runs, comparison artifacts, and research reports.
- Local proposer/evaluator/pruner orchestration loop for deterministic EML route search.
- Optional PySR comparison commands that degrade gracefully when PySR or Julia is absent.
- Cross-method comparison artifacts, saved-run analytics, snapshot reports, and snapshot
  history reports.
- Research-tier hard targets for `x^2`, `x*y`, `x/y`, and `sin(x)` with explicit failure
  reporting instead of shipped success claims.
- Operator zoo benchmark for EML-like variants, keeping the paper operator marked as the
  exact reference.
- Per-target research report bundles for saved hard-target campaigns.
- Streamlit Cloud and Docker packaging for local-first hosted-demo handoff.

### Verified

- `ruff check .`
- `pytest`
- CLI smoke: `python -m eml_lab research-report`
- Streamlit smoke: app dry-run plus local HTTP 200 boot check
- Docker smoke for the hosted Streamlit image
