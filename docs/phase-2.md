# Phase 2 Status

Detailed execution plan: [phase-2-plan.md](phase-2-plan.md)

EML Lab v1 is intentionally small. Phase 2 turns the repo into a local research
workbench around the same exact verifier, without claiming hard-target recovery as a
solved problem.

## Multi-Agent Symbolic Orchestrator

- Initial local proposer/evaluator/pruner loop is now implemented.
- Proposer suggests deterministic route mutations and optional depth expansions.
- Evaluator runs the exact verifier and candidate score model.
- Pruner deduplicates structurally equivalent trees and keeps the top beam.

This should wrap the real EML engine. It should not replace it.

## External Benchmarks

- PySR comparison command now exists as an optional baseline.
- Aggregated compare-suite execution now exists for stable compare-eligible targets.
- Cross-method comparison now exists for one target at a time, lining up gradient,
  agentic, and optional PySR results in one summary artifact.
- Saved cross-method runs can now be rediscovered from disk and reloaded in the app.
- Saved cross-method runs can now be aggregated into target-level analytics.
- The app now supports target/status/seed filtering plus artifact-backed charts over those analytics.
- Filtered saved-run analytics can now be exported as JSON/CSV bundles.
- Filtered saved-run analytics can now be packaged as snapshot bundles with markdown reports and PNG plots.
- Saved snapshot bundles can now be indexed into longer-horizon history reports with CSVs, markdown, and trend plots.
- Keep this optional so the quickstart stays small and Julia stays off the critical path.

## Operator Zoo

- Operator zoo benchmark/report now exists for EML-like numerical variants.
- It tracks finite output rate, safe-domain finite rate, gradient finite rate, exact-operator error, and stability score.
- It keeps the faithful paper operator explicitly marked as the exact reference.
- Stabilized variants are treated as training/research candidates, not verifier replacements.
- `phase2-operator-zoo` runs the zoo as a repeatable campaign artifact.

## Hosted Demo

- Streamlit Cloud packaging now exists through `runtime.txt`, `requirements.txt`, and `.streamlit/config.toml`.
- Docker packaging now exists through `Dockerfile` and `.dockerignore`.
- Deployment notes live in [hosted-demo.md](hosted-demo.md).
- No external deployment is claimed until a host/account is actually connected.

## Hard Targets

- Research-tier target specs now exist for `x*y`, `x/y`, `x^2`, and `sin(x)`.
- `phase2-research` runs them as non-required training experiments.
- Campaign artifacts now capture verifier output, target tier, expected depth, and
  known failure modes for each hard target.
- Per-target research report bundles now aggregate saved research campaigns into
  `summary.json`, `targets.csv`, `runs.csv`, `report.md`, and `manifest.json`.
- Reports include unrun research targets as `not_run`, keeping scope gaps explicit.

These should be treated as research experiments, not v1 promises.

## Final Polish

- The dashboard Campaigns tab can scan saved research campaigns and build the same
  report bundle as the CLI.
- Release notes live in [CHANGELOG.md](../CHANGELOG.md).
