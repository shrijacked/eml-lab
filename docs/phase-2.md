# Phase 2 Backlog

Detailed execution plan: [docs/phase-2-plan.md](/Users/owlxshri/Desktop/personal%20projects/paper_viz/docs/phase-2-plan.md)

EML Lab v1 is intentionally small. These are the next research directions once the
core optimizer, verifier, benchmark suite, and dashboard are stable.

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

- Search for EML cousins with better numerical behavior.
- Track whether candidates need a distinguished constant.
- Produce a small report per candidate.

## Hard Targets

- Research-tier target specs now exist for `x*y`, `x/y`, `x^2`, and `sin(x)`.
- `phase2-research` runs them as non-required training experiments.
- Campaign artifacts now capture verifier output, target tier, expected depth, and
  known failure modes for each hard target.

These should be treated as research experiments, not v1 promises.
