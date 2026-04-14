# Phase 2 Backlog

EML Lab v1 is intentionally small. These are the next research directions once the
core optimizer, verifier, benchmark suite, and dashboard are stable.

## Multi-Agent Symbolic Orchestrator

- Proposer suggests route mutations or depth expansions.
- Evaluator runs the exact verifier and benchmark suite.
- Pruner simplifies snapped trees and rejects numerically fragile candidates.

This should wrap the real EML engine. It should not replace it.

## External Benchmarks

- Add PySR as an optional dependency.
- Compare exact recovery, runtime, expression size, and extrapolation error.
- Keep this optional so the quickstart stays small.

## Operator Zoo

- Search for EML cousins with better numerical behavior.
- Track whether candidates need a distinguished constant.
- Produce a small report per candidate.

## Hard Targets

- `x*y`
- `x/y`
- `x^2`
- `sin(x)`

These should be treated as research experiments, not v1 promises.

