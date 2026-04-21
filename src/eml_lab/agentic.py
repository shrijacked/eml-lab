"""Deterministic proposer/evaluator/pruner loop for shallow EML search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eml_lab.artifacts import ArtifactFile, write_artifact_manifest
from eml_lab.mutations import (
    RouteMutation,
    RouteTuple,
    as_route_tuple,
    deterministic_seed_mutations,
    enumerate_depth_expansion_mutations,
    enumerate_single_edit_mutations,
    route_to_tree,
)
from eml_lab.pruning import dedupe_candidates, prune_top
from eml_lab.scoring import CandidateScore, score_tree
from eml_lab.targets import TargetSpec, get_target
from eml_lab.trees import TreeNode, rpn_string


@dataclass(frozen=True)
class OrchestratorConfig:
    target: str = "ln"
    seed: int = 0
    budget: int = 64
    beam_width: int = 8
    seed_count: int = 4
    max_depth: int | None = None
    points: int = 128
    tolerance: float = 1e-20
    complexity_weight: float = 1e-6
    failure_penalty: float = 1e6


@dataclass(frozen=True)
class RouteCandidate:
    route: RouteTuple
    tree: TreeNode
    rpn: str
    score: CandidateScore
    generation: int
    mutation: str
    parent_route: RouteTuple | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "route": [list(step) for step in self.route],
            "rpn": self.rpn,
            "generation": self.generation,
            "mutation": self.mutation,
            "parent_route": None
            if self.parent_route is None
            else [list(step) for step in self.parent_route],
            "score": self.score.to_dict(),
        }


@dataclass(frozen=True)
class OrchestratorResult:
    target: str
    output_dir: str
    manifest_path: str
    summary_path: str
    leaderboard_path: str
    events_path: str
    config: OrchestratorConfig
    success: bool
    initial_best_rpn: str
    best_rpn: str
    best_route: tuple[tuple[str, str], ...]
    best_score: dict[str, object]
    evaluated_candidates: int
    generations: int
    leaderboard: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "summary_path": self.summary_path,
            "leaderboard_path": self.leaderboard_path,
            "events_path": self.events_path,
            "config": {
                "target": self.config.target,
                "seed": self.config.seed,
                "budget": self.config.budget,
                "beam_width": self.config.beam_width,
                "seed_count": self.config.seed_count,
                "max_depth": self.config.max_depth,
                "points": self.config.points,
                "tolerance": self.config.tolerance,
                "complexity_weight": self.config.complexity_weight,
                "failure_penalty": self.config.failure_penalty,
            },
            "success": self.success,
            "initial_best_rpn": self.initial_best_rpn,
            "best_rpn": self.best_rpn,
            "best_route": [list(step) for step in self.best_route],
            "best_score": self.best_score,
            "evaluated_candidates": self.evaluated_candidates,
            "generations": self.generations,
            "leaderboard": list(self.leaderboard),
        }


def run_orchestrator(
    config: OrchestratorConfig, output_dir: str | Path = "runs"
) -> OrchestratorResult:
    spec = get_target(config.target)
    if spec.known_route is None:
        raise ValueError(f"Target {spec.name!r} has no known route for deterministic orchestration")

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    root = Path(output_dir) / f"orchestrate-{spec.name}-{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    events_path = root / "events.jsonl"
    leaderboard_path = root / "leaderboard.json"
    summary_path = root / "summary.json"

    max_depth = config.max_depth or len(spec.known_route)
    seed_mutations = deterministic_seed_mutations(
        spec.known_route,
        spec.variables,
        seed=config.seed,
        count=config.seed_count,
    )
    if not seed_mutations:
        seed_mutations = [RouteMutation(name="known-route", route=as_route_tuple(spec.known_route))]

    frontier = _evaluate_mutations(
        seed_mutations,
        spec,
        config,
        generation=0,
        events_path=events_path,
    )
    evaluated = len(frontier)
    frontier = prune_top(frontier, config.beam_width)
    best = prune_top(frontier, 1)[0]
    initial_best = best
    _append_generation_summary(
        events_path,
        generation=0,
        evaluated_candidates=len(seed_mutations),
        frontier=frontier,
        best=best,
        best_changed=True,
    )
    seen_routes = {candidate.route for candidate in frontier}
    generations = 0

    while evaluated < config.budget and frontier and not best.score.passed:
        generations += 1
        previous_best = best
        proposals: list[RouteMutation] = []
        for candidate in frontier:
            proposals.extend(
                enumerate_single_edit_mutations(candidate.route, spec.variables)
            )
            proposals.extend(
                enumerate_depth_expansion_mutations(
                    candidate.route,
                    spec.variables,
                    max_depth=max_depth,
                )
            )

        next_mutations: list[RouteMutation] = []
        for proposal in proposals:
            if proposal.route in seen_routes:
                continue
            seen_routes.add(proposal.route)
            next_mutations.append(proposal)
            if evaluated + len(next_mutations) >= config.budget:
                break
        if not next_mutations:
            break

        evaluated_candidates = _evaluate_mutations(
            next_mutations,
            spec,
            config,
            generation=generations,
            events_path=events_path,
        )
        evaluated += len(evaluated_candidates)
        combined = [*frontier, *evaluated_candidates, best]
        frontier = prune_top(combined, config.beam_width)
        best = prune_top([best, *frontier], 1)[0]
        _append_generation_summary(
            events_path,
            generation=generations,
            evaluated_candidates=len(next_mutations),
            frontier=frontier,
            best=best,
            best_changed=best.rpn != previous_best.rpn,
        )

    leaderboard_candidates = prune_top(dedupe_candidates([*frontier, best, initial_best]), 10)
    leaderboard = tuple(candidate.to_dict() for candidate in leaderboard_candidates)
    leaderboard_path.write_text(json.dumps(list(leaderboard), indent=2), encoding="utf-8")

    manifest = write_artifact_manifest(
        root,
        files=[
            ArtifactFile(label="summary", path=str(summary_path), kind="json"),
            ArtifactFile(label="leaderboard", path=str(leaderboard_path), kind="json"),
            ArtifactFile(label="events", path=str(events_path), kind="jsonl"),
        ],
        metadata={
            "kind": "orchestration",
            "target": spec.name,
            "success": best.score.passed,
            "evaluated_candidates": evaluated,
        },
    )
    result = OrchestratorResult(
        target=spec.name,
        output_dir=str(root),
        manifest_path=manifest.manifest_path,
        summary_path=str(summary_path),
        leaderboard_path=str(leaderboard_path),
        events_path=str(events_path),
        config=config,
        success=best.score.passed,
        initial_best_rpn=initial_best.rpn,
        best_rpn=best.rpn,
        best_route=best.route,
        best_score=best.score.to_dict(),
        evaluated_candidates=evaluated,
        generations=generations,
        leaderboard=leaderboard,
    )
    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _evaluate_mutations(
    mutations: list[RouteMutation],
    spec: TargetSpec,
    config: OrchestratorConfig,
    *,
    generation: int,
    events_path: Path,
) -> list[RouteCandidate]:
    candidates: list[RouteCandidate] = []
    with events_path.open("a", encoding="utf-8") as handle:
        for mutation in mutations:
            tree = route_to_tree(mutation.route, spec.variables)
            score = score_tree(
                tree,
                spec,
                points=config.points,
                tolerance=config.tolerance,
                complexity_weight=config.complexity_weight,
                failure_penalty_value=config.failure_penalty,
            )
            candidate = RouteCandidate(
                route=mutation.route,
                tree=tree,
                rpn=rpn_string(tree),
                score=score,
                generation=generation,
                mutation=mutation.name,
                parent_route=mutation.parent_route,
            )
            handle.write(json.dumps(_event_payload(candidate), default=str) + "\n")
            candidates.append(candidate)
    return candidates


def _event_payload(candidate: RouteCandidate) -> dict[str, object]:
    return {
        "kind": "candidate",
        "generation": candidate.generation,
        "mutation": candidate.mutation,
        "route": [list(step) for step in candidate.route],
        "parent_route": None
        if candidate.parent_route is None
        else [list(step) for step in candidate.parent_route],
        "rpn": candidate.rpn,
        "passed": candidate.score.passed,
        "total_score": candidate.score.total_score,
        "max_mse": candidate.score.max_mse,
        "failure_reason": candidate.score.failure_reason,
    }


def _append_generation_summary(
    events_path: Path,
    *,
    generation: int,
    evaluated_candidates: int,
    frontier: list[RouteCandidate],
    best: RouteCandidate,
    best_changed: bool,
) -> None:
    payload = {
        "kind": "generation_summary",
        "generation": generation,
        "evaluated_candidates": evaluated_candidates,
        "kept_in_beam": len(frontier),
        "passed_candidates": sum(1 for candidate in frontier if candidate.score.passed),
        "best_rpn": best.rpn,
        "best_total_score": best.score.total_score,
        "best_max_mse": best.score.max_mse,
        "best_passed": best.score.passed,
        "best_failure_reason": best.score.failure_reason,
        "best_changed": best_changed,
    }
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
