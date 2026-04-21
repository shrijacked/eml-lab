from pathlib import Path

from eml_lab.agentic import OrchestratorConfig, RouteCandidate, run_orchestrator
from eml_lab.mutations import (
    deterministic_seed_mutations,
    route_to_tree,
)
from eml_lab.pruning import dedupe_candidates, prune_top
from eml_lab.scoring import score_tree
from eml_lab.targets import get_target
from eml_lab.trees import rpn_string


def test_route_to_tree_matches_known_ln_fixture() -> None:
    spec = get_target("ln")
    tree = route_to_tree(spec.known_route, spec.variables)

    assert rpn_string(tree) == "1 1 x E 1 E E"


def test_deterministic_seed_mutations_are_repeatable() -> None:
    spec = get_target("ln")

    first = deterministic_seed_mutations(spec.known_route, spec.variables, seed=7, count=4)
    second = deterministic_seed_mutations(spec.known_route, spec.variables, seed=7, count=4)

    assert [mutation.route for mutation in first] == [mutation.route for mutation in second]
    assert all(mutation.route != tuple(spec.known_route) for mutation in first)


def test_prune_top_deduplicates_by_rpn() -> None:
    spec = get_target("exp")
    route = tuple(spec.known_route)
    tree = route_to_tree(route, spec.variables)
    better = RouteCandidate(
        route=route,
        tree=tree,
        rpn=rpn_string(tree),
        score=score_tree(tree, spec),
        generation=0,
        mutation="seed-a",
    )
    worse = RouteCandidate(
        route=route,
        tree=tree,
        rpn=rpn_string(tree),
        score=score_tree(tree, spec, complexity_weight=1e-3),
        generation=0,
        mutation="seed-b",
    )

    deduped = dedupe_candidates([worse, better])
    pruned = prune_top([worse, better], beam_width=1)

    assert len(deduped) == 1
    assert pruned[0].mutation == "seed-a"


def test_orchestrator_recovers_ln_from_seed_mutations(tmp_path: Path) -> None:
    result = run_orchestrator(
        OrchestratorConfig(target="ln", seed=0, budget=32, beam_width=6, seed_count=4),
        tmp_path,
    )

    assert result.success
    assert result.initial_best_rpn != result.best_rpn
    assert result.best_rpn == "1 1 x E 1 E E"
    assert (Path(result.output_dir) / "summary.json").exists()
    assert (Path(result.output_dir) / "leaderboard.json").exists()
    assert (Path(result.output_dir) / "events.jsonl").exists()


def test_orchestrator_writes_generation_summary_events(tmp_path: Path) -> None:
    result = run_orchestrator(
        OrchestratorConfig(target="exp", seed=0, budget=12, beam_width=4, seed_count=3),
        tmp_path,
    )

    events = (Path(result.output_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines()
    summaries = [event for event in events if '"kind": "generation_summary"' in event]

    assert summaries
    assert '"best_rpn": "x 1 E"' in summaries[-1]
    assert '"kept_in_beam":' in summaries[-1]
