"""Pruning helpers for local route-search candidates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class _CandidateLike(Protocol):
    rpn: str
    score: object
    tree: object


def dedupe_candidates(candidates: Iterable[_CandidateLike]) -> list[_CandidateLike]:
    best_by_rpn: dict[str, _CandidateLike] = {}
    for candidate in candidates:
        current = best_by_rpn.get(candidate.rpn)
        if current is None or _candidate_sort_key(candidate) < _candidate_sort_key(current):
            best_by_rpn[candidate.rpn] = candidate
    return list(best_by_rpn.values())


def prune_top(candidates: Iterable[_CandidateLike], beam_width: int) -> list[_CandidateLike]:
    deduped = dedupe_candidates(candidates)
    deduped.sort(key=_candidate_sort_key)
    return deduped[:beam_width]


def _candidate_sort_key(candidate: _CandidateLike) -> tuple[float, float, int, str]:
    score = candidate.score
    tree = candidate.tree
    return (
        float(score.total_score),
        float(score.max_mse),
        int(tree.node_count()),
        candidate.rpn,
    )
