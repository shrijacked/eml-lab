"""Streamlit dashboard for EML Lab."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from eml_lab.agentic import OrchestratorConfig, OrchestratorResult, run_orchestrator
from eml_lab.benchmarks import benchmark_table, run_benchmark_suite
from eml_lab.campaigns import list_campaigns, run_campaign
from eml_lab.comparison import (
    ComparisonResult,
    ComparisonSuiteResult,
    MethodComparisonResult,
    detect_pysr_environment,
    run_method_comparison,
    run_pysr_compare_suite,
    run_pysr_comparison,
)
from eml_lab.mutations import route_to_tree
from eml_lab.targets import PAPER_FIXTURES, get_target, list_targets
from eml_lab.training import TrainConfig, train_target
from eml_lab.trees import rpn_string
from eml_lab.visualize import tree_figure


def _orchestratable_targets() -> list[str]:
    return [name for name in list_targets() if get_target(name).known_route is not None]


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _method_compare_targets() -> list[str]:
    return [
        name
        for name in list_targets(comparison_eligible=True)
        if get_target(name).known_route is not None
    ]


def _comparison_rows(result: ComparisonResult) -> list[dict[str, object]]:
    return [
        {
            "model": "EML",
            "status": "ok" if result.eml["success"] else "failed",
            "rpn_or_equation": result.eml["rpn"],
            "max_mse": result.eml["verification"]["max_mse"],
        },
        {
            "model": "PySR",
            "status": result.pysr.get("status"),
            "rpn_or_equation": result.pysr.get("best_equation", result.pysr.get("reason")),
            "max_mse": None,
        },
    ]


def _comparison_suite_rows(result: ComparisonSuiteResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in result.runs:
        rows.append(
            {
                "target": run.target,
                "status": run.status,
                "eml_success": run.eml_success,
                "eml_rpn": run.eml_rpn,
                "eml_max_mse": run.eml_max_mse,
                "pysr_best_equation": run.pysr_best_equation,
                "pysr_reason": run.pysr_reason,
            }
        )
    return rows


def _method_comparison_rows(result: MethodComparisonResult) -> list[dict[str, object]]:
    return [
        {
            "method": "gradient",
            "status": result.gradient.get("status"),
            "expression": result.gradient.get("rpn"),
            "max_mse": result.gradient.get("verification", {}).get("max_mse"),
            "notes": result.gradient.get("verification", {}).get("failure_reason"),
        },
        {
            "method": "agentic",
            "status": result.agentic.get("status"),
            "expression": result.agentic.get("best_rpn"),
            "max_mse": result.agentic.get("max_mse"),
            "notes": result.agentic.get("best_score", {}).get("failure_reason"),
        },
        {
            "method": "pysr",
            "status": result.pysr.get("status"),
            "expression": result.pysr.get("best_equation"),
            "max_mse": None,
            "notes": result.pysr.get("reason", result.pysr.get("install_hint")),
        },
    ]


def _orchestrator_leaderboard_rows(result: OrchestratorResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, entry in enumerate(result.leaderboard, start=1):
        score = entry["score"]
        rows.append(
            {
                "rank": index,
                "generation": entry["generation"],
                "mutation": entry["mutation"],
                "rpn": entry["rpn"],
                "passed": score["passed"],
                "total_score": score["total_score"],
                "max_mse": score["max_mse"],
            }
        )
    return rows


def _campaign_rows(campaign_result: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in campaign_result.runs:
        metrics = run.metrics
        rows.append(
            {
                "name": run.name,
                "kind": run.kind,
                "target": metrics.get("target", metrics.get("config", {}).get("target")),
                "tier": metrics.get("target_tier", "stable"),
                "status": run.status,
                "failure_reason": metrics.get(
                    "failure_reason",
                    metrics.get("verification", {}).get("failure_reason"),
                ),
                "required": run.required,
                "effective_success": run.effective_success,
                "output_dir": run.output_dir,
            }
        )
    return rows


def main() -> None:
    st.set_page_config(page_title="EML Lab", layout="wide")
    st.title("EML Lab")
    st.caption("Differentiable formula discovery with one complex-valued binary operator.")

    train_tab, snap_tab, bench_tab, compare_tab, orchestrate_tab, campaign_tab, paper_tab = st.tabs(
        ["Train", "Snap", "Bench", "Compare", "Orchestrate", "Campaigns", "Paper Explorer"]
    )

    with train_tab:
        target_name = st.selectbox("Target", list_targets(), index=list_targets().index("ln"))
        spec = get_target(target_name)
        depth = st.number_input("Depth", min_value=1, max_value=5, value=spec.default_depth)
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=0)
        steps = st.slider("Steps", min_value=20, max_value=600, value=180, step=20)
        learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=0.2, value=0.03)
        snap_strategy = st.selectbox("Snap strategy", ["best_discrete", "logits"])
        init_choices = ["random", "known_route"] if spec.known_route is not None else ["random"]
        init_strategy = st.selectbox("Init strategy", init_choices)
        st.caption(f"Tier: `{spec.tier}`")
        if spec.failure_modes:
            st.caption("Known failure modes: " + "; ".join(spec.failure_modes))
        st.write(spec.notes)
        if st.button("Train and snap", type="primary"):
            result = train_target(
                TrainConfig(
                    target=target_name,
                    depth=int(depth),
                    seed=int(seed),
                    steps=int(steps),
                    learning_rate=float(learning_rate),
                    snap_strategy=snap_strategy,
                    init_strategy=init_strategy,
                )
            )
            st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    with snap_tab:
        if result is None:
            st.info("Run a training job first.")
        else:
            st.metric("Verifier", "passed" if result.success else "failed")
            st.metric("Max MSE", f"{result.verification.max_mse:.3e}")
            st.metric("Snap source", result.snap_source)
            st.code(result.rpn)
            st.pyplot(tree_figure(result.tree))
            st.subheader("Loss")
            st.line_chart({"loss": list(result.losses)})
            st.subheader("Logit probabilities")
            st.dataframe(list(result.logits_table), use_container_width=True)
            st.subheader("Verification")
            st.json(result.verification.to_dict())

    with bench_tab:
        if st.button("Run shallow benchmark"):
            bench = run_benchmark_suite("shallow")
            st.session_state["last_bench"] = bench
        bench = st.session_state.get("last_bench")
        if bench is None:
            st.info("Run the shallow suite to see recovery rate and run artifacts.")
        else:
            st.metric("Recovery rate", f"{bench.recovery_rate:.0%}")
            st.write(f"Artifacts: `{bench.output_dir}`")
            st.dataframe(benchmark_table(bench), use_container_width=True)

    with compare_tab:
        status = detect_pysr_environment()
        st.subheader("Optional PySR baseline")
        if status.available:
            st.success("PySR and Julia are available.")
        else:
            st.warning(status.reason or "PySR baseline unavailable.")
            st.code(status.install_hint)
            st.caption(
                "You can still run the EML baseline; "
                "the result will include the PySR install guidance."
            )
        compare_target = st.selectbox(
            "Comparison target",
            list_targets(comparison_eligible=True),
            index=list_targets(comparison_eligible=True).index("ln"),
            key="compare_target",
        )
        method_compare_target = st.selectbox(
            "Cross-method target",
            _method_compare_targets(),
            index=_method_compare_targets().index("ln") if "ln" in _method_compare_targets() else 0,
            key="method_compare_target",
        )
        if st.button("Run comparison", key="run_single_compare"):
            comparison = run_pysr_comparison(compare_target)
            st.session_state["last_comparison"] = comparison
        if st.button("Run compare suite", key="run_compare_suite"):
            compare_suite = run_pysr_compare_suite("shallow")
            st.session_state["last_compare_suite"] = compare_suite
        if st.button("Run cross-method comparison", key="run_method_compare"):
            method_comparison = run_method_comparison(method_compare_target)
            st.session_state["last_method_comparison"] = method_comparison
        comparison = st.session_state.get("last_comparison")
        compare_suite = st.session_state.get("last_compare_suite")
        method_comparison = st.session_state.get("last_method_comparison")
        if comparison is None:
            st.info("Run a comparison to capture the EML baseline and optional PySR result.")
        else:
            st.subheader("Single target result")
            st.dataframe(_comparison_rows(comparison), use_container_width=True)
            st.json(comparison.to_dict())
        if method_comparison is None:
            st.info(
                "Run the cross-method comparison to line up gradient, agentic, and PySR "
                "results on one target."
            )
        else:
            st.subheader("Cross-method comparison")
            st.metric(
                "Required methods",
                "passed" if method_comparison.required_success else "failed",
            )
            st.metric("PySR status", method_comparison.status)
            st.write(f"Artifacts: `{method_comparison.output_dir}`")
            st.dataframe(_method_comparison_rows(method_comparison), use_container_width=True)
            st.json(method_comparison.to_dict())
        if compare_suite is None:
            st.info(
                "Run the compare suite to aggregate PySR baseline results across stable targets."
            )
        else:
            st.subheader("Compare suite")
            st.metric("PySR success rate", f"{compare_suite.pysr_success_rate:.0%}")
            st.write(f"Artifacts: `{compare_suite.output_dir}`")
            st.dataframe(_comparison_suite_rows(compare_suite), use_container_width=True)
            st.json(compare_suite.to_dict())

    with orchestrate_tab:
        orchestrate_targets = _orchestratable_targets()
        target_name = st.selectbox(
            "Orchestration target",
            orchestrate_targets,
            index=orchestrate_targets.index("ln") if "ln" in orchestrate_targets else 0,
            key="orchestrate_target",
        )
        budget = st.slider("Budget", min_value=4, max_value=96, value=24, step=4)
        beam_width = st.slider("Beam width", min_value=1, max_value=16, value=6, step=1)
        seed_count = st.slider("Seed count", min_value=1, max_value=8, value=4, step=1)
        seed = st.number_input("Orchestrator seed", min_value=0, max_value=10_000, value=0)
        if st.button("Run orchestrator"):
            orchestrator_result = run_orchestrator(
                OrchestratorConfig(
                    target=target_name,
                    budget=int(budget),
                    beam_width=int(beam_width),
                    seed_count=int(seed_count),
                    seed=int(seed),
                )
            )
            st.session_state["last_orchestrator"] = orchestrator_result
        orchestrator_result = st.session_state.get("last_orchestrator")
        if orchestrator_result is None:
            st.info("Run the local route-search loop to see leaderboards and event traces.")
        else:
            spec = get_target(orchestrator_result.target)
            best_tree = route_to_tree(orchestrator_result.best_route, spec.variables)
            st.metric("Verifier", "passed" if orchestrator_result.success else "failed")
            st.metric("Candidates", orchestrator_result.evaluated_candidates)
            st.metric("Generations", orchestrator_result.generations)
            st.code(orchestrator_result.best_rpn)
            st.caption(f"Started from `{orchestrator_result.initial_best_rpn}`")
            st.pyplot(tree_figure(best_tree))
            st.subheader("Leaderboard")
            st.dataframe(
                _orchestrator_leaderboard_rows(orchestrator_result),
                use_container_width=True,
            )
            st.subheader("Event log")
            st.dataframe(_read_jsonl(orchestrator_result.events_path), use_container_width=True)
            st.subheader("Summary")
            st.json(orchestrator_result.to_dict())

    with campaign_tab:
        suite_name = st.selectbox(
            "Campaign suite",
            list_campaigns(),
            index=list_campaigns().index("phase2-agentic")
            if "phase2-agentic" in list_campaigns()
            else 0,
        )
        if st.button("Run campaign"):
            campaign_result = run_campaign(suite_name)
            st.session_state["last_campaign"] = campaign_result
        campaign_result = st.session_state.get("last_campaign")
        if campaign_result is None:
            st.info("Run a campaign to aggregate benchmark, comparison, and orchestration runs.")
        else:
            st.metric("Campaign", campaign_result.suite)
            st.metric("Success", "passed" if campaign_result.success else "failed")
            st.write(f"Artifacts: `{campaign_result.output_dir}`")
            st.dataframe(_campaign_rows(campaign_result), use_container_width=True)
            st.json(campaign_result.to_dict())

    with paper_tab:
        fixture_name = st.selectbox("Known EML tree", list(PAPER_FIXTURES))
        tree = PAPER_FIXTURES[fixture_name]
        st.code(rpn_string(tree))
        st.pyplot(tree_figure(tree))
        st.write(
            "These fixtures are for inspection. Final training proofs still run through "
            "the exact verifier."
        )


if __name__ == "__main__":
    main()
