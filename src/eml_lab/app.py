"""Streamlit dashboard for EML Lab."""

from __future__ import annotations

import streamlit as st

from eml_lab.benchmarks import benchmark_table, run_benchmark_suite
from eml_lab.targets import PAPER_FIXTURES, get_target, list_targets
from eml_lab.training import TrainConfig, train_target
from eml_lab.trees import rpn_string
from eml_lab.visualize import tree_figure


def main() -> None:
    st.set_page_config(page_title="EML Lab", layout="wide")
    st.title("EML Lab")
    st.caption("Differentiable formula discovery with one complex-valued binary operator.")

    train_tab, snap_tab, bench_tab, paper_tab = st.tabs(
        ["Train", "Snap", "Bench", "Paper Explorer"]
    )

    with train_tab:
        target_name = st.selectbox("Target", list_targets(), index=list_targets().index("ln"))
        spec = get_target(target_name)
        depth = st.number_input("Depth", min_value=1, max_value=5, value=spec.default_depth)
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=0)
        steps = st.slider("Steps", min_value=20, max_value=600, value=180, step=20)
        learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=0.2, value=0.03)
        snap_strategy = st.selectbox("Snap strategy", ["best_discrete", "logits"])
        init_strategy = st.selectbox("Init strategy", ["random", "known_route"])
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
