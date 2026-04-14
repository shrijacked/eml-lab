"""Command line interface for EML Lab."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from eml_lab.agentic import OrchestratorConfig, run_orchestrator
from eml_lab.benchmarks import run_benchmark_suite
from eml_lab.campaigns import list_campaigns, run_campaign
from eml_lab.comparison import run_pysr_compare_suite, run_pysr_comparison
from eml_lab.targets import get_target, list_targets
from eml_lab.training import TrainConfig, train_target, write_train_artifacts


def _orchestratable_targets() -> list[str]:
    return [name for name in list_targets() if get_target(name).known_route is not None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eml-lab", description="EML Lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train and snap one target")
    train.add_argument("--target", choices=list_targets(), default="ln")
    train.add_argument("--depth", type=int, default=None)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--learning-rate", type=float, default=0.03)
    train.add_argument(
        "--snap-strategy", choices=["logits", "best_discrete"], default="best_discrete"
    )
    train.add_argument("--init-strategy", choices=["random", "known_route"], default="random")
    train.add_argument("--output-dir", type=Path, default=None)

    bench = subparsers.add_parser("bench", help="run an internal benchmark suite")
    bench.add_argument("--suite", choices=["shallow"], default="shallow")
    bench.add_argument("--output-dir", type=Path, default=Path("runs"))

    campaign = subparsers.add_parser("campaign", help="run a Phase 2 campaign suite")
    campaign.add_argument("--suite", choices=list_campaigns(), default="phase2-foundation")
    campaign.add_argument("--output-dir", type=Path, default=Path("runs"))

    compare = subparsers.add_parser("compare", help="run an optional PySR baseline comparison")
    compare.add_argument("--target", choices=list_targets(comparison_eligible=True), default="ln")
    compare.add_argument("--output-dir", type=Path, default=Path("runs"))
    compare.add_argument("--points", type=int, default=128)
    compare.add_argument("--niterations", type=int, default=40)
    compare.add_argument("--maxsize", type=int, default=20)
    compare.add_argument("--seed", type=int, default=0)

    compare_suite = subparsers.add_parser(
        "compare-suite", help="run the aggregated optional PySR compare suite"
    )
    compare_suite.add_argument("--suite", choices=["shallow"], default="shallow")
    compare_suite.add_argument("--output-dir", type=Path, default=Path("runs"))
    compare_suite.add_argument("--points", type=int, default=128)
    compare_suite.add_argument("--niterations", type=int, default=40)
    compare_suite.add_argument("--maxsize", type=int, default=20)
    compare_suite.add_argument("--seed", type=int, default=0)

    orchestrate = subparsers.add_parser(
        "orchestrate", help="run the local proposer/evaluator/pruner loop"
    )
    orchestrate.add_argument("--target", choices=_orchestratable_targets(), default="ln")
    orchestrate.add_argument("--output-dir", type=Path, default=Path("runs"))
    orchestrate.add_argument("--budget", type=int, default=64)
    orchestrate.add_argument("--beam-width", type=int, default=8)
    orchestrate.add_argument("--seed-count", type=int, default=4)
    orchestrate.add_argument("--seed", type=int, default=0)
    orchestrate.add_argument("--max-depth", type=int, default=None)

    app = subparsers.add_parser("app", help="launch the local Streamlit app")
    app.add_argument(
        "--dry-run", action="store_true", help="print the command without starting Streamlit"
    )
    app.add_argument("--server-port", type=int, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        config = TrainConfig(
            target=args.target,
            depth=args.depth,
            seed=args.seed,
            steps=args.steps,
            learning_rate=args.learning_rate,
            snap_strategy=args.snap_strategy,
            init_strategy=args.init_strategy,
        )
        result = train_target(config)
        if args.output_dir is not None:
            write_train_artifacts(result, args.output_dir)
        print(json.dumps(result.to_metrics_dict(), indent=2, default=str))
        return 0 if result.success else 2
    if args.command == "bench":
        result = run_benchmark_suite(args.suite, args.output_dir)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.recovery_rate == 1.0 else 2
    if args.command == "campaign":
        result = run_campaign(args.suite, args.output_dir)
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.success else 2
    if args.command == "compare":
        result = run_pysr_comparison(
            target=args.target,
            output_dir=args.output_dir,
            points=args.points,
            niterations=args.niterations,
            maxsize=args.maxsize,
            seed=args.seed,
        )
        print(json.dumps(result.to_dict(), indent=2, default=str))
        if result.success:
            return 0
        if not result.available:
            return 3
        return 2
    if args.command == "compare-suite":
        result = run_pysr_compare_suite(
            suite=args.suite,
            output_dir=args.output_dir,
            points=args.points,
            niterations=args.niterations,
            maxsize=args.maxsize,
            seed=args.seed,
        )
        print(json.dumps(result.to_dict(), indent=2, default=str))
        if result.available:
            return 0 if result.pysr_success_rate == 1.0 else 2
        return 3
    if args.command == "orchestrate":
        result = run_orchestrator(
            OrchestratorConfig(
                target=args.target,
                budget=args.budget,
                beam_width=args.beam_width,
                seed_count=args.seed_count,
                seed=args.seed,
                max_depth=args.max_depth,
            ),
            args.output_dir,
        )
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.success else 2
    if args.command == "app":
        app_path = Path(__file__).with_name("app.py")
        command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
        if args.server_port is not None:
            command.extend(["--server.port", str(args.server_port)])
        if args.dry_run:
            print(" ".join(command))
            return 0
        return subprocess.call(command)
    parser.error(f"Unknown command {args.command!r}")
    return 2
