"""Isolated PySR worker.

This module intentionally avoids importing torch. PySR loads Julia through
``juliacall``, which warns and can become unstable if torch is already imported
in the same Python process.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _target_values(target: str, x: np.ndarray) -> np.ndarray:
    if target == "exp":
        return np.exp(x)
    if target == "ln":
        return np.log(x)
    if target == "identity":
        return x
    if target == "square":
        return x * x
    if target == "sin":
        return np.sin(x)
    raise ValueError(f"Unsupported PySR worker target: {target}")


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eml-lab-pysr-worker")
    parser.add_argument("--target", required=True)
    parser.add_argument("--low", type=float, required=True)
    parser.add_argument("--high", type=float, required=True)
    parser.add_argument("--points", type=int, required=True)
    parser.add_argument("--niterations", type=int, required=True)
    parser.add_argument("--maxsize", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    try:
        from pysr import PySRRegressor
    except Exception as exc:
        _write_summary(
            args.summary_path,
            {
                "status": "unavailable",
                "reason": str(exc),
                "install_hint": (
                    "PySR needs a reachable Julia download source the first time it boots a "
                    "managed runtime. If you already have Julia installed, add it to PATH."
                ),
            },
        )
        return 3

    x = np.linspace(args.low, args.high, args.points, dtype=np.float64)
    try:
        y = _target_values(args.target, x).astype(np.float64)
    except ValueError as exc:
        _write_summary(args.summary_path, {"status": "unsupported", "reason": str(exc)})
        return 2

    model = PySRRegressor(
        niterations=args.niterations,
        maxsize=args.maxsize,
        model_selection="best",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log"],
        progress=False,
        precision=64,
        output_directory=str(args.output_directory),
        run_id=f"eml_lab_{args.target}_{args.seed}",
    )

    start = time.perf_counter()
    try:
        model.fit(x.reshape(-1, 1), y)
    except Exception as exc:
        _write_summary(
            args.summary_path,
            {
                "status": "error",
                "reason": str(exc),
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
        return 2
    elapsed = time.perf_counter() - start

    equations = getattr(model, "equations_", None)
    equations_records: list[dict[str, object]] | None = None
    if equations is not None:
        if hasattr(equations, "to_dict"):
            equations_records = equations.to_dict(orient="records")
        if hasattr(equations, "to_csv"):
            equations.to_csv(args.output_directory / "equations.csv", index=False)

    best_equation = None
    if hasattr(model, "sympy"):
        try:
            best_equation = str(model.sympy())
        except Exception as exc:
            best_equation = f"<sympy export failed: {exc}>"

    _write_summary(
        args.summary_path,
        {
            "status": "ok",
            "elapsed_seconds": elapsed,
            "best_equation": best_equation,
            "operators": {
                "binary": ["+", "-", "*", "/"],
                "unary": ["exp", "log"],
            },
            "niterations": args.niterations,
            "maxsize": args.maxsize,
            "equations": equations_records,
            "output_directory": str(args.output_directory),
        },
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
