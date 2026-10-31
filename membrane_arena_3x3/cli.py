"""Command-line interface for the synthetic 3x3 membrane arena."""

from __future__ import annotations

import argparse
import json

from .engine import EngineConfig, MembraneArena3x3
from .lattice import LatticeConfig
from .shields import GuardConfig

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="membrane-arena-3x3",
        description="SYNTHETIC 3x3 density-matrix membrane lattice simulation",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coupling", type=float, default=0.02)
    parser.add_argument("--dephasing", type=float, default=0.5)
    parser.add_argument("--mirror", type=float, default=0.10)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--schumann", action="store_true")
    parser.add_argument("--schumann-hz", type=float, default=7.83)
    parser.add_argument("--schumann-amplitude", type=float, default=0.02)
    parser.add_argument("--residual-target", type=float, default=0.20)
    parser.add_argument("--divergence-min", type=float, default=0.02)
    parser.add_argument("--divergence-max", type=float, default=0.60)
    parser.add_argument("--purity-floor", type=float, default=0.05)
    parser.add_argument("--entropy-target", type=float, default=0.5)
    parser.add_argument("--failure-limit", type=int, default=5)
    parser.add_argument("--operator-mode", default="legacy", choices=["legacy", "exact"])
    parser.add_argument("--no-adaptive-retry", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--json", default=None, help="Write full JSON results to this path"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Accepted for compatibility; the output is always the short summary",
    )
    return parser

def main() -> None:
    """Run the simulation from the command line."""
    args = build_parser().parse_args()

    parser = build_parser()
    try:
        config = EngineConfig(
            lattice=LatticeConfig(
                coupling=args.coupling,
                dephasing=args.dephasing,
                mirror_strength=args.mirror,
                dt=args.dt,
                operator_mode=args.operator_mode,
            ),
            guards=GuardConfig(
                residual_target=args.residual_target,
                divergence_min=args.divergence_min,
                divergence_max=args.divergence_max,
                purity_floor=args.purity_floor,
                entropy_target=args.entropy_target,
            ),
            steps=args.steps,
            seed=args.seed,
            schumann=args.schumann,
            schumann_hz=args.schumann_hz,
            schumann_amplitude=args.schumann_amplitude,
            failure_limit=args.failure_limit,
            adaptive_retry=not args.no_adaptive_retry,
            max_retries=args.max_retries,
        )
    except ValueError as error:
        parser.error(str(error))

    result = MembraneArena3x3(config).run()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    final = result.get("final", {})
    summary = {
        "status": result["status"],
        "steps_executed": result["steps_executed"],
        "accepted_steps": result["accepted_steps"],
        "rejected_steps": result["rejected_steps"],
        "final_mean_purity": final.get("mean_purity"),
        "final_max_divergence": final.get("max_divergence"),
        "final_max_residual": final.get("max_residual"),
        "final_min_purity": final.get("min_purity"),
        "final_stability_penalty": final.get("stability_penalty"),
        "final_mean_off_diagonal_coherence": final.get("mean_off_diagonal_coherence"),
        "guards": result.get("guards"),
        "operator_mode": result.get("operator_mode"),
        "repair_accepts": result.get("repair_accepts"),
    }

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
