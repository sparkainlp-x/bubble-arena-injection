"""Transactional simulation engine for the 3x3 membrane lattice."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .core import is_valid_density_matrix
from .lattice import LatticeConfig, MembraneLattice
from .shields import GuardConfig, YangObserverShield, YinCalibratorShield


@dataclass
class EngineConfig:
    """Top-level simulation configuration."""

    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    guards: GuardConfig = field(default_factory=GuardConfig)
    steps: int = 200
    seed: int = 42
    schumann: bool = False
    schumann_hz: float = 7.83
    schumann_amplitude: float = 0.02
    failure_limit: int = 5  # consecutive rejected steps before the breaker latches
    # Bounded adaptive retry (default on): on a rejection, try variants
    # with a smaller dt, a softened mirror and a reduced coupling. One
    # failure is counted only after all repair attempts fail, so a run
    # can no longer stop from repeated identical rejections.
    adaptive_retry: bool = True
    max_retries: int = 3
    retry_dt_factors: tuple[float, ...] = (0.5, 0.25, 0.125)
    retry_mirror_factors: tuple[float, ...] = (0.5, 0.0, 0.0)
    retry_coupling_factors: tuple[float, ...] = (0.5, 0.25, 0.0)

    def __post_init__(self) -> None:
        """Reject settings that make a run meaningless or alias the signal."""
        if self.steps < 0:
            raise ValueError("steps must be >= 0")
        if self.failure_limit < 1:
            raise ValueError("failure_limit must be >= 1")
        if not self.lattice.dt > 0:
            raise ValueError("dt must be > 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if len(self.retry_dt_factors) < self.max_retries:
            raise ValueError("retry_dt_factors must cover max_retries")
        if len(self.retry_mirror_factors) < self.max_retries:
            raise ValueError("retry_mirror_factors must cover max_retries")
        if len(self.retry_coupling_factors) < self.max_retries:
            raise ValueError("retry_coupling_factors must cover max_retries")
        for factor in self.retry_dt_factors:
            if not 0.0 < factor < 1.0:
                raise ValueError("retry_dt_factors must be in (0, 1)")
        for factor in (*self.retry_mirror_factors, *self.retry_coupling_factors):
            if not 0.0 <= factor <= 1.0:
                raise ValueError("retry mirror/coupling factors must be in [0, 1]")
        if self.schumann:
            if not self.schumann_hz > 0:
                raise ValueError("schumann_hz must be > 0")
            nyquist = 0.5 / self.lattice.dt
            if self.schumann_hz >= nyquist:
                raise ValueError(
                    f"schumann_hz={self.schumann_hz} is at or above the sampling "
                    f"Nyquist limit {nyquist:g} Hz for dt={self.lattice.dt}; "
                    "the sampled sinusoid would alias"
                )

    def guards_dict(self) -> dict:
        """All guard thresholds, including failure_limit, for JSON output."""
        out = self.guards.to_dict()
        out["failure_limit"] = self.failure_limit
        return out


class MembraneArena3x3:
    """Run a fail-closed 3x3 membrane simulation with independent shields."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.lattice = MembraneLattice(self.config.lattice, self.rng)
        self.observer = YangObserverShield(guard=self.config.guards)
        self.calibrator = YinCalibratorShield()
        self.observations: list[dict] = []
        self.calibrations: list[dict] = []
        self.rejections: list[dict] = []
        self.repair_attempts_total = 0
        self.repair_accepts = 0
        self.failures = 0
        self.circuit_breaker = False
        self.current_step = 0

    def _validate(self, candidate: np.ndarray) -> tuple[bool, str]:
        """Validate an entire candidate lattice state before commit."""
        cfg = self.config.lattice

        for i in range(3):
            for j in range(3):
                rho = candidate[i, j]
                if not is_valid_density_matrix(rho):
                    return False, f"invalid_density_matrix[{i},{j}]"

        observation = self.observer.observe(
            candidate, self.lattice.reference, self.current_step + 1
        )
        healthy, reason = self.observer.is_healthy(observation)
        if not healthy:
            return False, reason

        if not (
            cfg.coupling_min <= cfg.coupling <= cfg.coupling_max
            and cfg.dephasing_min <= cfg.dephasing <= cfg.dephasing_max
            and cfg.mirror_min <= cfg.mirror_strength <= cfg.mirror_max
        ):
            return False, "calibrator_parameter_out_of_bounds"

        return True, "ok"

    def _validate_detailed(self, candidate: np.ndarray) -> tuple[bool, str, dict]:
        """Validate a candidate and return rejection telemetry with it."""
        ok, reason = self._validate(candidate)
        observation = self.observer.observe(
            candidate, self.lattice.reference, self.current_step + 1
        )
        report = self.observer.guard_report(observation)
        if ok:
            report["failed_guard"] = None
        elif reason.startswith("invalid_density_matrix"):
            report["failed_guard"] = reason
            report["offending_cells"] = [reason.split("[", 1)[1].rstrip("]")]
            report["offending_by_guard"] = {reason: report["offending_cells"]}
        elif reason == "calibrator_parameter_out_of_bounds":
            report["failed_guard"] = reason
            report["offending_cells"] = ["lattice:parameters"]
            report["offending_by_guard"] = {reason: ["lattice:parameters"]}
        return ok, reason, report

    def step(self) -> dict:
        """Execute one transactional global update, with adaptive repair."""
        if self.circuit_breaker:
            return {
                "status": "circuit_breaker_open",
                "step": self.current_step,
            }

        observation = self.observer.observe(
            self.lattice.state, self.lattice.reference, self.current_step
        )

        action = self.calibrator.propose(
            observation,
            self.current_step,
            self.config.schumann,
            self.config.schumann_hz,
            self.config.schumann_amplitude,
            self.config.lattice.dt,
        )

        # The calibrator never changes these; log the values in effect so the
        # audit trail does not show misleading zeros.
        lattice_cfg = self.config.lattice
        action.coupling = lattice_cfg.coupling
        action.dephasing = lattice_cfg.dephasing
        action.mirror_strength = lattice_cfg.mirror_strength

        candidate = self.lattice.propose_step(action.total_signal)
        accepted, reason, report = self._validate_detailed(candidate)
        repair_attempts: list[dict] = []
        accepted_via_repair = False

        if not accepted and self.config.adaptive_retry:
            for k in range(self.config.max_retries):
                dt_eff = lattice_cfg.dt * self.config.retry_dt_factors[k]
                mirror_eff = lattice_cfg.mirror_strength * self.config.retry_mirror_factors[k]
                coupling_eff = lattice_cfg.coupling * self.config.retry_coupling_factors[k]
                variant = self.lattice.propose_step(
                    action.total_signal,
                    dt=dt_eff,
                    mirror_strength=mirror_eff,
                    coupling=coupling_eff,
                )
                ok, variant_reason = self._validate(variant)
                self.repair_attempts_total += 1
                repair_attempts.append(
                    {
                        "attempt": k + 1,
                        "dt": dt_eff,
                        "mirror_strength": mirror_eff,
                        "coupling": coupling_eff,
                        "accepted": bool(ok),
                        "reason": variant_reason,
                    }
                )
                if ok:
                    candidate, accepted, reason = variant, True, "ok"
                    accepted_via_repair = True
                    self.repair_accepts += 1
                    # Calibration log keeps the stock values in effect;
                    # the variant's dt/mirror/coupling are in
                    # repair_attempts, not in the calibrator's proposal.
                    break

        if accepted:
            self.lattice.commit(candidate)
            self.failures = 0
        else:
            # One failure per step, counted only after all repair
            # attempts failed — never one per identical retry.
            self.failures += 1
            self.rejections.append(
                {
                    "step": self.current_step + 1,
                    "reason": reason,
                    "failed_guard": report["failed_guard"],
                    "offending_cells": report["offending_cells"],
                    "offending_by_guard": report["offending_by_guard"],
                    "metrics": report["metrics"],
                    "thresholds": report["thresholds"],
                    "repair_attempts": repair_attempts,
                }
            )
            if self.failures >= self.config.failure_limit:
                self.circuit_breaker = True

        self.calibrator.mark(action, accepted, reason)
        self.calibrations.append(action.to_dict())

        post = self.observer.observe(
            self.lattice.state, self.lattice.reference, self.current_step + 1
        )
        self.observations.append(post.to_dict())
        self.current_step += 1

        return {
            "status": "accepted" if accepted else "rejected",
            "reason": reason,
            "step": self.current_step,
            "observation": post.to_dict(),
            "calibration": action.to_dict(),
            "accepted_via_repair": accepted_via_repair,
            "repair_attempts": repair_attempts,
        }

    def run(self) -> dict:
        """Run the configured number of further steps, stopping if the breaker opens.

        The step counter and the logs continue from where the last call
        stopped; a second call does not restart them.
        """
        for _ in range(self.config.steps):
            result = self.step()
            if result["status"] == "circuit_breaker_open":
                break

        return self.summary()

    def summary(self) -> dict:
        """Return a machine-readable experiment summary."""
        final = self.observations[-1] if self.observations else {}
        accepted = sum(1 for item in self.calibrations if item["accepted"])

        return {
            "status": (
                "circuit_breaker_open"
                if self.circuit_breaker
                else "completed" if self.observations else "no_steps"
            ),
            "steps_requested": self.config.steps,
            "steps_executed": len(self.observations),
            "accepted_steps": accepted,
            "rejected_steps": len(self.calibrations) - accepted,
            "guards": self.config.guards_dict(),
            "operator_mode": self.config.lattice.operator_mode,
            "adaptive_retry": self.config.adaptive_retry,
            "rejections": self.rejections,
            "repair_attempts_total": self.repair_attempts_total,
            "repair_accepts": self.repair_accepts,
            "final": final,
            "observations": self.observations,
            "calibrations": self.calibrations,
        }


def run_simulation(config: EngineConfig | None = None) -> dict:
    """Convenience function for running one complete simulation."""
    return MembraneArena3x3(config).run()
