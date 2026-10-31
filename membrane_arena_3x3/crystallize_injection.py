"""Opt-in Crystallize injection engine for the SYNTHETIC 3x3 membrane arena.

The stock engine retries an identical proposal after a rejection (README,
"Known limits", item 1). This module adds a subclass that, on a rejected
step only, tries up to ``max_attempts`` *variant* proposals using a small
commit -> sample -> revise -> score loop (the Crystallize calibration loop):

1. Commit: state a probability ``p`` (``first_commit``) that a shrunk variant will pass the guards.
2. Sample: build the variant (smaller ``dt``, softened mirror) and validate it
   with the engine's own, unmodified ``_validate``.
3. Revise: after a miss, move ``p`` halfway toward 0.5 and shrink the step again.
4. Score: Brier = (p - truth)^2, points = 100 * (1 - 4 * Brier); every variant
   tried costs ``cost_per_variant`` (0.8) in the ledger.

What this does NOT do: it never changes a guard, a threshold, a parameter
bound or the circuit breaker, and it does not cure the arena. This is
SYNTHETIC classical software; nothing here is a physical claim.

Design notes
------------
* The injection uses its own random stream (``numpy.random.default_rng`` seeded
  from ``[EngineConfig.seed, InjectionConfig.rng_salt]``). The engine's main
  generator is never drawn from, so stock behaviour is unchanged.
* ``adaptive=True`` (default off) commits the engine's own smoothed pass rate
  instead of a fixed number. It moves only the ledger; acceptance, guards,
  thresholds and the breaker are untouched. 0.35 is the stock-regime default
  only; a fixed value cannot be right in every regime.
* With ``max_attempts=0`` ``step()`` delegates to the stock implementation.
* A variant is only committed if ``_validate`` accepts it. Failed variants
  never touch ``lattice.state``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import EngineConfig, MembraneArena3x3

P_MIN = 0.01
P_MAX = 0.99
ADAPTIVE_PRIOR_WEIGHT = 4  # pseudo-events of weight on first_commit
FACTOR_MAX = 0.99  # jitter can never push a variant step above the failed step


@dataclass
class InjectionConfig:
    """Settings for the injection loop. Defaults follow the design notes."""

    max_attempts: int = 3  # variants tried per rejected step; 0 = stock behaviour
    first_commit: float = 0.35  # committed probability that a variant passes (ledger only)
    first_factor: float = 0.5  # first variant uses dt * first_factor * jitter
    dt_shrink: float = 0.5  # each further attempt shrinks the step by this factor
    jitter_low: float = 0.8
    jitter_high: float = 1.2
    mirror_variant: float = 0.0  # softened mirror strength (inside [0, 0.50])
    cost_per_variant: float = 0.8
    # adaptive=True replaces the fixed first commit with the smoothed pass rate
    # (passes + 4 * first_commit) / (events + 4), starting at first_commit.
    # It changes only the committed p, hence only the ledger, never acceptance.
    adaptive: bool = False
    rng_salt: int = 0xC4157A11  # separates the injection stream from the main seed

    def __post_init__(self) -> None:
        if self.max_attempts < 0:
            raise ValueError("max_attempts must be >= 0")
        if not 0.0 < self.first_commit < 1.0:
            raise ValueError("first_commit must be in (0, 1)")
        if not 0.0 < self.first_factor < 1.0:
            raise ValueError("first_factor must be in (0, 1)")
        if not self.dt_shrink > 0:
            raise ValueError("dt_shrink must be > 0")
        if not 0.0 <= self.mirror_variant <= 0.50:
            raise ValueError("mirror_variant must be in [0, 0.50]")
        if not 0 < self.jitter_low <= self.jitter_high:
            raise ValueError("need 0 < jitter_low <= jitter_high")
        if self.cost_per_variant < 0:
            raise ValueError("cost_per_variant must be >= 0")
        if not isinstance(self.adaptive, bool):
            raise ValueError("adaptive must be a bool")


def clamp_p(p: float) -> float:
    """Clamp a probability into [0.01, 0.99]."""
    return float(min(P_MAX, max(P_MIN, p)))


def brier(p: float, truth: bool) -> float:
    """Brier score of probability ``p`` against a boolean outcome."""
    return float((p - (1.0 if truth else 0.0)) ** 2)


def calibration_points(p: float, truth: bool) -> float:
    """Crystallize points: 100 * (1 - 4 * Brier). 0 at p = 0.5, 100 at a sure hit."""
    return float(100.0 * (1.0 - 4.0 * brier(p, truth)))


def revise_toward_humility(p: float) -> float:
    """Move ``p`` halfway toward 0.5 (the maximally humble forecast)."""
    return float(p + 0.5 * (0.5 - p))


class MembraneArena3x3WithInjection(MembraneArena3x3):
    """Stock arena plus a Crystallize injection on the rejection path."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        injection: InjectionConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.injection = injection or InjectionConfig()
        self.injection_rng = np.random.default_rng(
            [int(self.config.seed), int(self.injection.rng_salt)]
        )
        self.injections: list[dict] = []
        self.injection_score_total = 0.0
        self.injection_cost_total = 0.0
        self._adaptive_events = 0
        self._adaptive_passes = 0

    def _first_commit(self) -> float:
        """Committed p for the first variant of an event.

        Fixed ``first_commit``, or with ``adaptive`` the smoothed pass rate
        (passes + w * first_commit) / (events + w) over earlier events.
        """
        inj = self.injection
        if not inj.adaptive:
            return clamp_p(inj.first_commit)
        return clamp_p(
            (self._adaptive_passes + ADAPTIVE_PRIOR_WEIGHT * inj.first_commit)
            / (self._adaptive_events + ADAPTIVE_PRIOR_WEIGHT)
        )

    def _propose_with_dt_factor(
        self, dt_factor: float, mirror_value: float, signal: float = 0.0
    ) -> np.ndarray:
        """Return a candidate built with ``dt * dt_factor`` and a given mirror strength.

        ``lattice.config.dt`` and ``lattice.config.mirror_strength`` are
        restored before returning, even on error. The lattice state is not
        modified. ``signal`` is the calibrator's total signal (default 0.0).
        """
        cfg = self.lattice.config
        dt0, mirror0 = cfg.dt, cfg.mirror_strength
        try:
            cfg.dt = dt0 * dt_factor
            cfg.mirror_strength = mirror_value
            return self.lattice.propose_step(signal)
        finally:
            cfg.dt, cfg.mirror_strength = dt0, mirror0

    def _inject(self, step_index: int, signal: float, base_reason: str):
        """Try variants after a rejection. Returns (event, candidate or None, mirror)."""
        inj = self.injection
        p = self._first_commit()
        attempts: list[dict] = []
        passed_candidate = None
        passed_mirror = None

        for k in range(inj.max_attempts):
            jitter = float(
                self.injection_rng.uniform(inj.jitter_low, inj.jitter_high)
            )
            factor = min(
                inj.first_factor * (inj.dt_shrink**k) * jitter, FACTOR_MAX
            )
            candidate = self._propose_with_dt_factor(
                factor, inj.mirror_variant, signal
            )
            ok, reason = self._validate(candidate)
            attempts.append(
                {
                    "attempt": k + 1,
                    "commit_p": p,
                    "factor": factor,
                    "mirror_strength": inj.mirror_variant,
                    "passed": bool(ok),
                    "reason": reason,
                    "brier": brier(p, ok),
                    "points": calibration_points(p, ok),
                }
            )
            if ok:
                passed_candidate = candidate
                passed_mirror = inj.mirror_variant
                break
            p = clamp_p(revise_toward_humility(p))

        event = {
            "step": step_index,
            "base_reason": base_reason,
            "attempts": attempts,
            "passed": passed_candidate is not None,
            "sampling_cost": inj.cost_per_variant * len(attempts),
            "score": float(sum(a["points"] for a in attempts)),
        }
        return event, passed_candidate, passed_mirror

    def step(self) -> dict:
        """One transactional update; on rejection, try injected variants first."""
        if self.circuit_breaker or self.injection.max_attempts == 0:
            return super().step()

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
        lattice_cfg = self.config.lattice
        action.coupling = lattice_cfg.coupling
        action.dephasing = lattice_cfg.dephasing
        action.mirror_strength = lattice_cfg.mirror_strength

        candidate = self.lattice.propose_step(action.total_signal)
        accepted, reason = self._validate(candidate)
        injected = False

        if not accepted:
            event, variant, mirror = self._inject(
                self.current_step + 1, action.total_signal, reason
            )
            self.injections.append(event)
            self.injection_score_total += event["score"]
            self.injection_cost_total += event["sampling_cost"]
            self._adaptive_events += 1
            self._adaptive_passes += int(event["passed"])
            if variant is not None:
                candidate, accepted, reason, injected = variant, True, "ok", True
                action.mirror_strength = mirror  # log the value actually used

        if accepted:
            self.lattice.commit(candidate)
            self.failures = 0
        else:
            self.failures += 1
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
            "injected": injected,
        }

    def summary(self) -> dict:
        out = super().summary()
        out["injection_events"] = len(self.injections)
        out["injection_passes"] = sum(1 for e in self.injections if e["passed"])
        out["injection_score_total"] = self.injection_score_total
        out["injection_cost_total"] = self.injection_cost_total
        out["adaptive_pass_rate"] = (
            self._first_commit() if self.injection.adaptive else None
        )
        out["injections"] = self.injections
        return out


def run_simulation_with_injection(
    config: EngineConfig | None = None,
    injection: InjectionConfig | None = None,
) -> dict:
    """Run one complete simulation with the injection enabled."""
    return MembraneArena3x3WithInjection(config, injection).run()
