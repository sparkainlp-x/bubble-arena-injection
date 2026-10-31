"""Independent Yang Observer and Yin Calibrator shield membranes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .core import (
    fidelity,
    max_off_diagonal_coherence,
    mean_off_diagonal_coherence,
    normalized_entropy,
    purity,
    residual_infinity,
)


def _read_only(array: np.ndarray) -> np.ndarray:
    """Return a view of `array` that raises if anything tries to write to it."""
    view = np.asarray(array).view()
    view.flags.writeable = False
    return view


@dataclass
class GuardConfig:
    """All guard thresholds, in one place, so a run can log them.

    Defaults reproduce the stock guards exactly: residual_target 0.20,
    divergence_max 0.60, purity_floor 0.05, entropy_target 0.5 (the value
    previously hard-coded in the stability-penalty formula). The
    failure_limit lives on EngineConfig and is logged alongside these
    in the run summary. ``divergence_min`` is now a real check: a lattice
    whose mean pairwise divergence falls below it is rejected as
    collapsed/synchronised (``divergence_below_minimum``). The default
    0.02 is far below any divergence a random 32-dimensional lattice
    shows, so it does not change stock behaviour.
    """

    residual_target: float = 0.20
    divergence_min: float = 0.02
    divergence_max: float = 0.60
    purity_floor: float = 0.05
    entropy_target: float = 0.5

    def __post_init__(self) -> None:
        if not self.residual_target > 0:
            raise ValueError("residual_target must be > 0")
        if not 0.0 <= self.divergence_min < self.divergence_max <= 1.0:
            raise ValueError("need 0 <= divergence_min < divergence_max <= 1")
        if not 0.0 <= self.purity_floor <= 1.0:
            raise ValueError("purity_floor must be in [0, 1]")
        if not 0.0 <= self.entropy_target <= 1.0:
            raise ValueError("entropy_target must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Observation:
    """Read-only stability telemetry produced by the Yang Observer Shield.

    Naming (renamed from the stock names, which were misleading):
    ``mean_purity`` is mean tr(rho^2) over the nine cells; the old name
    ``mean_coherence`` is kept as a read-only property for callers that
    still use it. ``stability_penalty`` is a weighted penalty score, not
    a Lyapunov function: nothing guarantees that it decreases; the old
    name ``lyapunov_proxy`` is likewise kept as a property.
    ``mean_off_diagonal_coherence`` / ``max_off_diagonal_coherence`` are
    true off-diagonal measures (mean / max |rho[i, j]|, i != j).
    """

    step: int
    mean_purity: float
    max_divergence: float
    max_residual: float
    min_purity: float
    mean_entropy: float
    stability_penalty: float
    mean_divergence: float = 0.0
    mean_off_diagonal_coherence: float = 0.0
    max_off_diagonal_coherence: float = 0.0
    offending_cells: list[str] = field(default_factory=list)
    offending_by_guard: dict[str, list[str]] = field(default_factory=dict)

    # Backwards-compatible read access to the old (misleading) names.
    @property
    def mean_coherence(self) -> float:  # noqa: D102
        return self.mean_purity

    @property
    def lyapunov_proxy(self) -> float:  # noqa: D102
        return self.stability_penalty

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibrationAction:
    """A bounded, fully logged calibrator proposal."""

    step: int
    feedback: float
    schumann: float
    total_signal: float
    coupling: float  # parameter values in effect (set by the engine), not proposals
    dephasing: float
    mirror_strength: float
    accepted: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class YangObserverShield:
    """Read-only observer. It never mutates lattice state."""

    def __init__(
        self,
        guard: GuardConfig | None = None,
        residual_target: float | None = None,
        divergence_min: float | None = None,
        divergence_max: float | None = None,
        purity_floor: float | None = None,
        entropy_target: float | None = None,
    ) -> None:
        if guard is not None and any(
            v is not None
            for v in (
                residual_target,
                divergence_min,
                divergence_max,
                purity_floor,
                entropy_target,
            )
        ):
            raise ValueError("pass a GuardConfig or individual thresholds, not both")
        self.guards = guard or GuardConfig(
            residual_target=residual_target if residual_target is not None else 0.20,
            divergence_min=divergence_min if divergence_min is not None else 0.02,
            divergence_max=divergence_max if divergence_max is not None else 0.60,
            purity_floor=purity_floor if purity_floor is not None else 0.05,
            entropy_target=entropy_target if entropy_target is not None else 0.5,
        )
        # Plain attributes for callers that read the stock names.
        self.residual_target = self.guards.residual_target
        self.divergence_min = self.guards.divergence_min
        self.divergence_max = self.guards.divergence_max
        self.purity_floor = self.guards.purity_floor
        self.entropy_target = self.guards.entropy_target

    def observe(
        self, state: np.ndarray, reference: np.ndarray, step: int
    ) -> Observation:
        """Compute stability metrics for all nine membrane cells.

        The inputs are wrapped in read-only views, so an accidental write
        inside the observer raises instead of changing the lattice.
        Offending cells are recorded per guard, so a rejection can say
        exactly which cells failed and on which guard.
        """
        state = _read_only(state)
        reference = _read_only(reference)
        cells = state.reshape(-1, state.shape[-2], state.shape[-1])
        refs = reference.reshape(-1, reference.shape[-2], reference.shape[-1])
        g = self.guards

        purities = []
        residuals = []
        entropies = []
        mean_off = []
        max_off = []
        for k, rho in enumerate(cells):
            purities.append(purity(rho))
            residuals.append(residual_infinity(rho, refs[k]))
            entropies.append(normalized_entropy(rho))
            mean_off.append(mean_off_diagonal_coherence(rho))
            max_off.append(max_off_diagonal_coherence(rho))

        divergences = []
        for a in range(len(cells)):
            for b in range(a + 1, len(cells)):
                divergences.append(1.0 - fidelity(cells[a], cells[b]))

        def cell_name(k: int) -> str:
            return f"{k // 3},{k % 3}"

        offending: dict[str, list[str]] = {
            "residual_above_target": [
                cell_name(k) for k, r in enumerate(residuals) if r > g.residual_target
            ],
            "divergence_above_maximum": [
                f"{cell_name(a)}-{cell_name(b)}"
                for idx, (a, b) in enumerate(
                    [(a, b) for a in range(len(cells)) for b in range(a + 1, len(cells))]
                )
                if divergences[idx] > g.divergence_max
            ],
            "purity_too_low": [
                cell_name(k) for k, p in enumerate(purities) if p < g.purity_floor
            ],
            "divergence_below_minimum": (
                ["lattice:mean_divergence"] if divergences and float(np.mean(divergences)) < g.divergence_min else []
            ),
        }
        union: list[str] = []
        for items in offending.values():
            for item in items:
                if item not in union:
                    union.append(item)

        mean_purity = float(np.mean(purities))
        max_divergence = float(np.max(divergences)) if divergences else 0.0
        mean_divergence = float(np.mean(divergences)) if divergences else 0.0
        max_residual = float(np.max(residuals))
        min_purity = float(np.min(purities))
        mean_entropy = float(np.mean(entropies))

        stability_penalty = (
            1.0 * max_residual
            + 0.5 * max(0.0, max_divergence - g.divergence_max)
            + 0.5 * abs(mean_entropy - g.entropy_target)
            + 0.5 * (1.0 - min_purity)
        )

        return Observation(
            step=step,
            mean_purity=mean_purity,
            max_divergence=max_divergence,
            max_residual=max_residual,
            min_purity=min_purity,
            mean_entropy=mean_entropy,
            stability_penalty=float(stability_penalty),
            mean_divergence=mean_divergence,
            mean_off_diagonal_coherence=float(np.mean(mean_off)),
            max_off_diagonal_coherence=float(np.max(max_off)) if max_off else 0.0,
            offending_cells=union,
            offending_by_guard={k: v for k, v in offending.items() if v},
        )

    def is_healthy(self, observation: Observation) -> tuple[bool, str]:
        """Return whether the observation passes stability guardrails."""
        g = self.guards
        if observation.max_residual > g.residual_target:
            return False, "residual_above_target"
        if observation.max_divergence > g.divergence_max:
            return False, "divergence_above_maximum"
        if observation.min_purity < g.purity_floor:
            return False, "purity_too_low"
        if observation.mean_divergence < g.divergence_min:
            return False, "divergence_below_minimum"
        return True, "ok"

    def guard_report(self, observation: Observation) -> dict:
        """Return rejection telemetry: failed guard, cells, metrics, thresholds."""
        healthy, reason = self.is_healthy(observation)
        return {
            "failed_guard": None if healthy else reason,
            "offending_cells": observation.offending_cells,
            "offending_by_guard": observation.offending_by_guard,
            "metrics": {
                "mean_purity": observation.mean_purity,
                "max_divergence": observation.max_divergence,
                "mean_divergence": observation.mean_divergence,
                "max_residual": observation.max_residual,
                "min_purity": observation.min_purity,
                "mean_entropy": observation.mean_entropy,
                "stability_penalty": observation.stability_penalty,
                "mean_off_diagonal_coherence": observation.mean_off_diagonal_coherence,
                "max_off_diagonal_coherence": observation.max_off_diagonal_coherence,
            },
            "thresholds": self.guards.to_dict(),
        }


class YinCalibratorShield:
    """Bounded calibrator. It proposes control but cannot bypass the checks.

    Its only control output is one clipped scalar, `total_signal`, which pulls
    every cell toward its initial reference state. It never proposes new
    coupling, dephasing or mirror values.
    """

    def __init__(
        self,
        coherence_target: float = 0.85,
        divergence_target: float = 0.15,
        residual_target: float = 0.20,
        max_signal: float = 0.05,
        gain_coherence: float = 0.8,
        gain_divergence: float = 0.4,
        gain_residual: float = 0.6,
    ) -> None:
        self.coherence_target = coherence_target
        self.divergence_target = divergence_target
        self.residual_target = residual_target
        self.max_signal = max_signal
        self.gain_coherence = gain_coherence
        self.gain_divergence = gain_divergence
        self.gain_residual = gain_residual
        self.log: list[CalibrationAction] = []

    def propose(
        self,
        observation: Observation,
        step: int,
        schumann_enabled: bool,
        schumann_hz: float,
        schumann_amplitude: float,
        dt: float,
    ) -> CalibrationAction:
        """Return a clipped calibration proposal based on observer telemetry."""
        feedback = (
            self.gain_coherence
            * (self.coherence_target - observation.mean_purity)
            + self.gain_divergence
            * (observation.max_divergence - self.divergence_target)
            + self.gain_residual
            * (observation.max_residual - self.residual_target)
        )
        feedback = float(np.clip(feedback, -self.max_signal, self.max_signal))

        schumann = 0.0
        if schumann_enabled:
            schumann = schumann_amplitude * np.sin(
                2 * np.pi * schumann_hz * step * dt
            )
            schumann = float(
                np.clip(schumann, -self.max_signal, self.max_signal)
            )

        total = float(
            np.clip(feedback + schumann, -self.max_signal, self.max_signal)
        )

        action = CalibrationAction(
            step=step,
            feedback=feedback,
            schumann=schumann,
            total_signal=total,
            coupling=0.0,
            dephasing=0.0,
            mirror_strength=0.0,
            accepted=False,
            reason="proposed",
        )
        self.log.append(action)
        return action

    def mark(self, action: CalibrationAction, accepted: bool, reason: str) -> None:
        """Record the engine's accept/reject decision for auditability."""
        action.accepted = accepted
        action.reason = reason
