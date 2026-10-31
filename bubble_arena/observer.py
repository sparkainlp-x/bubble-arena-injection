"""External observer for the arena bubble: measures, never mutates.

Every measurement is tagged REPORTED, never MEASURED: the observer sees
probe outcomes and run summaries, never the engine's internals directly.
Observation is pure reads only (state hash, counters, step outcomes the
run produces anyway). It never calls propose_step, never commits, never
changes a guard, a threshold, or the RNG stream.

Two documented differences from bubble-nervous-system's observer:
- Binary guard decisions are recorded VERBATIM. No noise is added to an
  accept/reject decision: noise on a binary outcome would corrupt the
  measurement rather than model it. The "noise" in a probe is the probe
  itself (a junk probe is a randomized candidate).
- Half-widths are computed from repeated measurements (across seeds),
  not from a declared sigma.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from membrane_arena_3x3.crystallize_injection import (
    InjectionConfig,
    MembraneArena3x3WithInjection,
)
from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3

JUNK_SALT = 0x6A11


def half_width(values, z=1.96):
    if len(values) < 2:
        return math.inf
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return z * math.sqrt(var / len(values))


class Report:
    def __init__(self, kind, value, half_width=math.inf, n=0, tag="REPORTED"):
        self.kind = kind
        self.value = value
        self.half_width = half_width
        self.n = n
        self.tag = tag

    def to_dict(self):
        return {"kind": self.kind, "value": self.value,
                "half_width": self.half_width, "n": self.n, "tag": self.tag}


class BubbleObserver:
    def __init__(self, bubble):
        self.bubble = bubble
        self.reports = []

    def snapshot(self, engine):
        """Pure read of an engine's observable state. Mutates nothing."""
        return {
            "state_sha256": hashlib.sha256(engine.lattice.state.tobytes()).hexdigest(),
            "step": engine.current_step,
            "failures": engine.failures,
            "breaker": engine.circuit_breaker,
            "n_calibrations": len(engine.calibrations),
            "n_observations": len(engine.observations),
        }

    def observed_run(self, seed, retry, injected):
        """Drive one run step by step, snapshotting (pure reads) between steps.

        The snapshot calls cannot influence the trajectory: they read
        state and counters only. The returned summary must equal a bare
        .run() of the same seed/config; any difference is a finding.
        """
        engine = self.bubble.new_engine(seed, retry, injected)
        snapshots = [self.snapshot(engine)]
        for _ in range(engine.config.steps):
            result = engine.step()
            snapshots.append(self.snapshot(engine))
            if result["status"] == "circuit_breaker_open":
                break
        summary = engine.summary()
        rep = Report("run", {
            "seed": seed, "retry": retry, "injected": injected,
            "steps": summary["steps_executed"],
            "accepted": summary["accepted_steps"],
            "status": summary["status"],
        }, half_width=math.inf, n=1)
        self.reports.append(rep)
        return summary, snapshots, engine

    def bare_run(self, seed, retry, injected):
        """The same run with no observer attached. For the non-invasion check."""
        engine = self.bubble.new_engine(seed, retry, injected)
        return engine.run(), engine

    def measure_gate(self, seed):
        """One clean probe and one junk probe on a fresh, held-out-seed engine.

        Clean probe: the first step of a fresh stock engine with adaptive
        retry OFF, so the outcome is the frozen guard's decision alone
        (no repair). Junk probe: a uniform-random complex array of the
        candidate shape, validated by the same frozen guard, on the same
        fresh engine BEFORE the clean step (validation is pure).
        Returns (clean_refused, junk_refused, clean_reason, junk_reason).
        """
        engine = self.bubble.new_engine(seed, retry=False, injected=False)
        rng = np.random.default_rng([int(seed), JUNK_SALT])
        shape = engine.lattice.state.shape
        junk = rng.uniform(size=shape) + 1j * rng.uniform(size=shape)
        junk_ok, junk_reason = engine._validate(junk)
        result = engine.step()
        clean_accepted = result["status"] == "accepted"
        rep = Report("gate", {
            "seed": seed,
            "clean_refused": not clean_accepted,
            "junk_refused": not junk_ok,
            "clean_reason": result["reason"],
            "junk_reason": junk_reason,
        }, half_width=math.inf, n=2)
        self.reports.append(rep)
        return (not clean_accepted), (not junk_ok), result["reason"], junk_reason
