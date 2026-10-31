"""Bubble holding the arena engine, unchanged inside.

Inside the bubble lives the vendored membrane_arena_3x3 engine (stock
and Crystallize-injection subclass), byte-identical to the upgraded
tree (hashes in VENDORED_SHA256.txt). Outside it live only a
BubbleObserver and a Calibrator, which see REPORTED measurements,
never the engine's internals directly.

Do not confuse the two observer/calibrator pairs: the ENGINE has its
own internal YangObserverShield / YinCalibratorShield (part of the
vendored code, driving each step); the BUBBLE has an external
BubbleObserver / Calibrator (this package), which only observe runs
and update an assumed model. The bubble calibrator never proposes a
step and never touches a guard.

Frozen gate: the stock GuardConfig thresholds (residual 0.20,
divergence 0.02-0.60, purity 0.05, entropy 0.50, failure_limit 5) plus
the default InjectionConfig (max_attempts 3, first_commit 0.35,
adaptive off). They are never re-tuned on held-out probes.
"""

from __future__ import annotations

from membrane_arena_3x3.crystallize_injection import (
    InjectionConfig,
    MembraneArena3x3WithInjection,
)
from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3


class BubbleArena:
    """Arena engines in a bubble, with assumed vs true separation."""

    def __init__(self):
        self.gates = {
            "gate_calibrated": True,
            "review_signoff": False,
            "money_gates_pass": True,
        }
        # Assumed model: what the outside world believes, before any
        # calibration. All None until a calibration is applied.
        self.assumed_steps_mean = None
        self.assumed_accept_rate = None
        self.assumed_clean_refusal = None
        self.assumed_junk_refusal = None
        self.assumed_injection_extends = None
        self.assumed_stock_steps = None
        self.assumed_injected_steps = None

    def set_review_signoff(self, value=True):
        self.gates["review_signoff"] = bool(value)

    def new_engine(self, seed, retry, injected):
        """Fresh engine: stock or injected, default injection config."""
        config = EngineConfig(steps=200, seed=int(seed), adaptive_retry=bool(retry))
        if injected:
            return MembraneArena3x3WithInjection(config, InjectionConfig())
        return MembraneArena3x3(config)

    def frozen_gate(self):
        """The frozen guard thresholds + injection defaults (read-only view)."""
        probe = self.new_engine(42, retry=True, injected=False)
        guards = dict(probe.config.guards_dict())
        inj = InjectionConfig()
        return {
            "guards": guards,
            "injection": {
                "max_attempts": inj.max_attempts,
                "first_commit": inj.first_commit,
                "adaptive": inj.adaptive,
            },
        }

    def public_state(self):
        """What the outside world may see: assumed model only."""
        return {
            "assumed_steps_mean": self.assumed_steps_mean,
            "assumed_accept_rate": self.assumed_accept_rate,
            "assumed_clean_refusal": self.assumed_clean_refusal,
            "assumed_junk_refusal": self.assumed_junk_refusal,
            "assumed_injection_extends": self.assumed_injection_extends,
            "gate_frozen": True,
        }
