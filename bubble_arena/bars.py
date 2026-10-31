"""Preregistered bars for bubble-arena-injection. Measure both, tune neither.

Bar 1: steps executed / accepted for stock vs injected, with stock
adaptive retry ON and OFF, on seed 42 and on seeds 0-19, measured
THROUGH the split (observed runs) and BARE (direct .run()), side by
side. If the split changes any fact, the split_matches_bare flag says
so; a mismatch is a finding, never smoothed over.

Bar 2: (a) frozen-gate per-probe refusal rates, clean vs junk, on
held-out seeds 100-199 (disjoint from bar 1's seeds), measured through
the observer and directly — reported SEPARATELY, never merged.
(b) calibrator FailureCapsule refusals by reason, enumerated by
scenario — also reported separately from (a), never merged into one
rate. The gate is frozen: stock GuardConfig thresholds plus default
InjectionConfig, never re-tuned on held-out probes.
"""

from __future__ import annotations

import math

from .calibrator import Calibrator, FailureCapsule
from .observer import BubbleObserver, Report, half_width

BAR1_SEEDS = [42] + list(range(20))
BAR2_SEEDS = list(range(100, 200))


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def bar1(observer):
    """Return the bar-1 table: per (retry, injected, seed) observed vs bare."""
    table = {}
    for retry in (True, False):
        for injected in (False, True):
            for seed in BAR1_SEEDS:
                summary_obs, _snaps, _eng = observer.observed_run(seed, retry, injected)
                summary_bare, _eng2 = observer.bare_run(seed, retry, injected)
                match = (
                    summary_obs["steps_executed"] == summary_bare["steps_executed"]
                    and summary_obs["accepted_steps"] == summary_bare["accepted_steps"]
                    and summary_obs["status"] == summary_bare["status"]
                )
                if injected:
                    inj_obs = _eng.injections
                    inj_bare = _eng2.injections
                    match = match and inj_obs == inj_bare
                table[(retry, injected, seed)] = {
                    "steps": summary_obs["steps_executed"],
                    "accepted": summary_obs["accepted_steps"],
                    "status": summary_obs["status"],
                    "split_matches_bare": match,
                }
    summary = {}
    for retry in (True, False):
        for injected in (False, True):
            cells = [table[(retry, injected, s)] for s in range(20)]
            summary[(retry, injected)] = {
                "seed42_steps": table[(retry, injected, 42)]["steps"],
                "seed42_accepted": table[(retry, injected, 42)]["accepted"],
                "mean_steps_0_19": _mean([c["steps"] for c in cells]),
                "mean_accepted_0_19": _mean([c["accepted"] for c in cells]),
                "all_split_match_bare": all(
                    table[(retry, injected, s)]["split_matches_bare"]
                    for s in BAR1_SEEDS
                ),
            }
    return table, summary


def _binom_half_width(p, n, z=1.96):
    if n <= 0:
        return math.inf
    return z * math.sqrt(p * (1.0 - p) / n)


def bar2_rates(observer):
    """Frozen-gate refusal rates, observer path and direct path, held-out seeds."""
    clean_obs, junk_obs, clean_dir, junk_dir = [], [], [], []
    for seed in BAR2_SEEDS:
        clean_ref, junk_ref, _cr, _jr = observer.measure_gate(seed)
        clean_obs.append(clean_ref)
        junk_obs.append(junk_ref)
        # Direct path: fresh engines, same seeds, no observer wrapper.
        engine = observer.bubble.new_engine(seed, retry=False, injected=False)
        import numpy as np
        from bubble_arena.observer import JUNK_SALT
        rng = np.random.default_rng([int(seed), JUNK_SALT])
        shape = engine.lattice.state.shape
        junk = rng.uniform(size=shape) + 1j * rng.uniform(size=shape)
        junk_ok, _reason = engine._validate(junk)
        result = engine.step()
        clean_dir.append(result["status"] != "accepted")
        junk_dir.append(not junk_ok)
    n = len(BAR2_SEEDS)
    rates = {
        "n_seeds": n,
        "clean_refusal_observer": _mean([1.0 if v else 0.0 for v in clean_obs]),
        "junk_refusal_observer": _mean([1.0 if v else 0.0 for v in junk_obs]),
        "clean_refusal_direct": _mean([1.0 if v else 0.0 for v in clean_dir]),
        "junk_refusal_direct": _mean([1.0 if v else 0.0 for v in junk_dir]),
        "observer_equals_direct": clean_obs == clean_dir and junk_obs == junk_dir,
    }
    return rates


def gate_aggregate_reports(observer, path="observer"):
    """Aggregate gate reports over BAR2_SEEDS, one per (path, probe type)."""
    if path == "observer":
        clean = [observer.measure_gate(s)[0] for s in BAR2_SEEDS]
        junk = [observer.measure_gate(s)[1] for s in BAR2_SEEDS]
    else:
        import numpy as np
        from bubble_arena.observer import JUNK_SALT
        clean, junk = [], []
        for seed in BAR2_SEEDS:
            engine = observer.bubble.new_engine(seed, retry=False, injected=False)
            rng = np.random.default_rng([int(seed), JUNK_SALT])
            shape = engine.lattice.state.shape
            junk_arr = rng.uniform(size=shape) + 1j * rng.uniform(size=shape)
            junk_ok, _r = engine._validate(junk_arr)
            result = engine.step()
            clean.append(result["status"] != "accepted")
            junk.append(not junk_ok)
    n = len(BAR2_SEEDS)
    reports = []
    for kind, vals in (("clean", clean), ("junk", junk)):
        p = _mean([1.0 if v else 0.0 for v in vals])
        reports.append(Report("gate-aggregate",
                              {f"{kind}_refusal_rate": p},
                              half_width=_binom_half_width(p, n), n=n))
    return reports


def calibrator_scenarios(bubble, observer):
    """Enumerate FailureCapsule refusals by reason, plus one applied case.

    Scenarios 1-3 use real gate-aggregate reports. Scenario 5 uses a
    gate-logic check (one report's half-width inflated to 0.25) and is
    labeled as such. The applied case uses four real aggregate reports
    (observer x direct, clean x junk); it applies only if every
    half-width is < 0.20, otherwise it is reported as a refusal.
    """
    cal = Calibrator(bubble)
    real_reports = gate_aggregate_reports(observer, "observer")
    applied_reports = real_reports + gate_aggregate_reports(observer, "direct")
    out = []

    saved = dict(bubble.gates)
    bubble.gates["gate_calibrated"] = False
    _v, cap = cal.calibrate(applied_reports)
    out.append(("gate_not_calibrated", cap.reason if cap else "APPLIED-UNEXPECTED"))
    bubble.gates.update(saved)

    bubble.gates["review_signoff"] = False
    _v, cap = cal.calibrate(applied_reports)
    out.append(("review_signoff_missing", cap.reason if cap else "APPLIED-UNEXPECTED"))
    # The gate check returns the FIRST failure in order (gate, signoff,
    # money), so the money case needs signoff True or it reports signoff.
    bubble.set_review_signoff(True)
    bubble.gates["money_gates_pass"] = False
    _v, cap = cal.calibrate(applied_reports)
    out.append(("money_gates_failed", cap.reason if cap else "APPLIED-UNEXPECTED"))
    bubble.gates.update(saved)

    bubble.set_review_signoff(True)
    _v, cap = cal.calibrate(applied_reports[:2])
    out.append(("too_few_reports", cap.reason if cap else "APPLIED-UNEXPECTED"))
    inflated = list(applied_reports)
    inflated[0] = Report(inflated[0].kind, inflated[0].value,
                         half_width=0.25, n=inflated[0].n)
    _v, cap = cal.calibrate(inflated)
    out.append(("half_width_too_wide_gate_logic_check",
                cap.reason if cap else "APPLIED-UNEXPECTED"))
    before_gate = bubble.frozen_gate()
    applied, cap = cal.calibrate(applied_reports)
    if cap is not None:
        out.append(("applied_case", f"REFUSED:{cap.reason}"))
    else:
        cal.apply(applied)
        after_gate = bubble.frozen_gate()
        out.append(("applied_case", {
            "assumed": applied,
            "frozen_gate_unchanged": before_gate == after_gate,
            "public_state": bubble.public_state(),
        }))
    bubble.gates.update(saved)
    bubble.set_review_signoff(False)
    return out
