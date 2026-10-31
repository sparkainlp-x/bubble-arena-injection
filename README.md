# bubble-arena-injection v0.1.0

Membrane-arena 3x3 with Crystallize injection under the bubble split:
the arena engine lives inside the bubble, unchanged; outside it live
only an observer and a calibrator. The observer records REPORTED
outcomes without mutating the system; the calibrator updates an
assumed model only, or refuses with a FailureCapsule.

## The split, precisely

- **Inside:** the vendored `membrane_arena_3x3` engine (stock
  `MembraneArena3x3` and `MembraneArena3x3WithInjection`), byte-identical
  to the upgraded tree — hashes in `VENDORED_SHA256.txt`, pinned by a
  test. The engine's own internal shields (YangObserverShield /
  YinCalibratorShield) are part of that vendored code and drive each
  step; they are NOT this package's observer/calibrator.
- **Outside:** `bubble_arena.BubbleObserver` (pure-read snapshots:
  state hash, step, failures, breaker flag; run and probe outcomes
  recorded as REPORTED) and `bubble_arena.Calibrator` (fail-closed
  gates, assumed-model updates only).
- **Frozen gate:** the stock GuardConfig thresholds (residual 0.20,
  divergence 0.02–0.60, purity 0.05, entropy 0.50, failure_limit 5)
  plus the default InjectionConfig (max_attempts 3, first_commit 0.35,
  adaptive off, ledger-only). Never re-tuned on held-out probes;
  pinned by a test that compares the gate before and after all
  measurements.

## Bars (preregistered) and measured outcome

Bar 1: steps executed / accepted for stock vs injected, with stock
adaptive retry ON and OFF, on seed 42 and on seeds 0–19, measured
THROUGH the split (observed runs) and BARE (direct `.run()`), side by
side. A split-vs-bare mismatch would be a finding, never smoothed
over.

Bar 2a: frozen-gate per-probe refusal rates, clean vs junk, on
held-out seeds 100–199 (disjoint from bar 1's seeds), measured through
the observer and directly. Clean probe: the first step of a fresh
stock engine with adaptive retry OFF, so the outcome is the frozen
guard's decision alone. Junk probe: a uniform-random complex array of
the candidate shape (3, 3, 32, 32), validated by the same frozen
guard on the same fresh engine before the clean step.

Bar 2b: calibrator FailureCapsule refusals BY REASON, enumerated by
scenario — reported separately from 2a, never merged into one rate.
The applied case uses four real aggregate reports (observer × direct,
clean × junk) and applies only if every half-width is < 0.20.

## Measured outcome (demo.py, 2026-10-30)

Bar 1 — seed 42 reproduces the background exactly, and the split
changes nothing (split_matches_bare True for every run, all four
cells, all 21 seeds):

- Retry ON, stock: seed 42 **12 steps / 7 accepted**; mean over seeds
  0–19 **10.80 / 5.80**.
- Retry ON, injected: seed 42 **11 / 6**; mean **10.95 / 5.85**.
- Retry OFF, stock: seed 42 **8 / 3**; mean **8.45 / 3.45**.
- Retry OFF, injected: seed 42 **11 / 6**; mean **10.95 / 5.85**.

So both background facts survive the split AND extend to seeds 0–19:
with retry ON the injection adds almost nothing (10.80 → 10.95 steps,
5.80 → 5.85 accepted — retry subsumes the injection); with retry OFF
the injection extends the run (8.45 → 10.95 steps, 3.45 → 5.85
accepted). All runs still latch the circuit breaker eventually.
Run-aggregate half-widths for steps across seeds 0–19 are 0.524
(ON/stock), 0.438 (ON/injected), 0.224 (OFF/stock), 0.438
(OFF/injected) — all above the calibrator's 0.20 gate, so run
aggregates would be refused on half-width (see below); they are
reported here as information, not fed to the calibrator.

Bar 2a — frozen-gate refusal rates, 100 held-out seeds (100–199),
observer path identical to direct path:

- Clean refusal **0.000** (100/100 genuine first-step candidates
  accepted), junk refusal **1.000** (100/100 randomized candidates
  refused), observer_equals_direct True.

Bar 2b — calibrator refusals by reason, each scenario producing
exactly its reason: gate_not_calibrated, review_signoff_missing,
money_gates_failed, too_few_reports, half_width_too_wide (the last a
labeled gate-logic check with an inflated half-width). Applied case:
assumed_clean_refusal **0.0**, assumed_junk_refusal **1.0**,
frozen_gate_unchanged True. Step and injection assumptions stay None
in the applied case because no run/injection reports were supplied —
the run aggregates' half-widths (0.224–0.524) exceed the 0.20 gate, so
a calibrator fed run aggregates would refuse them; that refusal is
the gate working, not a missing result.

One bug caught by reading the demo output before delivery: the first
money-gates scenario reported review_signoff_missing, because the
gate check returns the first failure in order and signoff was still
unset. The scenario setup was fixed (signoff set before the money
check); the calibrator's order was correct throughout. The fix was
re-verified with real aggregates, and a money_gates_failed case was
added to the tests.

## Run it

```sh
python3 demo.py            # bar 1, bar 2, calibrator scenarios
python3 tests/tests.py     # 9 tests, from the package root
```

Tests pin the mechanism (non-invasion on seeds 42/0/1, determinism,
frozen gate, gate-path agreement, calibrator gate logic, vendored
bytes, NumPy declaration) plus the upstream-pinned seed-42 background
cells. Full bar 1 (seeds 0–19) and bar 2 (seeds 100–199) numbers are
measured by demo.py and reported below, not re-run in the tests, so
the suite stays fast.

## Honest limits

- **NumPy exception, declared:** this package is NOT stdlib-only.
  The arena engine uses NumPy throughout, and `pyproject.toml`
  declares numpy as a dependency. No stdlib port was written: a port
  would change `np.random.default_rng` trajectories and break the
  bar-1 comparison against the background numbers. This exception is
  for bubble-arena-injection only.
- **Binary decisions recorded verbatim:** the observer adds no noise
  to an accept/reject decision — noise on a binary outcome would
  corrupt the measurement. The "noise" in a probe is the probe itself
  (a junk probe is a randomized candidate). Continuous aggregates
  carry honest half-widths (binomial for refusal rates, 95% CI for
  step counts).
- **Probe engines are separate from run engines.** Probing uses fresh
  engines on held-out seeds; no probe ever touches a bar-1 run.
- **Classical synthetic toy only.** No biology, medical, quantum,
  brain, or consciousness claim. Both bar-1 runs still latch the
  circuit breaker eventually; the injection measures the structural
  problem, it does not cure it.
- **The calibrator's assumed model is not the system.** An applied
  calibration changes what the outside world believes, never a guard,
  a threshold, or an engine parameter.

## Licence

AGPL-3.0-only. See `LICENSE`.
