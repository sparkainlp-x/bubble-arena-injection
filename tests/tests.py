"""Tests for bubble-arena-injection. Run from the package root: python3 tests/tests.py.

Tests pin the MECHANISM (non-invasion, determinism, frozen gate, gate
logic, vendored bytes, NumPy declaration) plus the upstream-pinned
seed-42 background cells. Full bar 1 (seeds 0-19) and bar 2 (seeds
100-199) numbers are measured by demo.py and reported in the README,
not re-run here, so the suite stays fast.
"""

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bubble_arena.bubble import BubbleArena  # noqa: E402
from bubble_arena.calibrator import Calibrator  # noqa: E402
from bubble_arena.observer import BubbleObserver, Report  # noqa: E402


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.bubble = BubbleArena()
        self.observer = BubbleObserver(self.bubble)

    def test_snapshot_does_not_mutate(self):
        engine = self.bubble.new_engine(42, retry=True, injected=False)
        before = self.observer.snapshot(engine)
        for _ in range(3):
            self.observer.snapshot(engine)
        after = self.observer.snapshot(engine)
        self.assertEqual(before, after)
        self.assertEqual(len(engine.calibrations), 0)
        self.assertEqual(len(engine.observations), 0)

    def test_observed_equals_bare_subset(self):
        for seed in (42, 0, 1):
            for retry in (True, False):
                for injected in (False, True):
                    summary_obs, _s, eng_obs = self.observer.observed_run(
                        seed, retry, injected)
                    summary_bare, eng_bare = self.observer.bare_run(
                        seed, retry, injected)
                    self.assertEqual(summary_obs["steps_executed"],
                                     summary_bare["steps_executed"],
                                     (seed, retry, injected))
                    self.assertEqual(summary_obs["accepted_steps"],
                                     summary_bare["accepted_steps"],
                                     (seed, retry, injected))
                    self.assertEqual(summary_obs["status"], summary_bare["status"],
                                     (seed, retry, injected))
                    if injected:
                        self.assertEqual(eng_obs.injections, eng_bare.injections,
                                         (seed, retry, injected))

    def test_bar1_background_seed42(self):
        # Upstream-pinned cells (membrane-arena-3x3 injection tests).
        cases = {
            (True, False): (12, 7),
            (True, True): (11, 6),
        }
        for (retry, injected), (steps, accepted) in cases.items():
            summary, _e = self.observer.bare_run(42, retry, injected)
            self.assertEqual((summary["steps_executed"], summary["accepted_steps"]),
                             (steps, accepted), (retry, injected))
            self.assertEqual(summary["status"], "circuit_breaker_open")
        stock_off, _e = self.observer.bare_run(42, False, False)
        inj_off, _e2 = self.observer.bare_run(42, False, True)
        self.assertEqual(stock_off["steps_executed"], 8)
        self.assertGreater(inj_off["steps_executed"], stock_off["steps_executed"])
        self.assertGreater(inj_off["accepted_steps"], stock_off["accepted_steps"])
        self.assertEqual(inj_off["status"], "circuit_breaker_open")

    def test_determinism(self):
        s1, _s, e1 = self.observer.observed_run(42, True, True)
        s2, _s2, e2 = self.observer.observed_run(42, True, True)
        self.assertEqual((s1["steps_executed"], s1["accepted_steps"], s1["status"]),
                         (s2["steps_executed"], s2["accepted_steps"], s2["status"]))
        self.assertEqual(e1.injections, e2.injections)

    def test_frozen_gate_unchanged(self):
        before = self.bubble.frozen_gate()
        for seed in (100, 101, 102):
            self.observer.measure_gate(seed)
        self.observer.observed_run(42, True, True)
        self.assertEqual(self.bubble.frozen_gate(), before)

    def test_gate_paths_agree_subset(self):
        for seed in (100, 101, 102, 103, 104):
            clean_ref, junk_ref, _cr, _jr = self.observer.measure_gate(seed)
            engine = self.bubble.new_engine(seed, retry=False, injected=False)
            import numpy as np
            from bubble_arena.observer import JUNK_SALT
            rng = np.random.default_rng([int(seed), JUNK_SALT])
            shape = engine.lattice.state.shape
            junk = rng.uniform(size=shape) + 1j * rng.uniform(size=shape)
            junk_ok, _r = engine._validate(junk)
            result = engine.step()
            self.assertEqual(clean_ref, result["status"] != "accepted", seed)
            self.assertEqual(junk_ref, not junk_ok, seed)

    def test_calibrator_gate_logic(self):
        # Gate-logic checks with hand-built reports (the measured
        # scenarios run in demo.py). Nothing here measures the arena.
        cal = Calibrator(self.bubble)
        good = [Report("gate-aggregate",
                       {"clean_refusal_rate": 0.1, "junk_refusal_rate": 0.9},
                       half_width=0.05, n=20)
                for _ in range(3)]
        saved = dict(self.bubble.gates)
        self.bubble.gates["gate_calibrated"] = False
        _v, cap = cal.calibrate(good)
        self.assertEqual(cap.reason, "gate_not_calibrated")
        self.bubble.gates.update(saved)
        _v, cap = cal.calibrate(good)
        self.assertEqual(cap.reason, "review_signoff_missing")
        self.bubble.set_review_signoff(True)
        self.bubble.gates["money_gates_pass"] = False
        _v, cap = cal.calibrate(good)
        self.assertEqual(cap.reason, "money_gates_failed")
        self.bubble.gates.update(saved)
        self.bubble.set_review_signoff(True)
        _v, cap = cal.calibrate(good[:2])
        self.assertEqual(cap.reason, "too_few_reports")
        bad = list(good)
        bad[0] = Report("gate-aggregate",
                        {"clean_refusal_rate": 0.1, "junk_refusal_rate": 0.9},
                        half_width=0.25, n=20)
        _v, cap = cal.calibrate(bad)
        self.assertEqual(cap.reason, "half_width_too_wide")
        before = self.bubble.frozen_gate()
        applied, cap = cal.calibrate(good)
        self.assertIsNone(cap)
        self.assertTrue(cal.apply(applied))
        self.assertEqual(self.bubble.frozen_gate(), before)
        self.bubble.gates.update(saved)
        self.bubble.set_review_signoff(False)

    def test_vendored_bytes_match_provenance(self):
        recorded = {}
        for line in (ROOT / "VENDORED_SHA256.txt").read_text().splitlines():
            digest, path = line.split(None, 1)
            recorded[path.strip()] = digest
        vendored = ROOT / "membrane_arena_3x3"
        for path, digest in recorded.items():
            actual = hashlib.sha256((vendored / path).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, path)

    def test_numpy_declared(self):
        pyproject = (ROOT / "pyproject.toml").read_text()
        self.assertIn("numpy", pyproject)
        readme = (ROOT / "README.md").read_text()
        self.assertIn("NumPy", readme)


if __name__ == "__main__":
    unittest.main(verbosity=2)
