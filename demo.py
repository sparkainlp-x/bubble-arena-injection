"""Demo: bar 1, bar 2, and calibrator scenarios. Run: python3 demo.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bubble_arena.bars import bar1, bar2_rates, calibrator_scenarios
from bubble_arena.bubble import BubbleArena
from bubble_arena.observer import BubbleObserver, half_width


def main() -> int:
    bubble = BubbleArena()
    observer = BubbleObserver(bubble)

    print("=== Bar 1: steps / accepted, stock vs injected, retry ON / OFF ===")
    _table, summary = bar1(observer)
    for retry in (True, False):
        for injected in (False, True):
            s = summary[(retry, injected)]
            print(f"retry={'ON' if retry else 'OFF'} injected={'YES' if injected else 'no'}: "
                  f"seed42 {s['seed42_steps']}/{s['seed42_accepted']} "
                  f"mean(0-19) {s['mean_steps_0_19']:.2f}/{s['mean_accepted_0_19']:.2f} "
                  f"split_matches_bare={s['all_split_match_bare']}")
    print("=== Bar 1 run-aggregate half-widths (steps across seeds 0-19, information only) ===")
    for retry in (True, False):
        for injected in (False, True):
            steps = [_table[(retry, injected, s)]["steps"] for s in range(20)]
            print(f"retry={'ON' if retry else 'OFF'} injected={'YES' if injected else 'no'}: "
                  f"steps half-width={half_width(steps):.3f} (n=20)")
    print("=== Bar 2a: frozen-gate refusal rates, held-out seeds 100-199 ===")
    rates = bar2_rates(observer)
    print(f"n_seeds={rates['n_seeds']} "
          f"clean(observer)={rates['clean_refusal_observer']:.3f} "
          f"junk(observer)={rates['junk_refusal_observer']:.3f} "
          f"clean(direct)={rates['clean_refusal_direct']:.3f} "
          f"junk(direct)={rates['junk_refusal_direct']:.3f} "
          f"observer_equals_direct={rates['observer_equals_direct']}")
    print("=== Bar 2b: calibrator refusals by reason (separate from 2a) ===")
    for name, outcome in calibrator_scenarios(bubble, observer):
        print(f"{name}: {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
