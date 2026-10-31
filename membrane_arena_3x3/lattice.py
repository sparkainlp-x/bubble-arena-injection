"""3x3 lattice of coupled synthetic membrane cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (
    DIM,
    apply_dephasing,
    apply_phase_damping_tp,
    crystallization_mirror,
    exact_unitary_step,
    fidelity,
    project_valid,
    random_density_matrix,
)

@dataclass
class LatticeConfig:
    """Configuration for the nine-cell lattice and its shared interaction rule."""

    dim: int = DIM
    coupling: float = 0.02
    coupling_min: float = 0.0
    coupling_max: float = 0.10
    dephasing: float = 0.5
    dephasing_min: float = 0.0
    dephasing_max: float = 2.0
    mirror_strength: float = 0.10
    mirror_min: float = 0.0
    mirror_max: float = 0.50
    resonance_threshold: float = 0.90
    dt: float = 0.05
    # "legacy" reproduces the published update exactly (Euler commutator
    # step + the non-trace-preserving apply_dephasing). "exact" uses the
    # structure-preserving stages below: exact unitary conjugation and
    # trace-preserving off-diagonal phase damping. The benchmark sweeps
    # both, so the improvement (or lack of it) is measured, not asserted.
    operator_mode: str = "legacy"

    def __post_init__(self) -> None:
        if self.operator_mode not in ("legacy", "exact"):
            raise ValueError("operator_mode must be 'legacy' or 'exact'")

class MembraneLattice:
    """Nine density-matrix cells arranged as a 3x3 grid.

    Index convention: `state[i, j]`, where `i` is the arena and `j` is the
    membrane inside that arena.
    """

    def __init__(self, config: LatticeConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.state = np.empty((3, 3, config.dim, config.dim), dtype=complex)
        self.hamiltonians = np.empty_like(self.state)

        for i in range(3):
            for j in range(3):
                self.state[i, j] = random_density_matrix(rng, config.dim)
                h = rng.normal(size=(config.dim, config.dim)) + 1j * rng.normal(
                    size=(config.dim, config.dim)
                )
                self.hamiltonians[i, j] = (h + h.conj().T) / 2

        self.reference = self.state.copy()

    def neighbors(self, i: int, j: int) -> list[tuple[int, int]]:
        """Return same-arena neighbors and corresponding cells in other arenas."""
        cells: list[tuple[int, int]] = []

        for jj in range(3):
            if jj != j:
                cells.append((i, jj))

        for ii in range(3):
            if ii != i:
                cells.append((ii, j))

        return cells

    def interaction_delta(
        self, i: int, j: int, coupling: float | None = None
    ) -> np.ndarray:
        """Compute the shared interaction contribution for one membrane cell."""
        cfg = self.config
        rho = self.state[i, j]
        delta = np.zeros_like(rho)
        coupling_eff = cfg.coupling if coupling is None else coupling

        for a, b in self.neighbors(i, j):
            other = self.state[a, b]
            score = fidelity(rho, other)
            drive = coupling_eff * (score - cfg.resonance_threshold)
            delta += drive * (other - rho)

        return delta / max(len(self.neighbors(i, j)), 1)

    # --- Independently testable evolution stages -------------------------
    def unitary_stage(
        self, rho: np.ndarray, hamiltonian: np.ndarray, dt: float,
        operator_mode: str | None = None,
    ) -> np.ndarray:
        """One unitary stage. Exact mode preserves trace and purity."""
        mode = operator_mode or self.config.operator_mode
        if mode == "exact":
            return exact_unitary_step(rho, hamiltonian, dt)
        commutator = hamiltonian @ rho - rho @ hamiltonian
        return rho - 1j * dt * commutator

    def dissipative_stage(
        self, rho: np.ndarray, gamma: float,
        operator_mode: str | None = None,
    ) -> np.ndarray:
        """One dissipative stage. Exact mode is trace-preserving."""
        mode = operator_mode or self.config.operator_mode
        if mode == "exact":
            return apply_phase_damping_tp(rho, gamma)
        return apply_dephasing(rho, gamma)

    def mirror_stage(self, rho: np.ndarray, strength: float) -> np.ndarray:
        """One nonlinear purity-filter stage."""
        return crystallization_mirror(rho, strength)

    def propose_step(
        self,
        calibrator_signal: float = 0.0,
        dt: float | None = None,
        mirror_strength: float | None = None,
        coupling: float | None = None,
        dephasing: float | None = None,
        operator_mode: str | None = None,
    ) -> np.ndarray:
        """Return a candidate next state without mutating the current lattice.

        Overrides let adaptive retry (and the benchmark) propose the same
        step with a smaller dt, a softened mirror and a reduced coupling
        without mutating the lattice configuration. In legacy mode the
        formula reproduces the published update exactly; in exact mode
        the stages compose as unitary -> trace-preserving damping ->
        interaction + calibrator pull -> numerical projection -> mirror.
        """
        cfg = self.config
        dt_eff = cfg.dt if dt is None else dt
        mirror_eff = cfg.mirror_strength if mirror_strength is None else mirror_strength
        coupling_eff = cfg.coupling if coupling is None else coupling
        dephasing_eff = cfg.dephasing if dephasing is None else dephasing
        mode = operator_mode or cfg.operator_mode
        candidate = np.empty_like(self.state)

        for i in range(3):
            for j in range(3):
                rho = self.state[i, j]
                h = self.hamiltonians[i, j]

                if mode == "legacy":
                    commutator = h @ rho - rho @ h
                    unitary = -1j * dt_eff * commutator
                    dissipative = apply_dephasing(rho, dephasing_eff * dt_eff)
                    delta = unitary + (dissipative - rho)
                    delta += self.interaction_delta(i, j, coupling_eff) * dt_eff
                    if calibrator_signal:
                        delta += calibrator_signal * dt_eff * (
                            self.reference[i, j] - rho
                        )
                    nxt = project_valid(rho + delta)
                else:
                    nxt = self.unitary_stage(rho, h, dt_eff, mode)
                    nxt = self.dissipative_stage(nxt, dephasing_eff * dt_eff, mode)
                    nxt = nxt + self.interaction_delta(i, j, coupling_eff) * dt_eff
                    if calibrator_signal:
                        nxt = nxt + calibrator_signal * dt_eff * (
                            self.reference[i, j] - rho
                        )
                    nxt = project_valid(nxt)

                candidate[i, j] = self.mirror_stage(nxt, mirror_eff)

        return candidate

    def commit(self, candidate: np.ndarray) -> None:
        """Atomically replace the lattice state with a checked candidate."""
        self.state = candidate.copy()
