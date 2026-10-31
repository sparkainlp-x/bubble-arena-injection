"""Density-matrix primitives for the synthetic membrane lattice.

All operations are classical NumPy simulations. No quantum hardware is used.
"""

from __future__ import annotations

import numpy as np

DIM = 32
EPS = 1e-12

def random_density_matrix(rng: np.random.Generator, dim: int = DIM) -> np.ndarray:
    """Return a valid random density matrix using the Ginibre construction."""
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = a @ a.conj().T
    rho /= np.trace(rho).real
    return project_valid(rho)

def project_valid(rho: np.ndarray) -> np.ndarray:
    """Project a matrix onto a Hermitian, positive, unit-trace matrix."""
    rho = (rho + rho.conj().T) / 2
    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(vals, 0.0, None)
    rho = (vecs * vals) @ vecs.conj().T
    trace = np.trace(rho).real
    if trace <= EPS:
        return np.eye(rho.shape[0], dtype=complex) / rho.shape[0]
    return rho / trace

def is_valid_density_matrix(rho: np.ndarray, tol: float = 1e-8) -> bool:
    """Check Hermiticity, unit trace, finiteness, and positive semidefiniteness."""
    if rho.shape[0] != rho.shape[1]:
        return False
    if not np.all(np.isfinite(rho)):
        return False
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    trace = np.trace(rho).real
    if trace < 1 - tol or trace > 1 + tol:
        return False
    vals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    return bool(np.min(vals) >= -tol)

def purity(rho: np.ndarray) -> float:
    """Return tr(rho^2), bounded to [0, 1]."""
    return float(np.clip(np.trace(rho @ rho).real, 0.0, 1.0))

def entropy(rho: np.ndarray) -> float:
    """Return von Neumann entropy in nats."""
    vals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    vals = np.clip(vals, EPS, None)
    return float(-np.sum(vals * np.log(vals)))

def normalized_entropy(rho: np.ndarray) -> float:
    """Return von Neumann entropy normalized by log(dim)."""
    return entropy(rho) / np.log(rho.shape[0])

def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Return squared root fidelity, used as a bounded resonance score."""
    sqrt_rho = _sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    vals = np.linalg.eigvalsh((inner + inner.conj().T) / 2)
    vals = np.clip(vals, 0.0, None)
    return float(np.clip(np.sum(np.sqrt(vals)) ** 2, 0.0, 1.0))

def _sqrtm(rho: np.ndarray) -> np.ndarray:
    """Return the principal matrix square root of a Hermitian PSD matrix."""
    vals, vecs = np.linalg.eigh((rho + rho.conj().T) / 2)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T

def mean_off_diagonal_coherence(rho: np.ndarray) -> float:
    """Return the mean |rho[i, j]| over off-diagonal entries.

    This is a true off-diagonal coherence measure, unlike mean purity
    (tr(rho^2)), which mixes diagonal populations with off-diagonal terms.
    """
    if rho.shape[0] < 2:
        return 0.0
    mask = ~np.eye(rho.shape[0], dtype=bool)
    return float(np.mean(np.abs(rho[mask])))


def max_off_diagonal_coherence(rho: np.ndarray) -> float:
    """Return the largest |rho[i, j]| over off-diagonal entries."""
    if rho.shape[0] < 2:
        return 0.0
    mask = ~np.eye(rho.shape[0], dtype=bool)
    return float(np.max(np.abs(rho[mask])))


def apply_phase_damping_tp(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Trace-preserving phase damping: damp off-diagonals only.

    rho[i, j] -> rho[i, j] * exp(-gamma * |i - j|) for i != j; the diagonal
    (populations) is left exactly unchanged, so the trace is preserved by
    construction and no renormalization is needed. The factor matrix
    exp(-gamma * |i - j|) is positive semidefinite, so this is a completely
    positive trace-preserving map. Unlike the legacy ``apply_dephasing``
    below, it cannot purify a state by shifting weight to low indices.
    """
    if gamma <= 0:
        return rho
    idx = np.arange(rho.shape[0])
    factors = np.exp(-gamma * np.abs(idx[:, None] - idx[None, :]))
    damped = rho * factors
    return (damped + damped.conj().T) / 2


def exact_unitary_step(rho: np.ndarray, hamiltonian: np.ndarray, dt: float) -> np.ndarray:
    """Return U rho U^dagger with U = exp(-i H dt), exactly.

    The Hermitian Hamiltonian is diagonalized once (H = V diag(e) V^dagger),
    so U = V diag(exp(-i e dt)) V^dagger is unitary to numerical precision.
    This preserves trace and purity exactly (up to round-off), unlike an
    explicit Euler step of the commutator. It is independently testable:
    a pure state stays pure and a mixed state's purity is unchanged.
    """
    vals, vecs = np.linalg.eigh((hamiltonian + hamiltonian.conj().T) / 2)
    phases = np.exp(-1j * vals * dt)
    unitary = (vecs * phases) @ vecs.conj().T
    out = unitary @ rho @ unitary.conj().T
    return (out + out.conj().T) / 2


def apply_dephasing(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Damp matrix entries by exp(-gamma * (i + j)) and renormalize.

    Despite the name this is not trace-preserving phase damping: the diagonal
    is damped too (by exp(-2 * gamma * i)), so after renormalization the
    populations shift toward low basis indices and the state becomes purer.
    The behavior is kept as published; see "Known limits" in the README.
    """
    if gamma <= 0:
        return rho
    factors = np.exp(-gamma * np.arange(rho.shape[0]))
    damped = rho * factors[:, None] * factors[None, :]
    return project_valid(damped)

def crystallization_mirror(rho: np.ndarray, strength: float) -> np.ndarray:
    """Apply a PHMT-style nonlinear purity filter: rho -> rho^g / tr(rho^g)."""
    if strength <= 0:
        return rho
    power = 1.0 + strength
    vals, vecs = np.linalg.eigh((rho + rho.conj().T) / 2)
    vals = np.clip(vals, 0.0, None)
    powered = vals**power
    total = powered.sum()
    if total <= EPS:
        return rho
    filtered = (vecs * (powered / total)) @ vecs.conj().T
    return project_valid(filtered)

def residual_infinity(rho: np.ndarray, reference: np.ndarray) -> float:
    """Return the infinity-norm distance from a reference state."""
    return float(np.max(np.abs(rho - reference)))
