"""Small explicit validators; probability roundoff tolerance is absolute 1e-6."""
from collections.abc import Sequence
from numbers import Real
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

PROBABILITY_TOLERANCE = 1e-6


def nonnegative_number(value: float, *, name: str) -> float:
    """Return a finite nonnegative real, rejecting booleans and coercive strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def identifier(value: str, *, name: str = 'identifier') -> str:
    """Require a nonempty string without surrounding whitespace."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty with no surrounding whitespace")
    return value


def modality_names(values: Sequence[str], *, name: str = 'modalities') -> tuple[str, ...]:
    """Snapshot unique modality names without changing their supplied order."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of modality names")
    result = tuple(identifier(v, name='modality') for v in values)
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate modalities in {name}")
    return result


def probability_array(values: ArrayLike, *, ndim: int | None = None) -> NDArray[np.float64]:
    """Validate [K] or [N,K] probabilities; renormalize only roundoff within 1e-6.

    Every class dimension has K>=2. Values must be finite and within [0,1].
    No clipping or repair of invalid distributions is performed.
    """
    raw = np.asarray(values)
    if raw.dtype.kind not in 'fiu':
        raise TypeError('Probabilities must be real numeric values')
    p = np.asarray(raw, dtype=np.float64)
    if p.ndim not in (1, 2) or (ndim is not None and p.ndim != ndim):
        raise ValueError('Expected a probability vector or matrix with the requested rank')
    if p.size == 0 or p.shape[-1] < 2:
        raise ValueError('Probabilities must be nonempty with at least two classes')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Probabilities must be finite and within [0,1]')
    sums = p.sum(axis=-1, keepdims=True)
    if not np.all(np.abs(sums - 1) <= PROBABILITY_TOLERANCE):
        raise ValueError('Probability rows must sum to 1 within absolute tolerance 1e-6')
    return p / sums


def probability_vector(values: ArrayLike) -> tuple[float, ...]:
    """Return an immutable normalized snapshot of one validated class vector."""
    return tuple(float(v) for v in probability_array(values, ndim=1))
