"""Natural-log categorical entropy, optionally normalized by log(K)."""
import numpy as np
from numpy.typing import ArrayLike, NDArray
from ..validation.inputs import probability_array


def entropy(probabilities: ArrayLike, *, normalized: bool = False) -> float | NDArray[np.float64]:
    """Return H(p) in nats for [K], or row entropies for [N,K]; 0 log 0 = 0."""
    p = probability_array(probabilities)
    logs = np.zeros_like(p)
    np.log(p, out=logs, where=p > 0)
    value = -np.sum(p * logs, axis=-1)
    if normalized:
        value = value / np.log(p.shape[-1])
    return float(value) if p.ndim == 1 else value
