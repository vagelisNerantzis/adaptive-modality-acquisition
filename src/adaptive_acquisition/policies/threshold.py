"""Strict minimum utility criterion used by the greedy policy."""
import math
from numbers import Real


def finite_score(value: float, *, name: str = 'utility') -> float:
    """Require a finite real score, allowing scientifically meaningful negatives."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a finite real number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def above_threshold(score: float, minimum: float | None = 0.0) -> bool:
    """Accept strictly above minimum; None disables utility-based stopping."""
    score = finite_score(score)
    return minimum is None or score > finite_score(minimum, name='minimum utility')
