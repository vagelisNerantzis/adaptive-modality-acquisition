"""One-step maximum utility acquisition with deterministic lexical tie-breaking."""
from collections.abc import Mapping
from dataclasses import dataclass
from ..state.acquisition_state import AcquisitionState
from ..validation.inputs import identifier
from .threshold import above_threshold, finite_score


@dataclass(frozen=True)
class Selection:
    """One selected modality and its finite utility score."""
    modality: str
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, 'modality', identifier(self.modality, name='modality'))
        object.__setattr__(self, 'score', finite_score(self.score))


@dataclass(frozen=True)
class GreedyPolicy:
    """Select an eligible utility maximum; ties use lexical modality order.

    Utilities must describe an explicit subset of unacquired candidates, e.g.
    those affordable under a budget. The caller supplies a reached stop reason
    to prohibit selection before any scoring. Default accepts only utility>0.
    """
    min_utility: float | None = 0.0

    def __post_init__(self) -> None:
        if self.min_utility is not None:
            object.__setattr__(self, 'min_utility', finite_score(self.min_utility, name='minimum utility'))

    def select(self, state: AcquisitionState, utilities: Mapping[str, float],
               stop_reason: str | None = None) -> Selection | None:
        """Return a best decision or None when stopped/below threshold; empty maps fail."""
        if stop_reason is not None:
            identifier(stop_reason, name='stop_reason')
            return None
        if not isinstance(utilities, Mapping):
            raise TypeError('utilities must be a mapping')
        if not utilities:
            raise ValueError('Cannot select from empty candidate utilities')
        values = {identifier(name, name='modality'): finite_score(score) for name, score in utilities.items()}
        if not set(values).issubset(state.remaining_modalities):
            raise ValueError('Selection candidate is unknown or already acquired')
        best = min(values, key=lambda name: (-values[name], name))
        if not above_threshold(values[best], self.min_utility):
            return None
        return Selection(best, values[best])
