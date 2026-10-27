"""Supplied predictive entropy reduction minus an explicit acquisition-cost penalty."""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
import math
from ..state.acquisition_state import AcquisitionState
from ..costs.models import FixedCostModel
from ..simulation.lookup import PredictionProvider
from ..validation.inputs import modality_names, nonnegative_number, probability_vector
from .entropy import entropy


class CandidateScorer(Protocol):
    """Extension point for externally implemented utilities, without hidden fitting."""
    def score(self, state: AcquisitionState, candidates: Sequence[str],
              cost_model: FixedCostModel, predictor: PredictionProvider) -> dict[str, float]:
        """Score the supplied eligible unacquired modalities."""
        ...


@dataclass(frozen=True)
class EntropyReductionScorer:
    """U=H(current)-H(supplied future belief)-cost_weight*cost, in common units.

    One supplied future vector is a point forecast; its entropy is not generally
    the expected entropy of a distribution over future outcomes.
    """
    cost_weight: float = 0.15

    def __post_init__(self) -> None:
        object.__setattr__(self, 'cost_weight', nonnegative_number(self.cost_weight, name='cost_weight'))

    def score(self, state: AcquisitionState, candidates: Sequence[str],
              cost_model: FixedCostModel, predictor: PredictionProvider) -> dict[str, float]:
        """Return finite utilities without using a ground-truth label."""
        names = modality_names(candidates, name='candidates')
        if not set(names).issubset(state.remaining_modalities):
            raise ValueError('Candidate modality is unknown or already acquired')
        current = entropy(state.probabilities)
        scores = {}
        for name in sorted(names):
            p = probability_vector(predictor.predict(state.sample_id, frozenset((*state.acquired_modalities, name))))
            if len(p) != len(state.probabilities):
                raise ValueError('Forecast class dimension differs from current state')
            value = float(current - entropy(p) - self.cost_weight * cost_model.cost(name))
            if not math.isfinite(value):
                raise ValueError('Candidate utility is nonfinite')
            scores[name] = value
        return scores
