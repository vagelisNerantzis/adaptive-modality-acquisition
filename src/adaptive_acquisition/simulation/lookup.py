"""Bound-sample synthetic prediction tables, not a trained information-gain oracle."""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol
from ..validation.inputs import identifier, modality_names, probability_vector


class PredictionProvider(Protocol):
    """External forecast/prediction contract; no labels or fitting are needed here."""
    def predict(self, sample_id: str, acquired_modalities: frozenset[str]) -> tuple[float, ...]:
        """Supply the declared belief for this sample and evidence subset."""
        ...


@dataclass(frozen=True)
class LookupPredictor:
    """Deterministic subset-to-probability forecasts bound to exactly one sample."""
    sample_id: str
    predictions: Mapping[frozenset[str], tuple[float, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'sample_id', identifier(self.sample_id, name='sample_id'))
        if not isinstance(self.predictions, Mapping) or not self.predictions:
            raise ValueError('prediction lookup must be a nonempty mapping')
        values = {}
        for subset, probabilities in self.predictions.items():
            if not isinstance(subset, frozenset):
                raise TypeError('Prediction lookup keys must be frozenset modality subsets')
            modality_names(tuple(subset))
            values[subset] = probability_vector(probabilities)
        if len({len(p) for p in values.values()}) != 1:
            raise ValueError('Prediction lookup entries have inconsistent class dimension')
        object.__setattr__(self, 'predictions', MappingProxyType(values))

    def predict(self, sample_id: str, acquired_modalities: frozenset[str]) -> tuple[float, ...]:
        """Look up one bound sample/subset; missing evidence states never fall back."""
        if sample_id != self.sample_id:
            raise ValueError('Prediction lookup requested for unknown physical sample')
        if not isinstance(acquired_modalities, frozenset):
            raise TypeError('Acquired modality subset must be a frozenset')
        modality_names(tuple(acquired_modalities))
        if acquired_modalities not in self.predictions:
            raise ValueError('Missing prediction state for acquired modality subset')
        return self.predictions[acquired_modalities]
