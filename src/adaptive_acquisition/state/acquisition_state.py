"""Immutable physical-sample state and auditable acquisition transitions."""
from dataclasses import dataclass
import math
from numbers import Integral
from ..validation.inputs import identifier, modality_names, nonnegative_number, probability_vector


@dataclass(frozen=True)
class AcquisitionRecord:
    """One newly acquired modality with its resulting cost and class probabilities."""
    step_index: int
    modality: str
    cost: float
    total_cost: float
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, Integral) or self.step_index < 1:
            raise ValueError('step_index must be a positive integer')
        object.__setattr__(self, 'modality', identifier(self.modality, name='modality'))
        object.__setattr__(self, 'cost', nonnegative_number(self.cost, name='cost'))
        object.__setattr__(self, 'total_cost', nonnegative_number(self.total_cost, name='total_cost'))
        object.__setattr__(self, 'probabilities', probability_vector(self.probabilities))
        if self.cost > self.total_cost:
            raise ValueError('Acquisition cost exceeds cumulative cost')


@dataclass(frozen=True)
class AcquisitionState:
    """One sample's evidence; initial acquired evidence/cost are supplied by caller.

    History records new transitions only. Probabilities and modality collections
    are copied into immutable tuples, so external input mutation cannot leak in.
    """
    sample_id: str
    available_modalities: tuple[str, ...]
    probabilities: tuple[float, ...]
    acquired_modalities: tuple[str, ...] = ()
    total_cost: float = 0.0
    history: tuple[AcquisitionRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'sample_id', identifier(self.sample_id, name='sample_id'))
        available = modality_names(self.available_modalities, name='available_modalities')
        acquired = modality_names(self.acquired_modalities, name='acquired_modalities')
        if not set(acquired).issubset(available):
            raise ValueError('Acquired modality is unknown to this physical sample')
        object.__setattr__(self, 'available_modalities', available)
        object.__setattr__(self, 'acquired_modalities', acquired)
        object.__setattr__(self, 'probabilities', probability_vector(self.probabilities))
        object.__setattr__(self, 'total_cost', nonnegative_number(self.total_cost, name='total_cost'))
        events = tuple(self.history)
        if not all(isinstance(event, AcquisitionRecord) for event in events):
            raise TypeError('history must contain AcquisitionRecord values')
        if events:
            if tuple(event.modality for event in events) != acquired[-len(events):]:
                raise ValueError('History does not match acquired modality order')
            if len(events) > len(acquired):
                raise ValueError('History contains too many acquisitions')
            for i, event in enumerate(events):
                if event.step_index != i + 1 or len(event.probabilities) != len(self.probabilities):
                    raise ValueError('History step or class dimension mismatch')
                if i and not math.isclose(event.total_cost, events[i-1].total_cost + event.cost, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('History cumulative cost mismatch')
            if not math.isclose(events[-1].total_cost, self.total_cost, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Final history cost differs from state')
            if any(abs(a-b) > 1e-12 for a,b in zip(events[-1].probabilities, self.probabilities)):
                raise ValueError('Final history prediction differs from state')
        object.__setattr__(self, 'history', events)

    @property
    def remaining_modalities(self) -> tuple[str, ...]:
        """Return unacquired candidate names in lexical order."""
        return tuple(sorted(set(self.available_modalities) - set(self.acquired_modalities)))

    @property
    def step_index(self) -> int:
        """Number of transitions recorded after the original supplied evidence."""
        return len(self.history)

    def acquire(self, modality: str, probabilities: tuple[float, ...], cost: float) -> 'AcquisitionState':
        """Return a new state with one distinct modality acquired, never mutate self."""
        identifier(modality, name='modality')
        if modality not in self.available_modalities:
            raise ValueError('Unknown modality for this sample')
        if modality in self.acquired_modalities:
            raise ValueError('Cannot acquire a modality twice')
        p = probability_vector(probabilities)
        if len(p) != len(self.probabilities):
            raise ValueError('Prediction class dimension changed during acquisition')
        cost = nonnegative_number(cost, name='cost')
        total = nonnegative_number(self.total_cost + cost, name='total_cost')
        event = AcquisitionRecord(self.step_index + 1, modality, cost, total, p)
        return AcquisitionState(self.sample_id, self.available_modalities, p,
                                self.acquired_modalities + (modality,), total, self.history + (event,))
