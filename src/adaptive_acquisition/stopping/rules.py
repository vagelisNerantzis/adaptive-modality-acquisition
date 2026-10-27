"""Composable stopping reasons and eligible acquisition sets."""
from dataclasses import dataclass
from typing import Protocol
from ..state.acquisition_state import AcquisitionState
from ..costs.models import FixedCostModel
from ..validation.inputs import nonnegative_number, modality_names


class StopRule(Protocol):
    """A reason plus eligibility constraint; returning None means continue."""
    def reason(self, state: AcquisitionState, costs: FixedCostModel) -> str | None:
        """Return a reached stopping reason, otherwise None."""
        ...

    def candidates(self, state: AcquisitionState, costs: FixedCostModel) -> tuple[str, ...]:
        """Return currently eligible unacquired modalities."""
        ...


@dataclass(frozen=True)
class ConfidenceThresholdStop:
    """Stop at max(p)>=threshold, with threshold in (0,1]."""
    threshold: float

    def __post_init__(self) -> None:
        value = nonnegative_number(self.threshold, name='confidence threshold')
        if not 0 < value <= 1:
            raise ValueError('confidence threshold must be in (0,1]')
        object.__setattr__(self, 'threshold', value)

    def reason(self, state: AcquisitionState, costs: FixedCostModel) -> str | None:
        """Return confidence when the current validated prediction is sufficient."""
        return 'confidence' if max(state.probabilities) >= self.threshold else None

    def candidates(self, state: AcquisitionState, costs: FixedCostModel) -> tuple[str, ...]:
        """Confidence imposes no cost eligibility filter; reason gates selection."""
        return state.remaining_modalities


@dataclass(frozen=True)
class BudgetStop:
    """Stop when no remaining single acquisition fits a finite total-cost budget."""
    max_budget: float

    def __post_init__(self) -> None:
        object.__setattr__(self, 'max_budget', nonnegative_number(self.max_budget, name='budget'))

    def candidates(self, state: AcquisitionState, costs: FixedCostModel) -> tuple[str, ...]:
        """Filter before any lookahead; strict float <= prevents budget overspending."""
        if state.total_cost > self.max_budget:
            raise ValueError('Current state cost already exceeds maximum budget')
        return tuple(name for name in state.remaining_modalities
                     if state.total_cost + costs.cost(name) <= self.max_budget)

    def reason(self, state: AcquisitionState, costs: FixedCostModel) -> str | None:
        """Return budget only if unacquired modalities exist but none are affordable."""
        eligible = self.candidates(state, costs)
        return 'budget' if state.remaining_modalities and not eligible else None


@dataclass(frozen=True)
class NoModalitiesStop:
    """Stop when this physical sample has no unacquired modalities left."""
    def reason(self, state: AcquisitionState, costs: FixedCostModel) -> str | None:
        """Return the explicit evidence-exhaustion reason."""
        return None if state.remaining_modalities else 'no_modalities'

    def candidates(self, state: AcquisitionState, costs: FixedCostModel) -> tuple[str, ...]:
        """Return the unchanged remaining modality set."""
        return state.remaining_modalities


@dataclass(frozen=True)
class AnyStop:
    """First reached reason wins; eligibility is the intersection of all rules."""
    rules: tuple[StopRule, ...]

    def __post_init__(self) -> None:
        values = tuple(self.rules)
        if any(not callable(getattr(rule, 'reason', None)) or not callable(getattr(rule, 'candidates', None)) for rule in values):
            raise TypeError('Each stopping rule must implement reason and candidates')
        object.__setattr__(self, 'rules', values)

    def reason(self, state: AcquisitionState, costs: FixedCostModel) -> str | None:
        """Return the first reached rule's reason in configured priority order."""
        for rule in self.rules:
            reason = rule.reason(state, costs)
            if reason is not None:
                return reason
        return None

    def candidates(self, state: AcquisitionState, costs: FixedCostModel) -> tuple[str, ...]:
        """Intersect every rule's eligible set, validating budget constraints first."""
        remaining = set(state.remaining_modalities)
        eligible = remaining.copy()
        for rule in self.rules:
            supplied = modality_names(rule.candidates(state, costs), name='eligible candidates')
            if not set(supplied).issubset(remaining):
                raise ValueError('Stopping rule returned unknown or acquired candidates')
            eligible.intersection_update(supplied)
        return tuple(sorted(eligible))
