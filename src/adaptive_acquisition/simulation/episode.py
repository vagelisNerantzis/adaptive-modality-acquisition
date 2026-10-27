"""Finite, deterministic acquisition episodes with explicit terminal trace records."""
from dataclasses import dataclass
import math
from ..state.acquisition_state import AcquisitionState
from ..costs.models import FixedCostModel
from ..policies.greedy import GreedyPolicy
from ..stopping.rules import AnyStop, NoModalitiesStop
from ..utility.scoring import CandidateScorer, EntropyReductionScorer
from ..validation.inputs import identifier, probability_vector
from .lookup import PredictionProvider


@dataclass(frozen=True)
class DecisionRecord:
    """Pre-decision state and all eligible utilities, including the terminal reason."""
    state: AcquisitionState
    candidate_utilities: tuple[tuple[str, float], ...]
    selected_modality: str | None
    stop_reason: str | None


@dataclass(frozen=True)
class EpisodeResult:
    """One physical sample's initial/final states and full audited decision trace."""
    initial_state: AcquisitionState
    final_state: AcquisitionState
    decision_history: tuple[DecisionRecord, ...]
    stop_reason: str

    def __post_init__(self) -> None:
        identifier(self.stop_reason, name='stop_reason')
        if self.initial_state.sample_id != self.final_state.sample_id:
            raise ValueError('Episode result cannot change physical sample identity')
        if len(self.initial_state.probabilities) != len(self.final_state.probabilities):
            raise ValueError('Episode result cannot change class dimension')
        if self.final_state.history[:self.initial_state.step_index] != self.initial_state.history:
            raise ValueError('Final state does not preserve initial acquisition history')
        if self.final_state.acquired_modalities[:len(self.initial_state.acquired_modalities)] != self.initial_state.acquired_modalities:
            raise ValueError('Final state does not preserve initial evidence')
        new_events = self.final_state.history[self.initial_state.step_index:]
        if self.final_state.acquired_modalities != self.initial_state.acquired_modalities + tuple(event.modality for event in new_events):
            raise ValueError('New evidence must correspond exactly to acquisition history')
        expected_cost = self.initial_state.total_cost
        for event in new_events:
            expected_cost += event.cost
            if event.total_cost != expected_cost:
                raise ValueError('Episode history does not preserve cumulative cost')
        if self.final_state.total_cost != expected_cost:
            raise ValueError('Final cost must equal initial cost plus new acquisition costs')
        if not new_events and self.final_state.probabilities != self.initial_state.probabilities:
            raise ValueError('Prediction cannot change without a recorded acquisition')
        object.__setattr__(self, 'decision_history', tuple(self.decision_history))

    @property
    def selected_modalities(self) -> tuple[str, ...]:
        """New acquisitions in this episode; initially available evidence is excluded."""
        return tuple(event.modality for event in self.final_state.history[self.initial_state.step_index:])

    @property
    def n_steps(self) -> int:
        """Number of additional acquisition transitions in this episode."""
        return self.final_state.step_index - self.initial_state.step_index

    @property
    def total_cost(self) -> float:
        """Final cumulative cost, including any caller-supplied initial cost."""
        return self.final_state.total_cost

    @property
    def additional_cost(self) -> float:
        """Sum of newly incurred event costs, excluding supplied initial cost."""
        return math.fsum(event.cost for event in self.final_state.history[self.initial_state.step_index:])

    @property
    def final_probabilities(self) -> tuple[float, ...]:
        """Final ordered class distribution; class meanings are caller-declared."""
        return self.final_state.probabilities

    @property
    def final_confidence(self) -> float:
        """Maximum final class probability, not a guarantee of correctness."""
        return max(self.final_probabilities)


def run_episode(initial_state: AcquisitionState, predictor: PredictionProvider,
                cost_model: FixedCostModel, scorer: CandidateScorer | None = None,
                policy: GreedyPolicy | None = None, stopping: AnyStop | None = None) -> EpisodeResult:
    """Run until exhaustion, a composed rule, or the policy's utility threshold.

    All candidate forecasts are supplied externally. Budgets filter candidates
    before forecasts are requested. No training or label access occurs here.
    Exhaustion has first terminal priority; configured rules then keep their order.
    """
    scorer = EntropyReductionScorer() if scorer is None else scorer
    policy = GreedyPolicy() if policy is None else policy
    stopping = AnyStop((NoModalitiesStop(),)) if stopping is None else stopping
    for name in initial_state.available_modalities:
        cost_model.cost(name)
    prediction = probability_vector(predictor.predict(initial_state.sample_id, frozenset(initial_state.acquired_modalities)))
    if len(prediction) != len(initial_state.probabilities) or any(abs(a-b) > 1e-12 for a,b in zip(prediction, initial_state.probabilities)):
        raise ValueError('Initial state probabilities differ from predictor for acquired evidence')
    state = initial_state
    trace = []
    while True:
        # Validate every budget, even when another criterion has already fired.
        candidates = stopping.candidates(state, cost_model)
        reason = 'no_modalities' if not state.remaining_modalities else stopping.reason(state, cost_model)
        if reason is not None:
            trace.append(DecisionRecord(state, (), None, reason))
            return EpisodeResult(initial_state, state, tuple(trace), reason)
        if not candidates:
            reason = 'no_eligible_modalities'
            trace.append(DecisionRecord(state, (), None, reason))
            return EpisodeResult(initial_state, state, tuple(trace), reason)
        utilities = scorer.score(state, candidates, cost_model, predictor)
        if set(utilities) != set(candidates):
            raise ValueError('Scorer must return exactly one utility per eligible candidate')
        selection = policy.select(state, utilities)
        snapshot = tuple((name, float(utilities[name])) for name in sorted(utilities))
        if selection is None:
            reason = 'non_positive_utility' if policy.min_utility == 0 else 'utility_threshold'
            trace.append(DecisionRecord(state, snapshot, None, reason))
            return EpisodeResult(initial_state, state, tuple(trace), reason)
        if selection.modality not in candidates:
            raise ValueError('Policy selected a modality outside eligible candidates')
        trace.append(DecisionRecord(state, snapshot, selection.modality, None))
        forecast = predictor.predict(state.sample_id, frozenset((*state.acquired_modalities, selection.modality)))
        state = state.acquire(selection.modality, forecast, cost_model.cost(selection.modality))
