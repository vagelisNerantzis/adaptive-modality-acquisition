"""Composable confidence/budget constraints, including exact budget boundaries."""
import pytest
from adaptive_acquisition.state.acquisition_state import AcquisitionState
from adaptive_acquisition.stopping.rules import ConfidenceThresholdStop, BudgetStop, NoModalitiesStop, AnyStop


def test_confidence_threshold_includes_equality(case_factory):
    state, costs, _ = case_factory()
    rule = ConfidenceThresholdStop(.8)
    assert rule.reason(state, costs) is None
    assert rule.reason(state.acquire("b", (.8, .2), 1), costs) == "confidence"


def test_budget_stops_only_when_no_candidate_is_affordable(case_factory):
    state, costs, _ = case_factory()
    rule = BudgetStop(1.)
    assert rule.reason(state, costs) is None
    assert rule.candidates(state, costs) == ("b",)
    assert rule.reason(state.acquire("b", (.8, .2), 1.), costs) == "budget"


def test_no_modality_stop(case_factory):
    state, costs, _ = case_factory()
    rule = NoModalitiesStop()
    assert rule.reason(state, costs) is None
    exhausted = state.acquire("b", (.8, .2), 1).acquire("c", (.96, .04), 2)
    assert rule.reason(exhausted, costs) == "no_modalities"


@pytest.mark.parametrize("threshold", [0., 1.1, float("nan")])
def test_invalid_confidence_threshold(threshold):
    with pytest.raises(ValueError):
        ConfidenceThresholdStop(threshold)


@pytest.mark.parametrize("budget", [-1., float("nan"), float("inf")])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        BudgetStop(budget)


def test_composition_intersects_budgets_and_retains_reason_priority(case_factory):
    state, costs, _ = case_factory()
    rules = AnyStop((ConfidenceThresholdStop(.5), BudgetStop(1.), BudgetStop(2.)))
    assert rules.candidates(state, costs) == ("b",)
    assert rules.reason(state, costs) == "confidence"
