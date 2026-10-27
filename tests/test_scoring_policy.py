"""Explicit lookahead beliefs, utility trade-offs and lexical greedy decisions."""
import pytest
from adaptive_acquisition.costs.models import FixedCostModel
from adaptive_acquisition.utility.entropy import entropy
from adaptive_acquisition.utility.scoring import EntropyReductionScorer
from adaptive_acquisition.policies.greedy import GreedyPolicy
from adaptive_acquisition.simulation.lookup import LookupPredictor


def test_higher_entropy_reduction_is_preferred(case_factory):
    state, costs, predictor = case_factory()
    scores = EntropyReductionScorer(cost_weight=0.).score(state, state.remaining_modalities, costs, predictor)
    assert scores["b"] > scores["c"]
    assert scores["b"] == pytest.approx(entropy(state.probabilities) - entropy([.8, .2]))


def test_high_acquisition_cost_reduces_utility(case_factory):
    state, _, predictor = case_factory()
    costs = FixedCostModel({"a": 0, "b": 10, "c": 0})
    scores = EntropyReductionScorer(cost_weight=.15).score(state, ("b", "c"), costs, predictor)
    assert scores["b"] < scores["c"]


def test_zero_cost_weight_ignores_costs(case_factory):
    state, costs, predictor = case_factory()
    expensive = FixedCostModel({"a": 0, "b": 100, "c": 200})
    scorer = EntropyReductionScorer(cost_weight=0)
    assert scorer.score(state, ("b", "c"), costs, predictor) == scorer.score(state, ("b", "c"), expensive, predictor)


def test_scoring_is_deterministic(case_factory):
    state, costs, predictor = case_factory()
    scorer = EntropyReductionScorer(.05)
    assert scorer.score(state, ("b", "c"), costs, predictor) == scorer.score(state, ("b", "c"), costs, predictor)


def test_missing_lookup_subset_is_rejected(case_factory):
    _, _, predictor = case_factory()
    with pytest.raises((ValueError, KeyError)):
        predictor.predict("sample_a", frozenset({"b"}))


def test_lookup_class_dimensions_must_match():
    with pytest.raises(ValueError):
        LookupPredictor("s", {frozenset(): [.5, .5], frozenset({"a"}): [.2, .3, .5]})


def test_lookup_rejects_wrong_physical_sample(case_factory):
    _, _, predictor = case_factory()
    with pytest.raises(ValueError):
        predictor.predict("different", frozenset({"a"}))


def test_greedy_selects_highest_utility(case_factory):
    state, _, _ = case_factory()
    decision = GreedyPolicy().select(state, {"b": .1, "c": .3})
    assert decision.modality == "c" and decision.score == pytest.approx(.3)


def test_greedy_tie_break_is_lexical_and_order_independent(case_factory):
    state, _, _ = case_factory()
    policy = GreedyPolicy()
    a = policy.select(state, {"c": .4, "b": .4})
    b = policy.select(state, {"b": .4, "c": .4})
    assert a.modality == b.modality == "b"


def test_empty_candidate_selection_fails(case_factory):
    state, _, _ = case_factory()
    with pytest.raises(ValueError):
        GreedyPolicy().select(state, {})


def test_nonpositive_utility_stops_unless_explicitly_disabled(case_factory):
    state, _, _ = case_factory()
    assert GreedyPolicy().select(state, {"b": 0., "c": -.1}) is None
    assert GreedyPolicy(min_utility=None).select(state, {"b": -.2}).modality == "b"


def test_no_selection_after_stopping_rule_is_reached(case_factory):
    state, _, _ = case_factory()
    assert GreedyPolicy().select(state, {}, stop_reason="confidence") is None


def test_policy_rejects_unknown_or_acquired_actions(case_factory):
    state, _, _ = case_factory()
    for utilities in ({"unknown": 1.}, {"a": 1.}):
        with pytest.raises(ValueError):
            GreedyPolicy().select(state, utilities)
