"""End-to-end synthetic acquisition episodes and sample-aware evaluation."""
import importlib.metadata
import pytest
import adaptive_acquisition
from adaptive_acquisition.state.acquisition_state import AcquisitionState
from adaptive_acquisition.costs.models import FixedCostModel
from adaptive_acquisition.utility.entropy import entropy
from adaptive_acquisition.utility.scoring import EntropyReductionScorer
from adaptive_acquisition.policies.greedy import GreedyPolicy
from adaptive_acquisition.stopping.rules import ConfidenceThresholdStop, BudgetStop, AnyStop
from adaptive_acquisition.simulation.lookup import LookupPredictor
from adaptive_acquisition.simulation.episode import run_episode
from adaptive_acquisition.evaluation.metrics import episode_metrics, batch_summary


def execute(case, *, budget=3., threshold=.9):
    state, costs, predictor = case
    return run_episode(state, predictor, costs, scorer=EntropyReductionScorer(.05),
                       stopping=AnyStop((ConfidenceThresholdStop(threshold), BudgetStop(budget))))


def test_expected_sequence_cost_confidence_and_terminal_history(case_factory):
    result = execute(case_factory())
    assert result.selected_modalities == ("b", "c")
    assert result.n_steps == 2 and result.total_cost == 3.
    assert result.final_probabilities == (.96, .04)
    assert result.final_confidence == pytest.approx(.96)
    assert result.stop_reason == "no_modalities"
    assert result.decision_history[-1].stop_reason == result.stop_reason
    assert len(result.decision_history) == 3


def test_repeated_execution_is_deterministic(case_factory):
    a, b = execute(case_factory()), execute(case_factory())
    assert a.selected_modalities == b.selected_modalities
    assert a.final_probabilities == b.final_probabilities
    assert a.total_cost == b.total_cost and a.stop_reason == b.stop_reason


def test_budget_filter_excludes_expensive_candidate_before_scoring():
    state = AcquisitionState("s", ("a", "b", "c"), (.5, .5), acquired_modalities=("a",))
    costs = FixedCostModel({"a": 0, "b": 1, "c": 100})
    # No a+c entry exists: filtering must prevent its lookup entirely.
    predictor = LookupPredictor("s", {frozenset({"a"}): (.5, .5), frozenset({"a", "b"}): (.8, .2)})
    result = run_episode(state, predictor, costs, scorer=EntropyReductionScorer(0), stopping=AnyStop((BudgetStop(1.),)))
    assert result.selected_modalities == ("b",)
    assert result.total_cost == 1. and result.stop_reason == "budget"


def test_initial_confidence_prevents_any_candidate_lookup():
    state = AcquisitionState("s", ("a", "b"), (.95, .05), acquired_modalities=("a",))
    predictor = LookupPredictor("s", {frozenset({"a"}): (.95, .05)})
    result = run_episode(state, predictor, FixedCostModel({"a": 0, "b": 1}), stopping=AnyStop((ConfidenceThresholdStop(.9),)))
    assert result.selected_modalities == () and result.n_steps == 0
    assert result.stop_reason == "confidence" and result.total_cost == 0


def test_nonpositive_utilities_stop_episode():
    state = AcquisitionState("s", ("a", "b"), (.8, .2), acquired_modalities=("a",))
    predictor = LookupPredictor("s", {frozenset({"a"}): (.8, .2), frozenset({"a", "b"}): (.5, .5)})
    result = run_episode(state, predictor, FixedCostModel({"a": 0, "b": 1}))
    assert result.stop_reason == "non_positive_utility"
    assert not result.selected_modalities and result.total_cost == 0


def test_missing_post_acquisition_prediction_fails():
    state = AcquisitionState("s", ("a", "b"), (.5, .5), acquired_modalities=("a",))
    predictor = LookupPredictor("s", {frozenset({"a"}): (.5, .5)})
    with pytest.raises((ValueError, KeyError)):
        run_episode(state, predictor, FixedCostModel({"a": 0, "b": 1}))


def test_episode_metrics_have_precise_denominators(case_factory):
    result = execute(case_factory())
    values = episode_metrics(result, true_label=0)
    gain = entropy([.55, .45]) - entropy([.96, .04])
    assert values["total_cost"] == 3
    assert values["acquisition_steps"] == 2
    assert values["n_acquired_modalities"] == 3
    assert values["correct"] is True
    assert values["final_confidence"] == pytest.approx(.96)
    assert values["final_entropy"] == pytest.approx(entropy([.96, .04]))
    assert values["entropy_reduction"] == pytest.approx(gain)
    assert values["entropy_reduction_per_cost"] == pytest.approx(gain / 3)


def test_batch_summary_accuracy_and_means(case_factory):
    first = execute(case_factory("s1"))
    second = execute(case_factory("s2"), budget=1.)
    values = batch_summary([first, second], true_labels=[0, 1])
    assert values["n_samples"] == 2
    assert values["mean_total_cost"] == pytest.approx(2.)
    assert values["mean_acquisition_steps"] == pytest.approx(1.5)
    assert values["accuracy"] == pytest.approx(.5)
    assert values["mean_final_confidence"] == pytest.approx(.88)


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError):
        batch_summary([])


def test_repeated_physical_ids_are_not_independent_batch_samples(case_factory):
    result = execute(case_factory())
    with pytest.raises(ValueError):
        batch_summary([result, result])


def test_package_and_distribution_versions_match():
    assert adaptive_acquisition.__version__ == "0.1.0"
    assert importlib.metadata.version("adaptive-modality-acquisition") == "0.1.0"



def test_free_improvement_has_no_infinite_efficiency_metric():
    state = AcquisitionState("s", ("a", "b"), (.5, .5), acquired_modalities=("a",), total_cost=5.)
    predictor = LookupPredictor("s", {frozenset({"a"}): (.5, .5), frozenset({"a", "b"}): (.95, .05)})
    result = run_episode(state, predictor, FixedCostModel({"a": 5, "b": 0}),
                         scorer=EntropyReductionScorer(0), stopping=AnyStop((BudgetStop(5.),)))
    values = episode_metrics(result)
    assert values["total_cost"] == 5
    assert values["entropy_reduction"] > 0
    assert values["entropy_reduction_per_cost"] is None
    assert values["correct"] is None


def test_ground_truth_must_index_the_declared_class_vector(case_factory):
    result = execute(case_factory())
    with pytest.raises(ValueError):
        episode_metrics(result, true_label=2)


def test_unrepresentable_positive_cost_efficiency_is_rejected():
    state = AcquisitionState("tiny_cost", ("a", "b"), (.5, .5), ("a",))
    predictor = LookupPredictor("tiny_cost", {frozenset({"a"}): (.5, .5), frozenset({"a", "b"}): (.99, .01)})
    result = run_episode(state, predictor, FixedCostModel({"a": 0., "b": 1e-320}), scorer=EntropyReductionScorer(0))
    with pytest.raises(ValueError, match="finite float"):
        episode_metrics(result)


def test_manually_constructed_result_cannot_decrease_cost():
    from adaptive_acquisition.simulation.episode import EpisodeResult
    initial = AcquisitionState("s", ("a",), (.5, .5), ("a",), total_cost=2.)
    final = AcquisitionState("s", ("a",), (.5, .5), ("a",), total_cost=1.)
    with pytest.raises(ValueError, match="Final cost"):
        EpisodeResult(initial, final, (), "no_modalities")


def test_batch_mean_large_costs_has_no_overflow():
    results = []
    for sample_id in ("large_1", "large_2"):
        state = AcquisitionState(sample_id, ("a",), (.8, .2), ("a",), total_cost=1e308)
        predictor = LookupPredictor(sample_id, {frozenset({"a"}): (.8, .2)})
        results.append(run_episode(state, predictor, FixedCostModel({"a": 1e308})))
    summary = batch_summary(results)
    assert summary["mean_total_cost"] == 1e308
