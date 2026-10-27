"""Physical sample identity and immutable, auditable transitions."""
from dataclasses import FrozenInstanceError
import pytest
from adaptive_acquisition.state.acquisition_state import AcquisitionState


def test_initial_state_distinguishes_known_and_candidate_modalities(case_factory):
    state, _, _ = case_factory()
    assert state.sample_id == "sample_a"
    assert state.acquired_modalities == ("a",)
    assert state.remaining_modalities == ("b", "c")
    assert state.step_index == 0 and state.total_cost == 0


def test_transition_preserves_original_and_records_history(case_factory):
    state, _, _ = case_factory()
    new = state.acquire("b", (.8, .2), 1.)
    assert state.remaining_modalities == ("b", "c") and not state.history
    assert new.remaining_modalities == ("c",)
    assert new.sample_id == state.sample_id
    assert new.total_cost == 1 and new.step_index == 1
    assert new.probabilities == (.8, .2)
    event = new.history[0]
    assert event.modality == "b" and event.cost == 1 and event.total_cost == 1
    assert event.step_index == 1 and event.probabilities == (.8, .2)
    with pytest.raises(FrozenInstanceError):
        new.total_cost = 100


def test_multistep_history_and_accumulated_cost(case_factory):
    state, _, _ = case_factory()
    result = state.acquire("b", (.8, .2), 1).acquire("c", (.96, .04), 2)
    assert result.acquired_modalities == ("a", "b", "c")
    assert result.remaining_modalities == ()
    assert result.step_index == 2 and result.total_cost == 3
    assert [event.total_cost for event in result.history] == [1, 3]


def test_duplicate_acquisition_is_rejected(case_factory):
    state, _, _ = case_factory()
    with pytest.raises(ValueError):
        state.acquire("a", (.8, .2), 1)


def test_unknown_modality_acquisition_is_rejected(case_factory):
    state, _, _ = case_factory()
    with pytest.raises(ValueError):
        state.acquire("unknown", (.8, .2), 1)


@pytest.mark.parametrize("modalities", [("a", ""), ("a", " b"), ("a", "a")])
def test_invalid_modality_names_or_duplicates(modalities):
    with pytest.raises((ValueError, TypeError)):
        AcquisitionState("s", modalities, (.5, .5))


@pytest.mark.parametrize("sample_id", ["", " s", None])
def test_invalid_physical_identity(sample_id):
    with pytest.raises((ValueError, TypeError)):
        AcquisitionState(sample_id, ("a",), (.5, .5))


def test_initial_acquired_modalities_must_be_available():
    with pytest.raises(ValueError):
        AcquisitionState("s", ("a",), (.5, .5), acquired_modalities=("b",))



def test_duplicate_initial_acquired_modalities_are_rejected():
    with pytest.raises(ValueError):
        AcquisitionState("s", ("a", "b"), (.5, .5), acquired_modalities=("a", "a"))


def test_transition_rejects_negative_cost(case_factory):
    state, _, _ = case_factory()
    with pytest.raises(ValueError):
        state.acquire("b", (.8, .2), -1.)
